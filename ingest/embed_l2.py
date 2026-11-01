"""Build the dense index of the Layer 2 literature store: one embedding per passage, then a FAISS index.

    uv run python -m ingest.embed_l2 [--db FILE] [--out DIR] [--restart]

Every passage is embedded in id order, as `title › heading` + newline + text, through the OpenAI-compatible
embeddings backend of the `embedding` config. Its own llama-server is started here unless one already answers.
It runs on the CPU unless EMBED_DEVICE and EMBED_CUDA_DEVICES give it a GPU. In WSL, take `flock /tmp/hai-gpu.lock`
for that, and only by agreement with the testing session. The vectors and their count in meta.json are saved
about every SAVE_S seconds, so a stopped build continues where it stopped. The index is built on the CPU after
the last passage, and meta.json says `complete` only then. The literature store is opened read-only. The Layer 1
store is never opened.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import faiss
import numpy as np

from haisenberg.config import ConfigError, load_config
from haisenberg.l2.dense import INDEX, META, VECTORS, Embed, passage_text
from haisenberg.l2.store import L2Error, L2Store
from haisenberg.launcher import LauncherError, Supervisor, embedding_view
from haisenberg.llm import EmbeddingClient, UpstreamError
from ingest.l1base import BuildError, sha256

SAVE_S = 30  # seconds between saves of the vectors and meta.json
WINDOW = 4  # requests per worker between two checks for a save
SHORTEN = 0.8  # a passage the backend rejects (HTTP 400: longer than its context) is cut to this share, then resent
MAX_SHORTEN = 8
TRAIN_ROWS = 200_000  # rows the index is trained on (IVF4096 wants at least 39 x 4096)
ADD_ROWS = 100_000  # rows added to the index at a time


def build(db: Path, out: Path, embed: Embed, model: dict, factory: str, batch: int, workers: int,
          restart: bool = False, log: Callable[[str], None] = print) -> dict:
    """Embed every passage of the store at db into out, then index them. `model`: model_id, model_sha256 and
    query_prefix, recorded in meta.json. Returns the build report."""
    store = L2Store(db)
    try:
        first, last, n = store.con.execute("SELECT min(id), max(id), count(*) FROM passage").fetchone()
        if (first, last) != (1, n):
            raise BuildError(f"{db}: passage ids must run from 1 to {n} without gaps (found {first}..{last})")

        def texts(start: int, stop: int) -> list[str]:
            return [passage_text(*row) for row in store.con.execute(
                "SELECT title, heading, text FROM passage WHERE id > ? AND id <= ? ORDER BY id", (start, stop))]

        return embed_rows(out, n, texts, {**model, **store.identity()}, embed, factory, batch, workers, restart, log,
                          "passages")
    finally:
        store.close()


def embed_rows(out: Path, n: int, texts: Callable[[int, int], list[str]], want: dict, embed: Embed, factory: str,
               batch: int, workers: int, restart: bool = False, log: Callable[[str], None] = print,
               rows_name: str = "rows") -> dict:
    """Embed rows 1..n (`texts(start, stop)`: the texts of rows start+1..stop) into out, resuming a stopped build,
    then index them. `want`: the model and the identity of the store the rows come from, recorded in meta.json and
    checked on resume. Returns the build report."""
    t0 = time.monotonic()
    meta = _read_meta(out) if not restart else {}
    if meta and any(meta.get(k) != v for k, v in want.items()):
        have = {k: meta.get(k) for k in want}
        raise BuildError(f"{out} holds vectors of another model or store build ({have}, now {want});"
                         " run again with --restart to discard them")
    if not meta:
        for name in (VECTORS, INDEX, META):
            (out / name).unlink(missing_ok=True)
        meta = {**want, "index": factory, "dim": None, "done_rows": 0, "shortened": 0, "complete": False}
    if meta["complete"]:
        log(f"{out} is complete; nothing to do (--restart builds it again)")
        return _report(meta, out, t0)
    out.mkdir(parents=True, exist_ok=True)
    vectors = np.load(out / VECTORS, mmap_mode="r+") if meta["dim"] else None
    done, start, run_start = meta["done_rows"], meta["done_rows"], time.monotonic()
    saved = run_start
    with ThreadPoolExecutor(workers) as pool:
        while done < n:
            stop = min(n, done + batch * workers * WINDOW)
            part = texts(done, stop)
            parts = list(pool.map(lambda b: _embed(embed, b), [part[i:i + batch] for i in range(0, len(part), batch)]))
            rows = np.concatenate([p[0] for p in parts])
            if vectors is None:
                meta["dim"] = int(rows.shape[1])
                vectors = np.lib.format.open_memmap(out / VECTORS, mode="w+", dtype=np.float16, shape=(n, meta["dim"]))
            vectors[done:stop] = rows
            meta["shortened"] += sum(p[1] for p in parts)
            done = stop
            if time.monotonic() - saved >= SAVE_S or done == n:
                vectors.flush()
                meta["done_rows"] = done
                _write_meta(out, meta)
                saved = time.monotonic()
                rate = (done - start) / max(saved - run_start, 1e-9)
                left = timedelta(seconds=round((n - done) / rate)) if rate else "?"
                log(f"embedded {done:,}/{n:,} {rows_name}, {rate:.1f}/s, {left} left, {meta['shortened']} shortened")
    log(f"indexing {n:,} vectors ({factory}) on the CPU")
    try:
        faiss.write_index(_index(vectors, factory), str(out / INDEX))
    except RuntimeError as e:  # FAISS: e.g. fewer vectors than the factory's lists
        raise BuildError(f"index {factory!r} (retrieval.dense.index): {e}") from e
    meta["complete"] = True
    _write_meta(out, meta)
    return _report(meta, out, t0)


def _embed(embed: Embed, texts: list[str]) -> tuple[np.ndarray, int]:
    """(rows, passages shortened). A rejected batch is sent again passage by passage, and a rejected passage is
    cut at its end until the backend takes it."""
    try:
        return embed(texts).astype(np.float16), 0
    except UpstreamError as e:
        if e.status != 400:
            raise
    rows, shortened = [], 0
    for text in texts:
        for attempt in range(MAX_SHORTEN + 1):
            try:
                rows.append(embed([text])[0])
                break
            except UpstreamError as e:
                if e.status != 400 or attempt == MAX_SHORTEN:
                    raise
                text = text[: int(len(text) * SHORTEN)]
        shortened += attempt > 0
    return np.stack(rows).astype(np.float16), shortened


def _index(vectors: np.ndarray, factory: str) -> faiss.Index:
    """An inner-product index over every row (id = row), trained on an evenly spread sample of rows."""
    n, dim = vectors.shape
    index = faiss.index_factory(dim, factory, faiss.METRIC_INNER_PRODUCT)
    if not index.is_trained:
        sample = np.linspace(0, n - 1, min(n, TRAIN_ROWS)).astype(np.int64)
        index.train(np.asarray(vectors[np.unique(sample)], dtype=np.float32))
    for i in range(0, n, ADD_ROWS):
        index.add(np.asarray(vectors[i:i + ADD_ROWS], dtype=np.float32))
    return index


def _read_meta(out: Path) -> dict:
    path = out / META
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _write_meta(out: Path, meta: dict) -> None:
    tmp = out / (META + ".tmp")
    tmp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    os.replace(tmp, out / META)


def _report(meta: dict, out: Path, t0: float) -> dict:
    size = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    return {**meta, "size_mb": round(size / 1e6, 1), "seconds": round(time.monotonic() - t0, 1)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, help="default: the literature store in retrieval.stores")
    parser.add_argument("--out", type=Path, help="default: the literature_dense store in retrieval.stores")
    parser.add_argument("--restart", action="store_true", help="discard vectors already in the folder")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    supervisor = None
    try:
        cfg = load_config()
        e = cfg.embedding
        db = args.db or cfg.retrieval.path("literature")
        out = args.out or cfg.retrieval.path("literature_dense")
        model = {"model_id": e.model_id, "query_prefix": e.query_prefix,
                 "model_sha256": sha256(e.server.model_path) if e.server.model_path.is_file() else None}
        if e.server.enabled:
            supervisor = Supervisor(embedding_view(cfg))
            supervisor.start()
        client = EmbeddingClient(e)
        report = build(db, out, client.embed, model, cfg.retrieval.dense.index, e.batch, e.workers, args.restart,
                       log=lambda line: print(line, flush=True))
    except (BuildError, ConfigError, L2Error, LauncherError, UpstreamError, OSError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    finally:
        if supervisor is not None:
            supervisor.stop()
    print(f"built {out}")
    for k, v in report.items():
        print(f"  {k:<16} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
