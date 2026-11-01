"""Build the dense index of the Layer 1 encyclopedia: one embedding per article, then an exact FAISS index.

    uv run python -m ingest.embed_l1 [--db FILE] [--out DIR] [--restart]

Every article is embedded in id order as `article_text` (title, newline, lead, cut to its first characters) through
the OpenAI-compatible embeddings backend of the `embedding` config, with the loop of ingest.embed_l2: its own
llama-server is started unless one already answers, on the CPU unless EMBED_DEVICE and EMBED_CUDA_DEVICES give it
a GPU (in WSL under `flock /tmp/hai-gpu.lock`, and only by agreement with the testing session), and a stopped build
continues where it stopped. The Layer 1 store is opened read-only. No literature store is opened.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path

from haisenberg.config import ConfigError, load_config
from haisenberg.l1.dense import article_text
from haisenberg.l1.store import L1Error, L1Store
from haisenberg.l2.dense import Embed
from haisenberg.launcher import LauncherError, Supervisor, embedding_view
from haisenberg.llm import EmbeddingClient, UpstreamError
from ingest.embed_l2 import embed_rows
from ingest.l1base import BuildError, sha256

INDEX_FACTORY = "Flat"  # exact inner product: ~37.5k articles need no trained index, and a query scans them in ms


def build(db: Path, out: Path, embed: Embed, model: dict, batch: int, workers: int, restart: bool = False,
          log: Callable[[str], None] = print) -> dict:
    """Embed every article of the Layer 1 store at db into out, then index them. `model`: model_id, model_sha256
    and query_prefix, recorded in meta.json. Returns the build report."""
    store = L1Store(db)
    try:
        first, last, n = store.con.execute("SELECT min(id), max(id), count(*) FROM article").fetchone()
        if (first, last) != (1, n):
            raise BuildError(f"{db}: article ids must run from 1 to {n} without gaps (found {first}..{last})")

        def texts(start: int, stop: int) -> list[str]:
            return [article_text(*row) for row in store.con.execute(
                "SELECT title, lead, infobox FROM article WHERE id > ? AND id <= ? ORDER BY id", (start, stop))]

        return embed_rows(out, n, texts, {**model, **store.identity()}, embed, INDEX_FACTORY, batch, workers, restart,
                          log, "articles")
    finally:
        store.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, help="default: the encyclopedia store in retrieval.stores")
    parser.add_argument("--out", type=Path, help="default: the encyclopedia_dense store in retrieval.stores")
    parser.add_argument("--restart", action="store_true", help="discard vectors already in the folder")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    supervisor = None
    try:
        cfg = load_config()
        e = cfg.embedding
        db = args.db or cfg.retrieval.path("encyclopedia")
        out = args.out or cfg.retrieval.path("encyclopedia_dense")
        model = {"model_id": e.model_id, "query_prefix": e.query_prefix,
                 "model_sha256": sha256(e.server.model_path) if e.server.model_path.is_file() else None}
        if e.server.enabled:
            supervisor = Supervisor(embedding_view(cfg))
            supervisor.start()
        client = EmbeddingClient(e)
        report = build(db, out, client.embed, model, e.batch, e.workers, args.restart,
                       log=lambda line: print(line, flush=True))
    except (BuildError, ConfigError, L1Error, LauncherError, UpstreamError, OSError) as err:
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
