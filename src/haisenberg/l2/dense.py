"""Dense (embedding) search over the Layer 2 literature store, fused with its BM25 ranking (KNOWLEDGE=hybrid).

The dense store is a folder built by ingest.embed_l2: `vectors.f16.npy` (one L2-normalised float16 row per passage,
row = passage id - 1), `index.faiss` (FAISS, inner product, ids = rows) and `meta.json` (embedding model, query
prefix, the literature store build it belongs to, whether the build is complete).
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path

import faiss
import numpy as np

from haisenberg.config import DenseConfig
from haisenberg.l2.store import L2Error, L2Store, PassageRecord, content_words

REBUILD_COMMAND = "uv run python -m ingest.embed_l2"
VECTORS, INDEX, META = "vectors.f16.npy", "index.faiss", "meta.json"

Embed = Callable[[list[str]], np.ndarray]  # texts -> one L2-normalised float32 row each


def passage_text(title: str, heading: str, text: str) -> str:
    """What is embedded for a passage: its title and heading path give a short passage its context."""
    return f"{title} › {heading}\n{text}" if heading else f"{title}\n{text}"


@dataclass
class DensePassageRecord(PassageRecord):
    """A passage the dense ranking found: `similarity` is its cosine with the query (the term-share filter keeps it
    from retrieval.dense.min_similarity on)."""

    similarity: float = 0.0


class DenseIndex:
    """A dense store folder, read-only, checked against the embedding model and the store whose rows it holds (the
    literature store, or the Layer 1 store with `name` and `rebuild` of the encyclopedia's vectors)."""

    def __init__(self, path: Path | str, model_id: str, store: L2Store, nprobe: int,
                 name: str = "literature", rebuild: str = REBUILD_COMMAND):
        path = Path(path)
        meta_path = path / META
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        if not meta.get("complete"):
            raise L2Error(f"dense {name} index at {path} is missing or unfinished; build it with: {rebuild}")
        if meta["model_id"] != model_id:
            raise L2Error(f"dense {name} index at {path} was built with the embedding model {meta['model_id']!r},"
                          f" the config uses {model_id!r}; rebuild it with: {rebuild} --restart")
        built_for = {k: meta.get(k) for k in store.identity()}
        if built_for != store.identity():
            raise L2Error(f"dense {name} index at {path} was built for the {name} store {built_for},"
                          f" not for {store.path} {store.identity()}; rebuild it with: {rebuild} --restart")
        self.path = path
        # One query at a time: a single thread searches as fast, and idle OpenMP threads spin on every core,
        # slowing the CPU embedding server ~20x (0.6-1.4 s instead of ~50 ms per query on Windows).
        faiss.omp_set_num_threads(1)
        self.index = faiss.read_index(str(path / INDEX))
        if (ivf := faiss.try_extract_index_ivf(self.index)) is not None:
            ivf.nprobe = nprobe
        self.vectors = np.load(path / VECTORS, mmap_mode="r")

    def search(self, vector: np.ndarray, rerank: int) -> list[tuple[int, float]]:
        """(passage id, cosine) of the `rerank` nearest index hits, re-scored exactly with their stored vectors,
        best first (ties: lower id first)."""
        _, hits = self.index.search(vector[None, :].astype(np.float32), rerank)
        rows = np.unique(hits[0][hits[0] >= 0])  # sorted: the memmap is read in file order
        scores = self.vectors[rows].astype(np.float32) @ vector.astype(np.float32)
        return [(int(rows[i]) + 1, float(scores[i])) for i in np.argsort(-scores, kind="stable")]


class HybridSearch:
    """Literature search by reciprocal rank fusion of the BM25 and the dense ranking; `search` returns the same
    records as `L2Store.search` with the fused score (higher is better), those of the dense ranking as
    DensePassageRecord with their similarity."""

    def __init__(self, store: L2Store, index: DenseIndex, embed: Embed, query_prefix: str, cfg: DenseConfig):
        self.store, self.index, self.embed = store, index, embed
        self.query_prefix, self.cfg = query_prefix, cfg

    def search(self, query: str, k: int = 5, domain: str | None = None) -> list[PassageRecord]:
        if not content_words(query):  # as the keyword search: nothing to look for
            return []
        c = self.cfg
        bm25 = self.store.bm25_ids(query, c.candidates, domain)
        hits = self.index.search(self.embed([self.query_prefix + query])[0], c.rerank)
        if domain:
            keep = set(self.store.in_domain([pid for pid, _ in hits], domain))
            hits = [h for h in hits if h[0] in keep]
        similarity = dict(hits[: c.candidates])
        scores: dict[int, float] = defaultdict(float)
        for ranking in (bm25, list(similarity)):
            for rank, pid in enumerate(ranking, 1):
                scores[pid] += 1 / (c.rrf_k + rank)
        best = sorted(scores, key=lambda pid: (-scores[pid], pid))[:k]
        records = []
        for pid, record in zip(best, self.store.passages(best)):
            record.score = round(scores[pid], 6)
            if pid in similarity:
                record = DensePassageRecord(**{f.name: getattr(record, f.name) for f in fields(record)},
                                            similarity=round(similarity[pid], 4))
            records.append(record)
        return records
