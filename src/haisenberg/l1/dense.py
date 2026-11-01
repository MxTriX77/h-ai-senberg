"""Dense (embedding) search over the Layer 1 encyclopedia for queries its BM25 cannot match against English text,
fused with the keyword search (KNOWLEDGE=crosslingual).

The dense store is a folder built by ingest.embed_l1, laid out like the literature's (haisenberg.l2.dense): one
L2-normalised float16 row per article (row = article id - 1) embedding `article_text`, an exact index, and
`meta.json` (embedding model, query prefix, the Layer 1 build it belongs to, whether the build is complete).
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, fields

from haisenberg.config import DenseConfig
from haisenberg.l1.store import ArticleRecord, L1Store
from haisenberg.l2.dense import DenseIndex, Embed

REBUILD_COMMAND = "uv run python -m ingest.embed_l1"
TEXT_CHARS = 2000  # an article is embedded by its first characters: the topic comes first, and it bounds the build
_LETTERS = re.compile(r"[^\W\d_]+")


def article_text(title: str, lead: str, infobox: str) -> str:
    """What is embedded for an article: its title, a newline, then its lead (its infobox when the lead is empty),
    cut to TEXT_CHARS."""
    return f"{title}\n{lead or infobox}"[:TEXT_CHARS]


def non_latin(query: str) -> bool:
    """Whether the query has a word of two or more letters none of which is Latin: 'теплота' and '水素' are such
    words, 'ΔH', 'λ', 'pH' and 'C₂H₆O' are not."""
    return any(len(w) >= 2 and not any(_latin(c) for c in w) for w in _LETTERS.findall(query))


def _latin(letter: str) -> bool:
    return letter.isascii() or unicodedata.name(letter, "").startswith("LATIN")


@dataclass
class DenseArticleRecord(ArticleRecord):
    """An article the dense ranking found: `similarity` is its cosine with the query (the term-share filter keeps it
    from retrieval.dense.min_similarity on). One that only the dense ranking found has the score 1 - similarity,
    never the 0 of a title match."""

    similarity: float = 0.0


class DenseEncyclopedia:
    """The encyclopedia search of the crosslingual knowledge set: `search_encyclopedia` returns the keyword search's
    records, but for a query with a non-Latin word it fuses the keyword ranking and the dense ranking by reciprocal
    rank fusion."""

    def __init__(self, store: L1Store, index: DenseIndex, embed: Embed, query_prefix: str, cfg: DenseConfig):
        self.store, self.index, self.embed = store, index, embed
        self.query_prefix, self.cfg = query_prefix, cfg

    def search_encyclopedia(self, query: str, k: int | None = None, domain: str | None = None) -> list[ArticleRecord]:
        if not non_latin(query):
            return self.store.search_encyclopedia(query, k, domain)
        k, c = k or self.store.top_k, self.cfg
        keyword = dict(self.store.ranked_articles(query, c.candidates, domain))  # article id -> record, best first
        hits = self.index.search(self.embed([self.query_prefix + query])[0], c.rerank)
        if domain:
            keep = set(self.store.in_domain([aid for aid, _ in hits], domain))
            hits = [h for h in hits if h[0] in keep]
        similarity = dict(hits[: c.candidates])
        scores: dict[int, float] = defaultdict(float)
        for ranking in (list(keyword), list(similarity)):
            for rank, aid in enumerate(ranking, 1):
                scores[aid] += 1 / (c.rrf_k + rank)
        best = sorted(scores, key=lambda aid: (-scores[aid], aid))[:k]
        dense_only = [aid for aid in best if aid not in keyword]
        found = {**keyword, **dict(zip(dense_only, self.store.articles(dense_only)))}
        records = []
        for aid in best:
            record = found[aid]
            if aid in similarity:
                score = record.score if aid in keyword else round(1 - similarity[aid], 4)
                values = {f.name: getattr(record, f.name) for f in fields(record)} | {"score": score}
                record = DenseArticleRecord(**values, similarity=round(similarity[aid], 4))
            records.append(record)
        return records
