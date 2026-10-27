"""Read-only access to the Layer 2 literature store, returning passages that carry their own citation."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from haisenberg.l1.store import STOPWORDS, Citation, Record, _cut
from haisenberg.l2.schema import APPLICATION_ID, SCHEMA_VERSION

REBUILD_COMMAND = "uv run python -m ingest.build_l2"
BM25_WEIGHTS = "4.0, 2.0, 1.0, 0.0"  # title, heading, text, domains (a filter only)
_WORD = re.compile(r"[^\W_]+")  # FTS5's unicode61 also splits at '_'


class L2Error(Exception):
    """The store is missing, outdated or not a literature store; the message says how to fix it."""


@dataclass
class PassageRecord(Record):
    title: str  # the document's title
    heading: str  # headings above the passage inside the page, ' › '-separated ('' at the top)
    path: str
    text: str
    score: float  # BM25 (lower is better)
    domains: list[str]
    collection: str  # the source's display name, e.g. 'LibreTexts Chemistry'
    citation: Citation

    def label(self) -> str:
        where = f"{self.title} › {self.heading}" if self.heading else self.title
        return f"{where} ({self.collection})"

    def passage(self, query: str, limit: int) -> str:
        return _cut(self.text, limit)


@dataclass
class _Source:
    name: str
    file: str
    version: str
    license: str
    cite_url: str
    title: str

    def cite(self, path: str, revision: int | None, passage_id: int) -> Citation:
        locator = {"path": path, "revision": revision or "", "passage": passage_id}
        url = self.cite_url.format(**{k: quote(str(v), safe="/") for k, v in locator.items()})
        return Citation(self.name, self.file, self.version, self.license, locator, url)


class L2Store:
    """One read-only connection to the literature store. Create one per thread."""

    def __init__(self, path: Path | str):
        path = Path(path).resolve()
        if not path.is_file():
            raise L2Error(f"Layer 2 literature store not found at {path}; build it with: {REBUILD_COMMAND}")
        self.path = path
        self.con = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, check_same_thread=False)
        app, version = (self.con.execute(f"PRAGMA {p}").fetchone()[0] for p in ("application_id", "user_version"))
        if (app, version) != (APPLICATION_ID, SCHEMA_VERSION):
            self.con.close()
            raise L2Error(
                f"{path} is not a Layer 2 literature store of schema {SCHEMA_VERSION}"
                f" (application_id {app}, schema {version}); rebuild it with: {REBUILD_COMMAND}"
            )
        self.con.execute("PRAGMA query_only=1")
        self.con.execute("PRAGMA mmap_size=268435456")
        self.sources = {
            row[0]: _Source(*row[1:])
            for row in self.con.execute("SELECT id, name, file, version, license, cite_url, title FROM source")
        }
        self.domains = sorted({d for (ds,) in self.con.execute("SELECT domains FROM source") for d in ds.split()})

    def close(self) -> None:
        self.con.close()

    def search(self, query: str, k: int = 5, domain: str | None = None) -> list[PassageRecord]:
        """Passages with every content word of the query (any of them if none has all), best BM25 first; with
        a domain, only passages having it."""
        words = [w for w in _WORD.findall(query.lower()) if w not in STOPWORDS]
        if not words:
            return []
        rows = self._fts(" ".join(f'"{w}"' for w in words), k, domain)
        if not rows and len(words) > 1:
            rows = self._fts(" OR ".join(f'"{w}"' for w in words), k, domain)
        return [
            PassageRecord(
                title=title,
                heading=heading,
                path=path,
                text=text,
                score=round(score, 4),
                domains=domains.split(),
                collection=self.sources[sid].title,
                citation=self.sources[sid].cite(path, rev, pid),
            )
            for pid, title, heading, text, domains, path, rev, sid, score in rows
        ]

    def _fts(self, match: str, k: int, domain: str | None) -> list[tuple]:
        # Rank inside the FTS query and join the passage rows only for the top k.
        if domain:
            match = f'domains : "{domain}" AND ({match})'
        return self.con.execute(
            f"WITH top AS (SELECT rowid AS id, bm25(passage_fts, {BM25_WEIGHTS}) AS score"
            " FROM passage_fts WHERE passage_fts MATCH ? ORDER BY score LIMIT ?)"
            " SELECT p.id, p.title, p.heading, p.text, p.domains, d.path, d.revision, d.source_id, top.score"
            " FROM top JOIN passage p ON p.id = top.id JOIN doc d ON d.id = p.doc_id ORDER BY top.score",
            (match, k),
        ).fetchall()
