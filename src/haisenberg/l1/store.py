"""Read-only access to the Layer 1 store, returning records that carry their own citation."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote

from haisenberg.l1.schema import SCHEMA_VERSION
from haisenberg.l1.text import normalize, to_hill

REBUILD_COMMAND = "uv run python -m ingest.build_l1"


class L1Error(Exception):
    """The store is missing, outdated or unreadable; the message says how to fix it."""


@dataclass
class Citation:
    source: str  # manifest name of the source file
    file: str  # path relative to the raw directory
    version: str
    license: str
    locator: dict  # what identifies the record inside the source
    url: str
    note: str = ""


@dataclass
class Record:
    """A citeable record. label() names it in lists; passage() is the text handed to a model."""

    def to_dict(self) -> dict:
        return asdict(self)

    def label(self) -> str:
        raise NotImplementedError

    def passage(self, query: str, limit: int) -> str:
        raise NotImplementedError


@dataclass
class ConstantRecord(Record):
    name: str
    value: float
    value_text: str
    uncertainty: float
    unit: str
    exact: bool
    truncated: bool
    domains: list[str]
    citation: Citation

    def label(self) -> str:
        return self.name

    def passage(self, query: str, limit: int) -> str:
        how = "exact" if self.exact else f"standard uncertainty {self.uncertainty:g} {self.unit}".rstrip()
        value = f"{self.name} = {self.value_text} {self.unit}".rstrip()
        return f"{value} ({how}); {self.citation.source}, version {self.citation.version}"


@dataclass
class ArticleRecord(Record):
    title: str
    path: str
    lead: str
    infobox: str
    snippet: str  # best-matching passage, matched words in [brackets]
    score: float  # BM25 (lower is better); 0.0 for an exact title or alias match
    domains: list[str]
    citation: Citation

    def label(self) -> str:
        return self.title

    def passage(self, query: str, limit: int) -> str:
        """Lead, then infobox lines, at most limit characters. Infobox lines with query words (most
        matches first) get up to half of the room, the lead the rest, other lines what is left."""
        words = {w for w in _WORD.findall(query.lower()) if w not in STOPWORDS}
        lines = [line for line in self.infobox.splitlines() if line.strip()]
        matches = {line: sum(w in line.lower() for w in words) for line in lines}
        picked: list[str] = []
        used = 0
        for line in sorted((x for x in lines if matches[x]), key=lambda x: -matches[x]):
            if used + len(line) + 1 <= limit // 2:
                picked.append(line)
                used += len(line) + 1
        text = "\n".join([x for x in [_cut(self.lead, limit - used)] if x] + picked)
        for line in lines:
            if line not in picked:
                if len(text) + len(line) + 1 > limit:
                    break
                text = f"{text}\n{line}" if text else line
        return text


@dataclass
class CompoundRecord(Record):
    cid: int
    title: str
    formula: str
    molar_mass: float | None  # g/mol, computed from the formula; None with molar_mass_note
    molar_mass_note: str | None
    monoisotopic_mass: float
    synonyms: list[str]
    matched_by: str  # 'cid', 'name' or 'formula'
    domains: list[str]
    citation: Citation

    def label(self) -> str:
        return f"{self.title} (CID {self.cid})"

    def passage(self, query: str, limit: int) -> str:
        if self.molar_mass is not None:
            mass = f"molar mass {self.molar_mass:.3f} g/mol ({self.citation.note})"
        else:
            mass = f"molar mass not given: {self.molar_mass_note}"
        synonyms = f"; synonyms: {', '.join(self.synonyms[:6])}" if self.synonyms else ""
        return (
            f"CID {self.cid}, {self.title}: formula {self.formula}; {mass};"
            f" monoisotopic mass {self.monoisotopic_mass:g} Da{synonyms}"
        )


def _cut(text: str, n: int) -> str:
    """text, shortened at a word boundary to at most n characters (with an ellipsis)."""
    if len(text) <= n:
        return text
    return text[: max(n - 1, 0)].rsplit(" ", 1)[0] + "â€¦" if n > 1 else ""


# Ignored in encyclopedia queries: they match nearly every article and slow BM25 down.
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how in is it its of on or "
    "that the their this to was were what when where which who why with".split()
)
BM25_WEIGHTS = "10.0, 5.0, 2.0, 1.0"  # title, aliases, infobox, lead
_WORD = re.compile(r"\w+")

# Kind of knowledge each source use tag provides. `text` sources are searched together; every other
# kind has one lookup (`lookup_<kind>`). A new kind = an ingest adapter + its lookup method + an entry here.
KIND_BY_USE = {
    "wikipedia-zim": "text",
    "wikipedia-mini-zim": "text",
    "pubchem-mass": "compound",
    "codata": "constant",
}

MOLAR_MASS_NOTE = "molar mass computed from the PubChem formula with IUPAC standard atomic weights"
_CID = re.compile(r"^\s*(?:cid\s*:?\s*)?(\d+)\s*$", re.I)


@dataclass
class _Source:
    name: str
    use: str
    file: str
    version: str
    license: str
    cite_url: str
    domains: str  # space-separated

    def cite(self, locator: dict, note: str = "") -> Citation:
        url = self.cite_url.format(**{k: quote(str(v), safe="") for k, v in locator.items()})
        return Citation(self.name, self.file, self.version, self.license, locator, url, note)


class L1Store:
    """One read-only connection to the store. Create one per thread."""

    def __init__(self, path: Path | str | None = None, top_k: int = 5):
        """Open the store at path, or at retrieval.layer1_encyclopedia_db (then top_k is retrieval.top_k_proofs)."""
        self.top_k = top_k
        if path is None:
            from haisenberg.config import load_config

            retrieval = load_config().retrieval
            path, self.top_k = retrieval.layer1_encyclopedia_db, retrieval.top_k_proofs
        path = Path(path).resolve()
        if not path.is_file():
            raise L1Error(f"Layer 1 store not found at {path}; build it with: {REBUILD_COMMAND}")
        self.path = path
        self.con = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, check_same_thread=False)
        version = self.con.execute("PRAGMA user_version").fetchone()[0]
        if version != SCHEMA_VERSION:
            self.con.close()
            raise L1Error(
                f"Layer 1 store {path} has schema {version}, expected {SCHEMA_VERSION};"
                f" rebuild it with: {REBUILD_COMMAND}"
            )
        self.con.execute("PRAGMA query_only=1")
        self.con.execute("PRAGMA mmap_size=268435456")
        self.sources = {
            row[0]: _Source(*row[1:])
            for row in self.con.execute("SELECT id, name, use, file, version, license, cite_url, domains FROM source")
        }

    def close(self) -> None:
        self.con.close()

    def kinds(self) -> dict[str, list[str]]:
        """Kinds of knowledge in the store (see KIND_BY_USE), each with the sorted domains of its sources."""
        found: dict[str, set[str]] = {}
        for s in self.sources.values():
            if kind := KIND_BY_USE.get(s.use):
                found.setdefault(kind, set()).update(s.domains.split())
        return {kind: sorted(domains) for kind, domains in sorted(found.items())}

    def lookup(self, kind: str, query: str, k: int = 5) -> list[Record]:
        """The lookup of a structured kind by its name, e.g. lookup('compound', 'caffeine')."""
        if kind == "text" or kind not in self.kinds():
            raise L1Error(f"no lookup for kind '{kind}' in this store")
        return getattr(self, f"lookup_{kind}")(query, k)

    def search_encyclopedia(self, query: str, k: int | None = None, domain: str | None = None) -> list[ArticleRecord]:
        """Exact title/alias match first, then BM25 over articles with all words (any word if none).

        With a domain, only articles having that domain are considered."""
        k = k or self.top_k
        words = [w for w in _WORD.findall(query.lower()) if w not in STOPWORDS]
        if not words:
            return []
        rows = self._fts(" ".join(f'"{w}"' for w in words), k, domain)
        if not rows and len(words) > 1:
            rows = self._fts(" OR ".join(f'"{w}"' for w in words), k, domain)
        in_domain = (
            " AND EXISTS (SELECT 1 FROM article_domain d WHERE d.article_id = article_alias.article_id AND d.domain = ?)"
            if domain
            else ""
        )
        exact = [
            r[0]
            for r in self.con.execute(
                f"SELECT article_id FROM article_alias WHERE alias = ?{in_domain}",
                (normalize(query), domain) if domain else (normalize(query),),
            )
        ]
        if exact:
            found = {r[0]: r for r in rows}
            first = []
            for article_id in exact:
                row = found.pop(article_id, None) or self.con.execute(
                    "SELECT id, path, title, lead, infobox, revision, source_id, 0.0, substr(lead, 1, 200)"
                    " FROM article WHERE id = ?",
                    (article_id,),
                ).fetchone()
                first.append(row[:7] + (0.0,) + row[8:])
            rows = first + [r for r in rows if r[0] in found]
        return [
            ArticleRecord(
                title=title,
                path=path,
                lead=lead,
                infobox=infobox,
                snippet=" | ".join(line for line in snippet.splitlines() if line.strip()),
                score=round(score, 4),
                domains=self._article_domains(article_id),
                citation=self.sources[sid].cite({"path": path, "revision": rev or ""}),
            )
            for article_id, path, title, lead, infobox, rev, sid, score, snippet in rows[:k]
        ]

    def _article_domains(self, article_id: int) -> list[str]:
        sql = "SELECT domain FROM article_domain WHERE article_id = ? ORDER BY domain"
        return [r[0] for r in self.con.execute(sql, (article_id,))]

    def _fts(self, match: str, k: int, domain: str | None = None) -> list[tuple]:
        # Rank inside the FTS query first and join article rows only for the top k (10x faster).
        # A domain filter applies inside the ranking query, before LIMIT; as a per-row EXISTS, because
        # `rowid IN (...)` makes FTS5 look up every article of the domain (100x slower).
        in_domain = (
            " AND EXISTS (SELECT 1 FROM article_domain d WHERE d.article_id = article_fts.rowid AND d.domain = ?)"
            if domain
            else ""
        )
        return self.con.execute(
            f"WITH top AS (SELECT rowid AS id, bm25(article_fts, {BM25_WEIGHTS}) AS score,"
            # passage from the lead (column 3), or from the infobox (column 2) when the lead is empty
            " coalesce(nullif(snippet(article_fts, 3, '[', ']', '…', 24), ''),"
            " snippet(article_fts, 2, '[', ']', '…', 24)) AS snip"
            f" FROM article_fts WHERE article_fts MATCH ?{in_domain} ORDER BY score LIMIT ?)"
            " SELECT a.id, a.path, a.title, a.lead, a.infobox, a.revision, a.source_id, top.score, top.snip"
            " FROM top JOIN article a ON a.id = top.id ORDER BY top.score",
            (match, domain, k) if domain else (match, k),
        ).fetchall()

    def lookup_constant(self, query: str, k: int = 5) -> list[ConstantRecord]:
        """Exact (case-insensitive) name first, then names containing every query word, shortest first."""
        q = " ".join(query.split())
        words = q.lower().split()
        if not words:
            return []
        cols = "name, value, value_text, uncertainty, unit, exact, truncated, source_id"
        rows = self.con.execute(f"SELECT {cols} FROM constant WHERE name = ?", (q,)).fetchall()
        where = " AND ".join("instr(lower(name), ?) > 0" for _ in words)
        rows += self.con.execute(
            f"SELECT {cols} FROM constant WHERE {where} AND name <> ? ORDER BY length(name), name LIMIT ?",
            (*words, q, k),
        ).fetchall()
        return [
            ConstantRecord(
                name=name,
                value=value,
                value_text=value_text,
                uncertainty=unc,
                unit=unit,
                exact=bool(exact),
                truncated=bool(truncated),
                domains=self.sources[sid].domains.split(),
                citation=self.sources[sid].cite({"name": name}),
            )
            for name, value, value_text, unc, unit, exact, truncated, sid in rows[:k]
        ]

    def lookup_compound(self, query: str, k: int = 5) -> list[CompoundRecord]:
        """By CID (`2519`, `CID 2519`), then by normalized name/synonym/CAS, then by formula (Hill order)."""
        cols = "c.cid, c.title, c.formula, c.molar_mass, c.mass_note, c.monoisotopic_mass, c.synonyms"
        if m := _CID.match(query):
            rows = self.con.execute(f"SELECT {cols} FROM compound c WHERE cid = ?", (int(m.group(1)),)).fetchall()
            return self._compounds(rows, "cid")
        key = normalize(query)
        if not key:
            return []
        rows = self.con.execute(
            f"SELECT {cols} FROM compound_name n JOIN compound c ON c.cid = n.cid WHERE n.name = ?"
            " ORDER BY n.rank, c.selected_by = 'both' DESC, c.cid LIMIT ?",
            (key, k),
        ).fetchall()
        if rows:
            return self._compounds(rows, "name")
        hill = to_hill(query.strip())
        if hill is None:
            return []
        rows = self.con.execute(
            f"SELECT {cols} FROM compound c WHERE formula = ? ORDER BY c.selected_by = 'both' DESC, c.cid LIMIT ?",
            (hill, k),
        ).fetchall()
        return self._compounds(rows, "formula")

    def _compounds(self, rows, matched_by: str) -> list[CompoundRecord]:
        source = next(s for s in self.sources.values() if s.use == "pubchem-mass")
        return [
            CompoundRecord(
                cid=cid,
                title=title,
                formula=formula,
                molar_mass=mm,
                molar_mass_note=note,
                monoisotopic_mass=mono,
                synonyms=syns.split("|") if syns else [],
                matched_by=matched_by,
                domains=source.domains.split(),
                citation=source.cite({"cid": cid}, MOLAR_MASS_NOTE),
            )
            for cid, title, formula, mm, note, mono, syns in rows
        ]


_default: L1Store | None = None


def default_store() -> L1Store:
    """The store at retrieval.layer1_encyclopedia_db, opened on first use."""
    global _default
    if _default is None:
        _default = L1Store()
    return _default


def search_encyclopedia(query: str, k: int | None = None, domain: str | None = None) -> list[ArticleRecord]:
    return default_store().search_encyclopedia(query, k, domain)


def lookup_constant(query: str, k: int = 5) -> list[ConstantRecord]:
    return default_store().lookup_constant(query, k)


def lookup_compound(query: str, k: int = 5) -> list[CompoundRecord]:
    return default_store().lookup_compound(query, k)
