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
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConstantRecord(Record):
    name: str
    value: float
    value_text: str
    uncertainty: float
    unit: str
    exact: bool
    truncated: bool
    citation: Citation


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
    citation: Citation


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

    def cite(self, locator: dict, note: str = "") -> Citation:
        url = self.cite_url.format(**{k: quote(str(v), safe="") for k, v in locator.items()})
        return Citation(self.name, self.file, self.version, self.license, locator, url, note)


class L1Store:
    """One read-only connection to the store. Create one per thread."""

    def __init__(self, path: Path | str | None = None):
        if path is None:
            from haisenberg.config import load_config

            path = load_config().retrieval.layer1_encyclopedia_db
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
            for row in self.con.execute("SELECT id, name, use, file, version, license, cite_url FROM source")
        }

    def close(self) -> None:
        self.con.close()

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


def lookup_constant(query: str, k: int = 5) -> list[ConstantRecord]:
    return default_store().lookup_constant(query, k)


def lookup_compound(query: str, k: int = 5) -> list[CompoundRecord]:
    return default_store().lookup_compound(query, k)
