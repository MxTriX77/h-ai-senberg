"""Shared pieces of the Layer 1 build: the build context, errors and source rows."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from ingest.manifest import Entry


class BuildError(Exception):
    """The build cannot continue; the message names the input or row at fault."""


@dataclass
class BuildContext:
    con: sqlite3.Connection
    raw_dir: Path
    inputs: dict[str, list[Entry]]  # use tag -> manifest entries
    report: dict[str, int] = field(default_factory=dict)  # counts printed and stored in meta
    linked_cids: dict[int, int] = field(default_factory=dict)  # PubChem CID -> article id

    def files(self, use: str) -> list[tuple[Entry, Path]]:
        return [(e, e.path(self.raw_dir)) for e in self.inputs.get(use, [])]

    def add_source(self, entry: Entry, version: str) -> int:
        path = entry.path(self.raw_dir)
        cur = self.con.execute(
            "INSERT INTO source(name, use, file, sha256, version, license, cite_url, domains)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry.name, entry.use, entry.dest, sha256(path), version, entry.license, entry.cite_url or "",
                " ".join(entry.domains),
            ),
        )
        return cur.lastrowid


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 22):
            h.update(block)
    return h.hexdigest()
