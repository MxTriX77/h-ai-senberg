"""Rebuild the Layer 2 literature store from the raw downloads alone (no network).

    uv run python -m ingest.build_l2 [--raw-dir DIR] [--db FILE] [--manifest FILE]

Inputs are found through the manifest's `use` tags; the Layer 1 store is never opened. The store is built into
`<db>.building` and moved over `<db>` only when the build succeeds. A passage whose text (whitespace collapsed,
case ignored) was stored before is left out. A Wikipedia article is stored from the first ZIM that holds it
(manifest order, topic ZIMs first); later ZIMs holding it only add their domains. A filtered ZIM's pages are rated
on their own WikiProject assessments or, as the top-1M dump has none, on their copy in the all-articles mini ZIM.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from haisenberg.config import ConfigError, load_config
from haisenberg.l2.schema import APPLICATION_ID, DDL, SCHEMA_VERSION
from ingest import l2sources
from ingest.l1base import BuildError, sha256
from ingest.manifest import Entry, ManifestError, load_manifest
from ingest.zim import Zim

WIKI_USES = ("wikipedia-zim", "wikipedia-text-zim", "wikipedia-filtered-zim")


@dataclass
class Writer:
    con: sqlite3.Connection
    raw_dir: Path
    limit: int  # longest passage, in characters
    report: Counter = field(default_factory=Counter)
    seen: set[int] = field(default_factory=set)  # digests of the stored passages' normalized text
    wiki: dict[str, int] = field(default_factory=dict)  # Wikipedia article path -> doc id
    more: dict[int, set[str]] = field(default_factory=dict)  # doc id -> domains of later ZIMs holding it
    minis: list[Path] = field(default_factory=list)  # all-articles mini ZIMs: WikiProject ratings for filtered ZIMs

    def source(self, entry: Entry, version: str, title: str) -> int:
        cur = self.con.execute(
            "INSERT INTO source(name, use, file, sha256, version, license, cite_url, domains, title)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (entry.name, entry.use, entry.dest, sha256(entry.path(self.raw_dir)), version, entry.license,
             entry.cite_url or "", " ".join(entry.domains), title),
        )
        return cur.lastrowid

    def add(self, source_id: int, entry: Entry, doc: l2sources.Doc) -> int | None:
        """Store doc's new passages; returns the doc id, or None when every passage was stored before."""
        rows = []
        for heading, text in doc.passages:
            key = int.from_bytes(hashlib.blake2b(" ".join(text.lower().split()).encode(), digest_size=8).digest())
            if key in self.seen:
                self.report["skipped_duplicate_passages"] += 1
                continue
            self.seen.add(key)
            rows.append((heading, text))
        if not rows:
            return None
        doc_id = self.con.execute(
            "INSERT INTO doc(source_id, path, title, revision) VALUES (?, ?, ?, ?)",
            (source_id, doc.path, doc.title, doc.revision),
        ).lastrowid
        domains = " ".join(doc.domains or entry.domains)
        self.con.executemany(
            "INSERT INTO passage(doc_id, title, heading, text, domains) VALUES (?, ?, ?, ?, ?)",
            [(doc_id, doc.title, h, t, domains) for h, t in rows],
        )
        self.report[f"{entry.name}: docs"] += 1
        self.report[f"{entry.name}: passages"] += len(rows)
        return doc_id

    def scans(self, source_id: int, entry: Entry, scans) -> None:
        for scan in scans:
            self.report.update(scan.dropped)
            for doc in scan.docs:
                self.add(source_id, entry, doc)


def load_wikipedia(w: Writer, entry: Entry, path: Path) -> None:
    with Zim(path) as z:
        sid = w.source(entry, z.metadata.get("Date", ""), z.metadata.get("Title", entry.name))
        skip, rated = frozenset(), {}
        if entry.use == "wikipedia-filtered-zim":
            skip = frozenset(w.wiki)
            for mini in w.minis:  # its pages may lack WikiProject ratings; the mini ZIM's copies have them
                with Zim(mini) as m:
                    rated.update(l2sources.rated(m, entry.filter))
            w.report[f"{entry.name}: rated in a mini ZIM"] = len(rated)
            for path, domains in rated.items():  # stored from a topic ZIM: still gains the domains it is rated for
                if (doc_id := w.wiki.get(path)) is not None:
                    w.more.setdefault(doc_id, set()).update(domains)
        for scan in l2sources.wiki_scans(z, w.limit, entry.filter, skip, rated=rated):
            w.report.update(scan.dropped)
            for doc in scan.docs:
                if (doc_id := w.wiki.get(doc.path)) is not None:
                    w.more.setdefault(doc_id, set()).update(entry.domains)
                elif (doc_id := w.add(sid, entry, doc)) is not None:
                    w.wiki[doc.path] = doc_id


def load_wikibooks(w: Writer, entry: Entry, path: Path) -> None:
    with Zim(path) as z:
        sid = w.source(entry, z.metadata.get("Date", ""), z.metadata.get("Title", entry.name))
        w.scans(sid, entry, l2sources.wiki_scans(z, w.limit, books=entry.filter))


def load_libretexts(w: Writer, entry: Entry, path: Path) -> None:
    with Zim(path) as z:
        sid = w.source(entry, z.metadata.get("Date", ""), z.metadata.get("Title", entry.name))
        w.scans(sid, entry, l2sources.libretexts_scans(z, w.limit))


def load_stackexchange(w: Writer, entry: Entry, path: Path) -> None:
    with Zim(path) as z:
        sid = w.source(entry, z.metadata.get("Date", ""), z.metadata.get("Title", entry.name))
        w.scans(sid, entry, l2sources.stackexchange_scans(z, w.limit, entry.filter.min_answer_score))


def load_chebi(w: Writer, entry: Entry, path: Path) -> None:
    release, terms = l2sources.chebi_terms(path)
    sid = w.source(entry, release, "ChEBI")
    for doc in l2sources.chebi_docs(terms, w.limit):
        w.add(sid, entry, doc)


# (use tag, loader) in build order: Wikipedia topic ZIMs first, so the first copy of an article comes from them.
LOADERS = [
    *((use, load_wikipedia) for use in WIKI_USES),
    ("wikibooks-zim", load_wikibooks),
    ("libretexts-zim", load_libretexts),
    ("stackexchange-zim", load_stackexchange),
    ("chebi-obo", load_chebi),
]


def build(manifest: Path, raw_dir: Path, db: Path, limit: int) -> dict[str, int | float | str]:
    """Build the store at db from every manifest entry a loader reads. Returns the build report."""
    t0 = time.monotonic()
    entries = load_manifest(manifest)
    todo = [(e, loader) for use, loader in LOADERS for e in entries if e.use == use]
    filtered = any(e.use == "wikipedia-filtered-zim" for e in entries)
    minis = [e for e in entries if e.use == "wikipedia-mini-zim"] if filtered else []  # inputs too: their ratings
    for e in [*(e for e, _ in todo), *minis]:
        if not e.path(raw_dir).is_file():
            raise BuildError(f"missing input for entry {e.name}: {e.path(raw_dir)}")

    db.parent.mkdir(parents=True, exist_ok=True)
    tmp = db.with_name(db.name + ".building")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    try:
        # 64 KB pages: a third fewer reads per search over WSL's /mnt/c (p95 107 -> 70 ms), same speed natively
        con.executescript("PRAGMA page_size=65536; PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;"
                          " PRAGMA cache_size=-262144;" + DDL)
        con.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        con.execute(f"PRAGMA application_id={APPLICATION_ID}")
        w = Writer(con, raw_dir, limit, minis=[e.path(raw_dir) for e in minis])
        for entry, loader in todo:
            loader(w, entry, entry.path(raw_dir))
            con.commit()
        for doc_id, extra in w.more.items():
            (old,) = con.execute("SELECT domains FROM passage WHERE doc_id = ? LIMIT 1", (doc_id,)).fetchone()
            con.execute("UPDATE passage SET domains = ? WHERE doc_id = ?",
                        (" ".join(sorted(set(old.split()) | extra)), doc_id))
        con.execute("INSERT INTO passage_fts(passage_fts) VALUES ('rebuild')")
        con.execute("INSERT INTO passage_fts(passage_fts) VALUES ('optimize')")
        report = {
            "docs": con.execute("SELECT count(*) FROM doc").fetchone()[0],
            "passages": con.execute("SELECT count(*) FROM passage").fetchone()[0],
            **dict(sorted(w.report.items())),
        }
        built_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        meta = [("built_at", built_at), ("chunk_chars", str(limit)), *((k, str(v)) for k, v in report.items())]
        con.executemany("INSERT INTO meta(key, value) VALUES (?, ?)", meta)
        con.execute("ANALYZE")
        con.commit()
    except BaseException:
        con.close()
        tmp.unlink(missing_ok=True)
        raise
    con.close()
    try:
        os.replace(tmp, db)
    except PermissionError as e:
        raise BuildError(f"cannot replace {db} (is it open in another program?): {e}") from e
    report["size_mb"] = round(db.stat().st_size / 1e6, 1)
    report["seconds"] = round(time.monotonic() - t0, 1)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, help="default: retrieval.raw_dir from the config")
    parser.add_argument("--db", type=Path, help="default: the literature store in retrieval.stores")
    parser.add_argument("--manifest", type=Path, help="default: config/sources.yaml in the project")
    args = parser.parse_args(argv)

    try:
        cfg = load_config()
        raw_dir = args.raw_dir or cfg.retrieval.raw_dir
        db = args.db or cfg.retrieval.path("literature")
        manifest = args.manifest or cfg.root / "config" / "sources.yaml"
        report = build(manifest, raw_dir, db, cfg.retrieval.layer2_chunk_chars)
    except (BuildError, ConfigError, ManifestError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"built {db}")
    for k, v in report.items():
        print(f"  {k:<48} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
