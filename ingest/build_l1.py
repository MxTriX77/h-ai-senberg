"""Rebuild the Layer 1 store from the raw downloads alone (no network).

    uv run python -m ingest.build_l1 [--raw-dir DIR] [--db FILE] [--manifest FILE]

Inputs are found through the manifest's `use` tags. The store is built into `<db>.building` and
moved over `<db>` only when the build succeeds, so a failed build leaves the old store untouched.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from haisenberg.config import ConfigError, load_config
from haisenberg.l1.schema import DDL, SCHEMA_VERSION
from ingest import codata, pubchem
from ingest.l1base import BuildContext, BuildError
from ingest.manifest import ManifestError, load_manifest

# (use tags the step needs, loader). Order matters: later steps read what earlier ones stored.
STEPS = [
    (("codata",), codata.load),
    (("pubchem-mesh", "pubchem-mass", "pubchem-title", "pubchem-synonyms"), pubchem.load),
]


def build(manifest: Path, raw_dir: Path, db: Path) -> dict[str, int | float | str]:
    """Build the store at db. Returns the build report. Raises BuildError or ManifestError."""
    t0 = time.monotonic()
    entries = load_manifest(manifest)
    inputs: dict[str, list] = {}
    for e in entries:
        inputs.setdefault(e.use, []).append(e)
    for uses, _ in STEPS:
        for use in uses:
            if use not in inputs:
                raise BuildError(f"manifest {manifest} has no entry with use '{use}'")
            for e in inputs[use]:
                if not e.path(raw_dir).is_file():
                    raise BuildError(f"missing input for entry {e.name}: {e.path(raw_dir)}")

    db.parent.mkdir(parents=True, exist_ok=True)
    tmp = db.with_name(db.name + ".building")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    try:
        con.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;" + DDL)
        con.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        ctx = BuildContext(con=con, raw_dir=raw_dir, inputs=inputs)
        for _, step in STEPS:
            step(ctx)
        report = {
            "articles": con.execute("SELECT count(*) FROM article").fetchone()[0],
            "aliases": con.execute("SELECT count(*) FROM article_alias").fetchone()[0],
            "constants": con.execute("SELECT count(*) FROM constant").fetchone()[0],
            "compounds": con.execute("SELECT count(*) FROM compound").fetchone()[0],
            "compound_names": con.execute("SELECT count(*) FROM compound_name").fetchone()[0],
            **{k: v for k, v in ctx.report.items() if k.startswith("skipped")},
        }
        built_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        con.executemany(
            "INSERT INTO meta(key, value) VALUES (?, ?)",
            [("built_at", built_at), *((k, str(v)) for k, v in report.items())],
        )
        if report["articles"]:
            con.execute("INSERT INTO article_fts(article_fts) VALUES ('optimize')")
        con.execute("ANALYZE")
        con.commit()
        con.execute("VACUUM")
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
    parser.add_argument("--db", type=Path, help="default: retrieval.layer1_encyclopedia_db")
    parser.add_argument("--manifest", type=Path, help="default: config/sources.yaml in the project")
    args = parser.parse_args(argv)

    raw_dir, db, manifest = args.raw_dir, args.db, args.manifest
    try:
        if None in (raw_dir, db, manifest):
            cfg = load_config()
            raw_dir = raw_dir or cfg.retrieval.raw_dir
            db = db or cfg.retrieval.layer1_encyclopedia_db
            manifest = manifest or cfg.root / "config" / "sources.yaml"
        report = build(manifest, raw_dir, db)
    except (BuildError, ConfigError, ManifestError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"built {db}")
    for k, v in report.items():
        print(f"  {k:<16} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
