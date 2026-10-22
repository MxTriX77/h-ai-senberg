import sqlite3

from ingest.build_l1 import build

from .l1_helpers import write_manifest

TABLES = ["source", "article", "article_alias", "constant", "compound", "compound_name"]


def dump(db):
    con = sqlite3.connect(db)
    out = {t: con.execute(f"SELECT * FROM {t} ORDER BY 1, 2").fetchall() for t in TABLES}
    out["meta"] = con.execute("SELECT * FROM meta WHERE key <> 'built_at' ORDER BY key").fetchall()
    return out


def test_two_builds_have_identical_rows(fixture_raw, tmp_path):
    manifest = write_manifest(tmp_path / "sources.yaml")
    build(manifest, fixture_raw, tmp_path / "a.db")
    build(manifest, fixture_raw, tmp_path / "b.db")
    assert dump(tmp_path / "a.db") == dump(tmp_path / "b.db")
