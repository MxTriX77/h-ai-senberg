import socket
import sqlite3

import pytest

from ingest.build_l1 import build, main
from ingest.l1base import BuildError, sha256


def test_build_stores_constants_and_sources(fresh_store, fixture_raw):
    con = sqlite3.connect(fresh_store)
    assert con.execute("SELECT count(*) FROM constant").fetchone()[0] == 12
    name, file, digest, version, cite = con.execute(
        "SELECT name, file, sha256, version, cite_url FROM source WHERE use = 'codata'"
    ).fetchone()
    assert (name, file, version) == ("codata_fixture", "codata/allascii.txt", "2022")
    assert digest == sha256(fixture_raw / "codata/allascii.txt")
    assert "{name}" in cite
    meta = dict(con.execute("SELECT key, value FROM meta"))
    assert meta["constants"] == "12" and meta["built_at"].endswith("Z")
    assert not (fresh_store.parent / "l1.db.building").exists()


def test_missing_input_leaves_old_store_untouched(fresh_store, fixture_raw, fixture_manifest):
    before = fresh_store.read_bytes()
    (fixture_raw / "codata/allascii.txt").unlink()
    with pytest.raises(BuildError) as err:
        build(fixture_manifest, fixture_raw, fresh_store)
    assert "codata_fixture" in str(err.value) and "allascii.txt" in str(err.value)
    assert fresh_store.read_bytes() == before


def test_build_needs_no_network(fixture_raw, fixture_manifest, tmp_path, monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access during build")

    monkeypatch.setattr(socket, "socket", refuse)
    build(fixture_manifest, fixture_raw, tmp_path / "offline.db")
    assert (tmp_path / "offline.db").exists()


def test_cli_reports_and_fails_cleanly(fixture_raw, fixture_manifest, tmp_path, capsys):
    db = tmp_path / "cli.db"
    assert main(["--raw-dir", str(fixture_raw), "--db", str(db), "--manifest", str(fixture_manifest)]) == 0
    assert "constants" in capsys.readouterr().out
    assert main(["--raw-dir", str(tmp_path / "nowhere"), "--db", str(db), "--manifest", str(fixture_manifest)]) == 1
    assert "missing input" in capsys.readouterr().err
