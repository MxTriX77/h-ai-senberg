import json
import os
import sqlite3
import subprocess
import sys

import pytest

from haisenberg.l1.store import L1Error, L1Store


def test_missing_store_names_path_and_rebuild_command(tmp_path):
    with pytest.raises(L1Error) as err:
        L1Store(tmp_path / "nope.db")
    assert "nope.db" in str(err.value) and "ingest.build_l1" in str(err.value)


def test_schema_mismatch_asks_for_rebuild(fresh_store):
    con = sqlite3.connect(fresh_store)
    con.execute("PRAGMA user_version=999")
    con.commit()
    con.close()
    with pytest.raises(L1Error, match="rebuild"):
        L1Store(fresh_store)


def test_exact_name_first(built_store):
    (first, *_) = L1Store(built_store).lookup_constant("planck CONSTANT")
    assert (first.name, first.value, first.unit, first.exact) == ("Planck constant", 6.62607015e-34, "J Hz^-1", True)
    c = first.citation
    assert (c.source, c.version, c.locator) == ("codata_fixture", "2022", {"name": "Planck constant"})
    assert c.url.endswith("search_for=Planck%20constant")


def test_partial_words_shortest_first(built_store):
    names = [r.name for r in L1Store(built_store).lookup_constant("electron mass", k=10)]
    assert names[0] == "electron mass"
    assert names.index("electron mass energy equivalent") < names.index("electron mass energy equivalent in MeV")
    assert "proton-electron mass ratio" in names


def test_no_match_and_empty_query(built_store):
    store = L1Store(built_store)
    assert store.lookup_constant("flux capacitor") == []
    assert store.lookup_constant("   ") == []


def test_records_round_trip_through_json(built_store):
    for r in L1Store(built_store).lookup_constant("constant", k=10):
        d = r.to_dict()
        assert json.loads(json.dumps(d)) == d


def test_lookups_never_modify_the_store(built_store):
    before = built_store.read_bytes()
    store = L1Store(built_store)
    store.lookup_constant("electron mass")
    with pytest.raises(sqlite3.OperationalError):
        store.con.execute("DELETE FROM constant")
    store.close()
    assert built_store.read_bytes() == before


def run_cli(*args, env=None):
    return subprocess.run(
        [sys.executable, "-m", "haisenberg.l1.cli", *args],
        capture_output=True,
        env={**os.environ, **(env or {})},
    )


def test_cli_json(built_store):
    out = run_cli("constant", "speed of light in vacuum", "--json", "--db", str(built_store))
    assert out.returncode == 0
    first = json.loads(out.stdout.decode("utf-8"))[0]
    assert (first["value"], first["unit"]) == (299792458, "m s^-1")


def test_cli_missing_store(tmp_path):
    out = run_cli("constant", "c", "--db", str(tmp_path / "nope.db"))
    assert out.returncode != 0 and b"ingest.build_l1" in out.stderr


def test_cli_utf8_on_cp1252_console(built_store):
    out = run_cli("constant", "electron mass", "--db", str(built_store), env={"PYTHONIOENCODING": "cp1252"})
    assert out.returncode == 0
    assert "±" in out.stdout.decode("utf-8")
