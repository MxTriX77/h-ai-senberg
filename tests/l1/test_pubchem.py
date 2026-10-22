import gzip
import sqlite3

import pytest

from haisenberg.l1.store import L1Store
from haisenberg.l1.text import parse_formula, to_hill
from ingest.l1base import BuildError
from ingest.pubchem import load_atomic_weights, molar_mass, stream_selected

W, ISO = load_atomic_weights()


def test_atomic_weights_table():
    assert len(W) == 118
    assert (W["H"], W["C"], W["Cl"], W["Fe"], W["Tc"]) == (1.008, 12.011, 35.45, 55.845, None)
    assert ISO["C"] == 12.0


def test_formula_parsing_and_hill_order():
    assert parse_formula("C9H18NO4+") == ({"C": 9, "H": 18, "N": 1, "O": 4}, "+")
    assert to_hill("NaCl") == "ClNa"
    assert to_hill("H6C6") == "C6H6"
    assert to_hill("C2H5OH") == "C2H6O"
    assert to_hill("SO4H2") == "H2O4S"
    assert to_hill("Fe+3") == "Fe+3"
    assert to_hill("benzene") is None


@pytest.mark.parametrize(
    "formula, mono, expected",
    [
        ("H2O", 18.010564683, 18.015),
        ("C8H10N4O2", 194.08037557, 194.19),
        ("ClNa", 57.9586220, 58.44),
        ("H2O4S", 97.96737971, 98.07),
        ("C9H18NO4+", 204.12358306, 204.246),  # charge ignored
    ],
)
def test_molar_mass(formula, mono, expected):
    mass, note = molar_mass(formula, mono, W, ISO)
    assert note is None and mass == pytest.approx(expected, abs=0.005)


def test_no_standard_atomic_weight():
    mass, note = molar_mass("Pu", 244.06420, W, ISO)
    assert mass is None and "Pu" in note


def test_heavy_water_is_isotope_labelled():
    mass, note = molar_mass("H2O", 20.023118175, W, ISO)
    assert mass is None and "isotope-labelled" in note


def test_stream_selected_caps_and_checks_order(tmp_path):
    p = tmp_path / "x.gz"
    p.write_bytes(gzip.compress(b"1\ta\n1\tb\n1\tc\n7\td\n12\te\n"))
    assert stream_selected(p, [1, 7, 9], cap=2) == {1: ["a", "b"], 7: ["d"]}
    p.write_bytes(gzip.compress(b"5\ta\n3\tb\n"))
    stream_selected(p, [3], cap=1)  # one chunk: order is only checked between chunks
    big = b"".join(b"%d\tx\n" % i for i in range(2_000_000, 0, -1))
    p.write_bytes(gzip.compress(big))
    with pytest.raises(BuildError, match="not sorted"):
        stream_selected(p, [1], cap=1)


def test_subset_rule_and_columns(built_store):
    con = sqlite3.connect(built_store)
    row = con.execute("SELECT title, formula, selected_by, molar_mass FROM compound WHERE cid = 241").fetchone()
    assert row[:3] == ("Benzene", "C6H6", "both") and row[3] == pytest.approx(78.114, abs=0.001)
    by = dict(con.execute("SELECT cid, selected_by FROM compound WHERE cid IN (702, 2519, 5234)"))
    assert by == {702: "wikipedia", 2519: "mesh", 5234: "both"}  # 702: Ethanol's chembox, not in MeSH fixture
    assert con.execute("SELECT count(*) FROM compound WHERE cid = 23931").fetchone()[0] == 0  # not in MeSH
    meta = dict(con.execute("SELECT key, value FROM meta"))
    assert meta["skipped_no_mass"] == "1"  # CID 999999999 is in the MeSH fixture only
    note = con.execute("SELECT molar_mass, mass_note FROM compound WHERE cid = 24602").fetchone()
    assert note[0] is None and "isotope-labelled" in note[1]
    sources = {r[0] for r in con.execute("SELECT use FROM source")}
    assert {"pubchem-mesh", "pubchem-mass", "pubchem-title", "pubchem-synonyms"} <= sources


def test_names(built_store):
    con = sqlite3.connect(built_store)
    names = lambda cid: {r[0]: r[1] for r in con.execute("SELECT name, rank FROM compound_name WHERE cid = ?", (cid,))}
    benzene = names(241)
    assert benzene["benzene"] == 0 and benzene["71-43-2"] == 3
    assert "[6]annulene" not in benzene  # 21st synonym: only the first 20 are indexed
    assert len(benzene) <= 21
    shared = {r[0] for r in con.execute("SELECT cid FROM compound_name WHERE name = 'glucose'")}
    assert shared == {5793, 107526}  # fixture adds 'glucose' to 107526 to test a shared name


def test_lookup_by_name(built_store):
    (r, *_) = L1Store(built_store).lookup_compound("Caffeine")
    assert (r.cid, r.formula, r.matched_by) == (2519, "C8H10N4O2", "name")
    assert r.molar_mass == pytest.approx(194.19, abs=0.01)
    assert r.citation.url.endswith("/compound/2519") and "standard atomic weights" in r.citation.note
    assert "Guaranine" in r.synonyms


def test_lookup_by_cas_cid_and_formula(built_store):
    store = L1Store(built_store)
    assert [(r.cid, r.matched_by) for r in store.lookup_compound("71-43-2")] == [(241, "name")]
    assert [(r.cid, r.matched_by) for r in store.lookup_compound("CID 2519")] == [(2519, "cid")]
    assert [(r.cid, r.matched_by) for r in store.lookup_compound("H6C6")] == [(241, "formula")]
    assert {r.cid for r in store.lookup_compound("NaCl")} == {5234}  # a PubChem synonym
    assert store.lookup_compound("notacompoundxyz") == []


def test_shared_name_returns_both(built_store):
    assert {r.cid for r in L1Store(built_store).lookup_compound("glucose")} == {5793, 107526}
