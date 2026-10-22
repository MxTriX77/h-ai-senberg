import json
import os
import subprocess
import sys

import pytest

from haisenberg.l1.store import L1Error, L1Store


def search(db, q, k=5):
    return L1Store(db).search_encyclopedia(q, k)


def test_exact_title_first(built_store):
    assert search(built_store, "Planck constant")[0].path == "Planck_constant"


def test_alias_resolves(built_store):
    assert search(built_store, "Heisenberg uncertainty principle")[0].path == "Uncertainty_principle"
    assert search(built_store, "C6H6")[0].path == "Benzene"


def test_keywords_rank_by_bm25(built_store):
    r = search(built_store, "subatomic particle")
    assert r[0].path == "Electron" and "[subatomic]" in r[0].snippet and r[0].score < 0


def test_any_word_fallback(built_store):
    assert {r.path for r in search(built_store, "flammable zzzunknown")} == {"Ethanol"}


def test_hostile_input(built_store):
    for q in ['carbon-14 "decay" AND (half-life):', "NEAR(", "*", '"', "title:iron OR -x"]:
        search(built_store, q)  # no FTS syntax error


def test_empty_and_stopwords(built_store):
    assert search(built_store, "") == [] and search(built_store, "what is the") == []


def test_k_limits_results(built_store):
    assert len(search(built_store, "is", k=1)) <= 1
    assert len(search(built_store, "chemical element compound electron constant", k=2)) <= 2


def test_citation(built_store):
    (r,) = [x for x in search(built_store, "ethanol") if x.path == "Ethanol"]
    c = r.citation
    assert (c.source, c.file, c.version, c.license) == (
        "wikipedia_en_chemistry_fixture", "kiwix/chem_fixture.zim", "2026-10-15", "CC BY-SA 4.0"
    )
    assert c.locator == {"path": "Ethanol", "revision": 100}
    assert c.url.endswith("title=Ethanol&oldid=100")
    assert json.loads(json.dumps(r.to_dict())) == r.to_dict()


def test_cli_search_prints_citation(built_store):
    out = subprocess.run(
        [sys.executable, "-m", "haisenberg.l1.cli", "search", "planck constant", "--db", str(built_store)],
        capture_output=True,
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
    )
    text = out.stdout.decode("utf-8")
    assert out.returncode == 0 and text.startswith("Planck constant (Planck_constant)")
    assert "oldid=400" in text and "CC BY-SA 4.0" in text


def test_domain_filter(built_store):
    store = L1Store(built_store)
    physics = store.search_encyclopedia("electron", 5, "physics")
    assert physics and all("physics" in r.domains for r in physics)
    assert store.search_encyclopedia("ethanol", 5, "chemistry")[0].path == "Ethanol"
    assert store.search_encyclopedia("ethanol", 5, "physics") == []  # Ethanol is chemistry only
    assert store.search_encyclopedia("electron", 5, "biology") == []
    assert store.search_encyclopedia("Planck constant", 5, "chemistry") == []  # exact title, wrong domain


def test_records_carry_source_and_domains(built_store):
    store = L1Store(built_store)
    assert store.search_encyclopedia("Electron")[0].domains == ["chemistry", "physics"]
    caffeine = store.lookup_compound("caffeine")[0]
    assert caffeine.citation.source == "pubchem_mass_fixture" and caffeine.domains == ["chemistry"]
    assert store.lookup_constant("Planck constant")[0].domains == ["physics", "chemistry"]


def test_kinds_and_lookup_by_kind(built_store):
    store = L1Store(built_store)
    assert store.kinds() == {"compound": ["chemistry"], "constant": ["chemistry", "physics"],
                             "text": ["chemistry", "physics"]}
    assert store.lookup("compound", "caffeine") == store.lookup_compound("caffeine")
    assert store.lookup("constant", "electron mass", 2) == store.lookup_constant("electron mass", 2)
    for kind in ("text", "nope"):
        with pytest.raises(L1Error, match=kind):
            store.lookup(kind, "x")


def test_entity_phrase_ranks_second(built_store):
    # 'benzene' names Benzene, which also contains 'boiling' and 'point': it outranks the BM25 order
    assert [r.path for r in search(built_store, "benzene boiling point")][:2] == ["Benzene", "Deuterated_benzene"]
    assert search(built_store, "boiling point of benzene")[0].path == "Benzene"
    # Benzene lacks 'deuterium', so the phrase does not promote it
    assert search(built_store, "benzene deuterium")[0].path == "Deuterated_benzene"
    # the domain filter applies to phrase matches too
    assert all(r.path != "Benzene" for r in L1Store(built_store).search_encyclopedia("benzene boiling point", 5, "physics"))


def test_phrases():
    from haisenberg.l1.store import _phrases

    got = dict(_phrases("Boiling point of ethanol?"))
    assert got["boiling point"] == 2 and got["ethanol"] == 1 and got["point of ethanol"] == 3
    assert "boiling point of ethanol" not in got  # the full query is the exact match
    assert "of ethanol" not in got and "boiling point of" not in got and "of" not in got
    assert dict(_phrases("half-life of carbon-14"))["carbon-14"] == 1
