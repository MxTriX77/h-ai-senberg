import json
import os
import subprocess
import sys

from haisenberg.l1.store import L1Store


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
