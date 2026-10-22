"""Checks on the real store (skipped unless retrieval.layer1_encyclopedia_db exists, e.g. via DATA_DIR)."""

import statistics
import time

import pytest

from haisenberg.l1.store import L1Store

from .l1_helpers import real_path

SEARCH = [
    "planck constant", "speed of light", "heisenberg uncertainty principle", "ideal gas law", "covalent bond",
    "benzene", "boiling point of ethanol", "electron charge", "second law of thermodynamics", "photoelectric effect",
    "avogadro constant", "half-life radioactive decay", "electronegativity", "noble gas", "schrodinger equation",
    "special relativity time dilation", "periodic table", "oxidation state", "black-body radiation", "caffeine",
    "superconductivity critical temperature", "xenon tetrafluoride", "acid dissociation constant", "entropy",
]
COMPOUNDS = [
    "caffeine", "benzene", "71-43-2", "water", "sodium chloride", "CID 2244", "C6H12O6", "NaCl", "heavy water",
    "sulfuric acid", "7732-18-5", "ozone", "H2O",
]
CONSTANTS = [
    "Planck constant", "speed of light in vacuum", "electron mass", "Avogadro constant", "Boltzmann constant",
    "elementary charge", "fine-structure constant", "Newtonian constant of gravitation", "proton mass", "molar gas constant",
]


@pytest.fixture(scope="module")
def store():
    return L1Store(real_path("layer1_encyclopedia_db"))


def p95_ms(fn, queries, rounds=5):
    for q in queries:
        fn(q)  # warm up
    times = []
    for _ in range(rounds):
        for q in queries:
            t = time.perf_counter()
            fn(q)
            times.append((time.perf_counter() - t) * 1000)
    return statistics.quantiles(times, n=20)[18]


@pytest.mark.parametrize("lookup, queries", [
    ("search_encyclopedia", SEARCH), ("lookup_compound", COMPOUNDS), ("lookup_constant", CONSTANTS)
])
def test_latency_p95_under_10ms(store, lookup, queries):
    p95 = p95_ms(getattr(store, lookup), queries)
    print(f"{lookup}: p95 {p95:.2f} ms over {len(queries)} queries")
    assert p95 < 10


def test_domain_search_latency_p95_under_10ms(store):
    p95 = p95_ms(lambda q: store.search_encyclopedia(q, None, "physics"), SEARCH)
    print(f"search_encyclopedia(domain=physics): p95 {p95:.2f} ms over {len(SEARCH)} queries")
    assert p95 < 10


def test_known_facts(store):
    (caffeine, *_) = store.lookup_compound("caffeine")
    assert caffeine.cid == 2519 and caffeine.molar_mass == pytest.approx(194.19, abs=0.01)
    heavy = store.lookup_compound("heavy water")[0]
    assert heavy.cid == 24602 and heavy.molar_mass is None and "isotope" in heavy.molar_mass_note
    assert store.lookup_compound("71-43-2")[0].cid == 241
    (me, *_) = store.lookup_constant("electron mass")
    assert me.name == "electron mass" and me.value == pytest.approx(9.1093837139e-31) and me.unit == "kg"
    assert store.search_encyclopedia("planck constant")[0].path == "Planck_constant"
    assert store.search_encyclopedia("benzene")[0].path == "Benzene"  # from the all-articles mini ZIM


def test_no_person_articles(store):
    for path in ("Albert_Einstein", "Marie_Curie", "Max_Planck", "Werner_Heisenberg", "Richard_Feynman"):
        assert store.con.execute("SELECT count(*) FROM article WHERE path = ?", (path,)).fetchone()[0] == 0
