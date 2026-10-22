"""Eval set and runner (eval-runner spec), with the fake upstream and the fixture store."""

import json
import re
import sqlite3

import pytest

from conftest import content, finish, tool_call
from evals.run import answer_text, is_correct, load_questions, pick, run_eval, summarize
from haisenberg.agent import Agent
from haisenberg.l1.store import L1Store
from haisenberg.tools import Knowledge
from l1.l1_helpers import real_path


def test_shipped_eval_set():
    sources, questions = load_questions()
    assert len(questions) >= 90
    assert {q["domain"] for q in questions} == {"chemistry", "physics"}
    assert set(sources) == {q["source"] for q in questions}
    assert all(re.search(r"public domain|no restrictions|CC BY", v) for v in sources.values())  # license stated


def test_known_answers_match_the_store():
    """Every CODATA value and PubChem molar mass/formula in the eval set exists in the real store."""
    con = sqlite3.connect(real_path("layer1_encyclopedia_db"))
    _, questions = load_questions()
    for q in questions:
        a = q["answer"]
        if q["source"] == "codata":
            assert con.execute("SELECT 1 FROM constant WHERE abs(value - ?) <= 1e-9 * abs(?)",
                               (a["number"], a["number"])).fetchone(), q["id"]
        elif q["source"] == "pubchem" and "number" in a:
            assert con.execute("SELECT 1 FROM compound WHERE abs(molar_mass - ?) < 0.0006", (a["number"],)).fetchone(), q["id"]
        elif q["source"] == "pubchem":
            assert con.execute("SELECT 1 FROM compound WHERE formula = ?", (a["formula"],)).fetchone(), q["id"]


@pytest.mark.parametrize("answer, gold, ok", [
    ("Ethanol boils at about 78.2 °C [S1].", {"number": 78.37, "abs_tol": 0.5}, True),
    ("Ethanol boils at 79 °C.", {"number": 78.23, "abs_tol": 0.5}, False),
    ("h = 6.626 × 10⁻³⁴ J·s", {"number": 6.62607015e-34, "rel_tol": 0.001}, True),
    ("Absolute zero is −273.15 °C.", {"number": -273.15, "abs_tol": 0.01}, True),
    ("Ethanol is CH3CH2OH.", {"formula": "C2H6O"}, True),
    ("Table salt is NaCl.", {"formula": "ClNa"}, True),
    ("It is Boyle's law.", {"text": ["Boyle"]}, True),
    ("Its symbol is Agx.", {"text": ["Ag"]}, False),
    ("No number here.\n\nSources:\n- [S1] Ethanol: wiki, https://x/?oldid=78", {"number": 78, "abs_tol": 0}, False),
])
def test_is_correct(answer, gold, ok):
    assert is_correct(answer, gold) is ok


def test_answer_text_drops_sources_summary_and_ids():
    text = "A [S1].\n\n*Verification: 1 claim: 1 supported, 0 unverified, 0 corrected, 0 removed.*\n\nSources:\n- [S1] x"
    assert answer_text(text).strip() == "A ."


def test_stratified_subset():
    qs = [{"id": f"{s}{d}{i}", "source": s, "domain": d} for s in "ab" for d in ("chemistry", "physics") for i in range(5)]
    chosen = pick(qs, limit=8)
    assert len(chosen) == 8
    assert {(q["source"], q["domain"]) for q in chosen} == {("a", "chemistry"), ("a", "physics"), ("b", "chemistry"), ("b", "physics")}
    assert [q["id"] for q in pick(qs, ids=["achemistry3", "bphysics0"])] == ["achemistry3", "bphysics0"]


@pytest.mark.anyio
async def test_run_eval_with_fake_backend(cfg, fake, built_store, tmp_path):
    questions = [
        {"id": "q1", "question": "Molar mass of caffeine?", "answer": {"number": 194.194, "rel_tol": 0.002}},
        {"id": "q2", "question": "Boiling point of benzene?", "answer": {"number": 80.1, "abs_tol": 0.5}},
        {"id": "q3", "question": "Boiling point of benzene?", "answer": {"number": 80.1, "abs_tol": 0.5}},
    ]
    fake.scripts = [
        # q1: looked up, cited, supported in code
        [tool_call("lookup_compound", '{"query": "caffeine"}', call_id="c1"), finish("tool_calls")],
        [content("The molar mass of caffeine is 194.19 g/mol [S1].")],
        # q2: wrong, and the judge wrongly supports it -> an unmarked contradiction
        [content("Benzene boils at 90 °C.")], [content("C1: supports S1")],
        # q3: wrong, but left unverified -> marked
        [content("Benzene boils at 90 °C.")], [content("C1: not_said")],
    ]
    agent = Agent(cfg, fake.client(cfg.model_backend), Knowledge(L1Store(built_store), cfg))
    out = tmp_path / "report.json"
    report = await run_eval(agent, questions, out, {"model": "fake"})
    by_id = {r["id"]: r for r in report["results"]}
    assert by_id["q1"]["correct"] and not by_id["q1"]["unmarked_contradiction"]
    assert not by_id["q2"]["correct"] and by_id["q2"]["unmarked_contradiction"]
    assert not by_id["q3"]["correct"] and not by_id["q3"]["unmarked_contradiction"]
    s = report["summary"]
    assert (s["questions"], s["correct_pct"], s["unmarked_contradictions"]) == (3, 33.3, 1)
    assert (s["claims"], s["supported_claims"], s["unverified_claims"], s["supported_claims_pct"]) == (3, 2, 1, 66.7)
    assert json.loads(out.read_text(encoding="utf-8"))["model"] == "fake"


def test_summary_of_errors():
    r = {"correct": False, "unmarked_contradiction": False, "verification": {}, "seconds": 2.0, "error": "down"}
    s = summarize([r])
    assert s["errors"] == 1 and s["correct_pct"] == 0.0 and s["supported_claims_pct"] == 0.0
