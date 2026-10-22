"""Eval set and runner (eval-runner spec), with the fake upstream and the fixture store."""

import json
import re
import sqlite3

import pytest

from conftest import content, finish, tool_call
from evals.run import (
    answer_lines,
    answer_text,
    is_correct,
    load_problems,
    load_questions,
    part_correct,
    pick,
    read_logs,
    run_eval,
    score_parts,
    select,
    summarize,
)
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


def log(**over):
    """A minimal item log, as run_one writes it."""
    return {"id": "x", "set": "S1", "mode": "config", "answer": "a", "error": None, "verification": {},
            "timings": {"total_s": 1.0}, "model_calls": 1, "correct": None, "scores": None, "artefacts": [],
            "unmarked_contradiction": False, **over}


@pytest.mark.anyio
async def test_run_eval_with_fake_backend(cfg, fake, built_store, tmp_path):
    questions = [
        {"id": "q1", "set": "S1", "question": "Molar mass of caffeine?", "answer": {"number": 194.194, "rel_tol": 0.002}},
        {"id": "q2", "set": "S1", "question": "Boiling point of benzene?", "answer": {"number": 80.1, "abs_tol": 0.5}},
        {"id": "q3", "set": "S1", "question": "Boiling point of benzene?", "answer": {"number": 80.1, "abs_tol": 0.5}},
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
    summary = await run_eval(agent, questions, tmp_path / "run")
    by_id = {r["id"]: r for r in read_logs(tmp_path / "run")}
    assert by_id["q1"]["correct"] and not by_id["q1"]["unmarked_contradiction"]
    assert not by_id["q2"]["correct"] and by_id["q2"]["unmarked_contradiction"]
    assert not by_id["q3"]["correct"] and not by_id["q3"]["unmarked_contradiction"]
    s = summary["all"]
    assert (s["items"], s["correct_pct"], s["unmarked_contradictions"]) == (3, 33.3, 1)
    assert (s["claims"], s["supported_claims"], s["unverified_claims"], s["supported_claims_pct"]) == (3, 2, 1, 66.7)
    assert json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8")) == summary
    assert set(summary["sets"]) == {"S1"}


def test_summary_of_errors():
    s = summarize([log(error="down", answer="")])
    assert s["errors"] == 1 and s["empty_answers"] == 0 and s["supported_claims_pct"] is None
    assert s["latency_median_s"] is None



# --- reasoning problems ---------------------------------------------------------------------------


def test_shipped_reasoning_set():
    problems = load_problems()
    tuning = [p for p in problems if not p.get("holdout")]
    assert len(tuning) >= 25 and len(problems) - len(tuning) >= 3
    assert {p["domain"] for p in problems} == {"chemistry", "physics"}
    assert sum(p["lang"] == "ru" for p in problems) >= 4
    topics = {p["topic"] for p in problems}
    for needed in ("stoichiometry", "unit-conversion", "kinematics", "circuits", "gas-laws", "ph", "equilibrium",
                   "inconsistent-data", "claim-triage", "error-finding", "constraints"):
        assert needed in topics, needed
    for p in problems:  # every part's label is asked for in the question
        assert all(f"{part['label']}:" in p["question"] for part in p["parts"]), p["id"]


@pytest.mark.parametrize("part, error", [
    ({"label": "A1", "number": 1, "text": ["x"]}, "exactly one"),
    ({"label": "A1", "number": 1}, "abs_tol or rel_tol"),
    ({"label": "A1", "verdict": []}, "non-empty"),
])
def test_malformed_part(tmp_path, part, error):
    bad = {"problems": [{"id": "p1", "domain": "physics", "topic": "t", "lang": "en", "question": "q", "parts": [part]}]}
    path = tmp_path / "r.yaml"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match=f"p1: part A1: .*{error}"):
        load_problems(path)


def test_answer_lines_last_wins_and_markdown():
    text = "Working: A1: 2.5\n\n- **A1:** 2.61 g\n**A2**: 80.5 %\nA3 : salicylic acid [S1]"
    assert answer_lines(text) == {"A1": "2.61 g", "A2": "80.5 %", "A3": "salicylic acid"}


@pytest.mark.parametrize("part, line, lang, ok", [
    ({"number": 21.97, "rel_tol": 0.01}, "21,97 г", "ru", True),
    ({"number": 16.77, "rel_tol": 0.01}, "16,775 л", "ru", True),
    ({"number": 1900, "rel_tol": 0.005}, "1,900 mmHg", "en", True),
    ({"number": 0.0748, "rel_tol": 0.01}, "0,0748", "en", True),
    ({"number": 3.34e22, "rel_tol": 0.005}, "3.34 × 10²² molecules", "en", True),
    ({"number": 2.61, "rel_tol": 0.01}, "about 2.5 g (2.61 g theoretical)", "en", False),
    ({"number": 180, "abs_tol": 0.01}, "180.16 g/mol", "en", False),
    ({"number": 0.0237, "rel_tol": 0.01}, "0.024 M", "en", True),  # the key rounded to the answer's precision
    ({"number": 0.0237, "rel_tol": 0.01}, "0.025 M", "en", False),
    ({"number": 0.0237, "rel_tol": 0.01}, "0.0240 M", "en", False),  # four places claimed: 0.0237 is not 0.0240
    ({"number": 2.6137e-3, "rel_tol": 0.001}, "2.61 × 10^-3", "en", True),
    ({"number": 6.38, "abs_tol": 0.01}, "6,4", "ru", True),
    ({"verdict": ["false", "неверно"]}, "неверно, масса меньше на 1,6 %", "ru", True),
    ({"verdict": ["false"]}, "not false", "en", False),
    ({"verdict": ["false"]}, "❌ False: 1.6 % short", "en", True),
    ({"verdict": ["верно"]}, "неверно", "ru", False),
    ({"verdict": ["oxygen", "O2"]}, "The oxygen (O2) is limiting", "en", True),
    ({"verdict": ["unconfirmed", "no"]}, "not verified", "en", False),
    ({"ids": ["S4", "S5"]}, "S4 and S5", "en", True),
    ({"ids": ["S4", "S5"]}, "S4", "en", False),
    ({"text": ["HCl"]}, "HCl is limiting", "en", True),
    ({"number": 1, "abs_tol": 0}, None, "en", False),
])
def test_part_correct(part, line, lang, ok):
    assert part_correct(part, line, lang) is ok


def test_holdout_and_sets():
    facts = [{"id": "f1", "category": "facts"}]
    problems = [{"id": "r1", "category": "reasoning"}, {"id": "r2", "category": "reasoning", "holdout": True}]
    assert [i["id"] for i in select(facts, problems)] == ["f1", "r1"]
    assert [i["id"] for i in select(facts, problems, "reasoning")] == ["r1"]
    assert [i["id"] for i in select(facts, problems, "facts")] == ["f1"]
    assert [i["id"] for i in select(facts, problems, "all", holdout=True)] == ["r2"]


def test_reasoning_groups_in_a_subset():
    _, facts = load_questions()
    items = facts + [p for p in load_problems() if not p.get("holdout")]
    chosen = pick(items, limit=22)
    groups = {f"{q.get('source', 'reasoning')}/{q['domain']}" for q in chosen}
    assert {"reasoning/chemistry", "reasoning/physics"} <= groups and len(chosen) == 22


@pytest.mark.anyio
async def test_reasoning_problem_end_to_end(cfg, fake, built_store, tmp_path):
    problem = {"id": "r1", "set": "S2", "domain": "chemistry", "topic": "t", "lang": "en",
               "question": "q", "parts": [{"label": "A1", "number": 2.61, "rel_tol": 0.01},
                                          {"label": "A2", "verdict": ["true"]}]}
    fact = {"id": "q1", "set": "S1", "question": "Boiling point of benzene?",
            "answer": {"number": 80.1, "abs_tol": 0.5}}
    fake.scripts = [
        [content("The yield is 2.61 g.\nA1: 2.61 g\nA2: false")], [content("C1: not_said")],
        [content("Benzene boils at 80.1 °C.")], [content("C1: not_said")],
    ]
    agent = Agent(cfg, fake.client(cfg.model_backend), Knowledge(L1Store(built_store), cfg))
    summary = await run_eval(agent, [problem, fact], tmp_path / "run")
    r1 = read_logs(tmp_path / "run")[1]
    assert r1["set"] == "S2" and r1["scores"]["final"]["parts"] == {"A1": True, "A2": False} and not r1["correct"]
    assert r1["scores"]["final"]["verdicts"] == {"A2": False} and r1["scores"]["final"]["missing"] == []
    s = summary["all"]
    assert s["items"] == 2 and s["correct_pct"] == 50.0 and s["parts_right"] == 2 and s["parts_total"] == 3
    assert set(summary["sets"]) == {"S1", "S2"}
    assert summary["sets"]["S2"]["parts_pct"] == 50.0 and summary["sets"]["S1"]["correct_pct"] == 100.0
    assert summary["sets"]["S2"]["verdict_pct"] == 0.0
    assert score_parts("A1: 2.61\nA2: true", problem) == [True, True]


def test_consistency_fixes_counted():
    r = log(verification={"counts": {}, "consistency": [{"old": "a", "new": "b", "why": ""}]})
    assert summarize([r])["consistency_fixes"] == 1


def test_check_fixes_counted():
    r = log(verification={"counts": {}, "checks": [{"checker": "x", "text": "a", "fixed": "b", "note": ""},
                                                   {"checker": "x", "text": "c", "fixed": None, "note": ""}]})
    assert summarize([r])["check_fixes"] == 1
