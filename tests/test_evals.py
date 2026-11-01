"""Eval set and runner (eval-runner spec), with the fake upstream and the fixture store."""

import json
import re
import sqlite3

import pytest
import yaml

from conftest import content, finish, tool_call
from evals.run import (
    QUESTIONS,
    answer_lines,
    answer_text,
    is_correct,
    load_problems,
    load_questions,
    part_correct,
    pick,
    read_logs,
    run_eval,
    score,
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
    target = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))["answer_target"]
    assert "A1:" in target and all(q["question"].endswith(f"?\n\n{target}") for q in questions)


def test_questions_need_an_answer_target(tmp_path):
    data = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))
    for target in (None, "Finish with: Answer: <value>"):
        path = tmp_path / "q.yaml"
        path.write_text(yaml.safe_dump({**data, "answer_target": target}), encoding="utf-8")
        with pytest.raises(ValueError, match="q.yaml: needs an answer_target"):
            load_questions(path)


def test_known_answers_match_the_store():
    """Every CODATA value and PubChem molar mass/formula in the eval set exists in the real store."""
    con = sqlite3.connect(real_path("encyclopedia"))
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


BENZENE = {"id": "q", "set": "S1", "question": "bp?", "answer": {"number": 80.1, "abs_tol": 0.5}}


@pytest.mark.parametrize("answer, gold, strict, loose", [
    ("Benzene boils at 80.1 °C, far below 100 °C.\nA1: 90 °C", BENZENE["answer"], False, True),
    ("Benzene boils at 80.1 °C.\n**A1:** 80.1 °C", BENZENE["answer"], True, True),
    ("Benzene boils at 80.1 °C.", BENZENE["answer"], False, True),
    ("Ethanol is CH3CH2OH.\nA1: CH3CH2OH", {"formula": "C2H6O"}, True, True),
    ("Ethanol is CH3CH2OH, not CH3OH.\nA1: CH3OH", {"formula": "C2H6O"}, False, True),
    ("It is Boyle's law.\nA1: Charles's law", {"text": ["Boyle"]}, False, True),
    ("A1: Boyle's law", {"text": ["Boyle"]}, True, True),
    ("The electron mass is 9.109 383 7139 × 10^-31 kg.\nA1: 9.109 383 7139 × 10^-31 kg",
     {"number": 9.1093837139e-31, "rel_tol": 0.001}, True, True),  # CODATA grouping, last group of four
    ("A1: 2 up quarks and 1 down quark", {"text": ["two up quarks and one down quark", "uud"]}, True, True),
    ("Two up quarks and one down quark.\nA1: 3 up quarks and 1 down quark",
     {"text": ["two up quarks and one down quark"]}, False, True),
])
def test_fact_score_strict_and_loose(answer, gold, strict, loose):
    s = score(answer, {**BENZENE, "answer": gold})
    assert (s["correct"], s["parts"], s["loose_correct"]) == (strict, {"answer": strict}, loose)
    assert s["missing"] == ([] if "A1:" in answer else ["answer"])


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
        [content("The molar mass of caffeine is 194.19 g/mol [S1].\nA1: 194.19 g/mol")],
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
    assert (s["facts"], s["facts_correct"], s["loose_correct"]) == (3, 1, 1)
    assert (s["claims"], s["supported_claims"], s["unverified_claims"], s["supported_claims_pct"]) == (3, 2, 1, 66.7)
    assert [by_id[q]["unsourced"] for q in ("q1", "q2", "q3")] == [0, 1, 1]  # 90 °C: the record gives 80.1 °C
    assert s["unsourced_per_answer"] == 0.67
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


@pytest.mark.parametrize("text, lines", [
    ("A1: верно A2: неверно A3: не подтверждено", {"A1": "верно", "A2": "неверно", "A3": "не подтверждено"}),
    ("Итог:\nA1: inconsistent A2: 0.167 mol A3: 0.200 mol",
     {"A1": "inconsistent", "A2": "0.167 mol", "A3": "0.200 mol"}),
    ("**A1:** 22.0 g **A2:** 16.8 L", {"A1": "22.0 g", "A2": "16.8 L"}),
    ("А1: верно, А2：неверно", {"A1": "верно,", "A2": "неверно"}),  # Cyrillic А, full-width colon
    ("A1: 2.5\nA1: 2.61 g A2: 80.5 %\nA2: 81 %", {"A1": "2.61 g", "A2": "81 %"}),  # last occurrence wins
])
def test_several_answer_lines_on_one_line(text, lines):
    assert answer_lines(text) == lines


@pytest.mark.parametrize("text, lines", [
    ("A1: 0.5 mol\nSee note: A2 is the mass\nA2: 16 g", {"A1": "0.5 mol", "A2": "16 g"}),
    ("A1: See note: A2 is 16 g", {"A1": "See note: A2 is 16 g"}),  # no colon after A2
    ("A1: 0.5 mol (A2: 16 g)", {"A1": "0.5 mol (A2: 16 g)"}),  # not after a space
    ("A2: mass of O2: 16 g", {"A2": "mass of O2: 16 g"}),  # a formula, not a label of the line's letter
    ("A1: S4: wrong unit, S5: wrong sum", {"A1": "S4: wrong unit, S5: wrong sum"}),
])
def test_colons_in_text_do_not_split(text, lines):
    assert answer_lines(text) == lines


def test_one_line_answer_scored_per_part():
    problem = {"lang": "en", "parts": [{"label": "A1", "number": 22.0, "rel_tol": 0.01},
                                      {"label": "A2", "number": 16.8, "rel_tol": 0.01}]}
    assert score_parts("A1: 22.0 g A2: 16.8 L", problem) == [True, True]


def test_cyrillic_look_alike_labels():
    text = "\u04101: 2,61 г\n**\u0410\u0032**\uff1a верно\n\u04303: Масса\n\u04121: x"  # А1, А2 full-width colon, а3, В1
    assert answer_lines(text) == {"A1": "2,61 г", "A2": "верно", "A3": "Масса", "B1": "x"}  # values untouched
    problem = {"lang": "ru", "parts": [{"label": "A1", "number": 2.61, "rel_tol": 0.01}, {"label": "A2", "verdict": ["верно"]}]}
    assert score_parts(text, problem) == [True, True]


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
    ({"number": 0.0237, "rel_tol": 0.01}, "0.02 M", "en", False),  # one significant figure is not enough
    ({"number": 2.61, "rel_tol": 0.01}, "3 g", "en", False),
    ({"number": 2.61, "rel_tol": 0.001}, "2.6 g", "en", True),
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
    ({"text": ["two up quarks and one down quark"]}, "2 up quarks and 1 down quark", "en", True),  # counts: digits
    ({"text": ["2 up quarks and 1 down quark"]}, "Two up quarks and one down quark", "en", True),  # or words
    ({"text": ["two up quarks and one down quark"]}, "3 up quarks and 1 down quark", "en", False),
    ({"text": ["два u-кварка и один d-кварк"]}, "2 u-кварка и 1 d-кварк", "ru", True),
    ({"text": ["из двух u-кварков"]}, "из 3 u-кварков", "ru", False),
    ({"text": ["someone"]}, "some1", "en", False),  # only whole words are counts
    ({"number": 1234.5, "abs_tol": 0}, "1 234,5", "en", True),  # grouping and decimal comma as before
    ({"number": 1234.5, "abs_tol": 0}, "1,234.5", "en", True),
    ({"number": 1234.5, "abs_tol": 0}, "1 234,5", "ru", True),
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
        [content("Benzene boils at 80.1 °C.\nA1: 80.1 °C")], [content("C1: not_said")],
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


# --- verdict parts read from the answer body ---------------------------------------------------------


AUDIT = {"lang": "en", "question": (
    "Check each claim.\nC1. The sum is 12.\nC2. The sample weighs 4 g.\nC3. The yield is 90 %.\n\n"
    "Finish with exactly these lines:\nA1: <total mass in g>\nA2: <verdict for C1>\nA3: <verdict for C2>\n"
    "A4: <verdict for C3>\nA5: <mass of the sample in C2, in g>\n"),
    "parts": [{"label": "A1", "number": 4, "abs_tol": 0.01}, {"label": "A2", "verdict": ["incorrect"]},
              {"label": "A3", "verdict": ["correct"]}, {"label": "A4", "verdict": ["unverified"]},
              {"label": "A5", "number": 4, "abs_tol": 0.01}]}
OFFSET = "A1: 4 g\nA2: Correct\nA3: Unverified\nA4: Incorrect\nA5: 5 g"  # each verdict line one item off


@pytest.mark.parametrize("body", [
    "**C1.** The sum is 12.\nIt is 11.\nVerdict: Incorrect — it is 11.\n\n**C2.** 4 g.\n**Verdict:** Correct\n\n"
    "### C3 — the yield\nVerdict: Unverified, no data.\n\n---\n",
    "**C1:** Incorrect — 11, not 12.\n**C2:** Correct — 4 g as stated.\n- C3: Unverified (no mass given).\n",
    "| Claim | Verdict | Why |\n|---|---|---|\n| C1 | Incorrect | 11 |\n| C2 | Correct | 4 g |\n| C3 | Unverified | – |\n",
])
def test_offset_lines_with_right_sections_score_from_the_body(body):
    s = score(body + "\n" + OFFSET, AUDIT)
    assert s["parts"] == {"A1": True, "A2": True, "A3": True, "A4": True, "A5": False}  # A5: a number, its line
    assert s["line_parts"] == {"A1": True, "A2": False, "A3": False, "A4": False, "A5": False}
    assert s["sources"] == {"A1": "line", "A2": "body", "A3": "body", "A4": "body", "A5": "line"}
    assert s["verdicts"] == {"A2": True, "A3": True, "A4": True}


@pytest.mark.parametrize("body", [
    "**C1:** Incorrect — 11.\n**C2:** Correct.\n**C3:** Unverified.\n\nSummary\n- C1: Correct\n",  # two sections
    "| C1 | Correct | Incorrect: 11 |\n| C2 | Correct | – |\n| C3 | Unverified | – |\n",  # two verdicts in a row
    "**C1:** The sum is 12, so it looks fine at first.\n**C2:** Correct.\n**C3:** Unverified.\n",  # no verdict
])
def test_unclear_section_falls_back_to_the_line(body):
    s = score(body + "\n" + OFFSET, AUDIT)
    assert s["sources"]["A2"] == "line" and s["parts"]["A2"] is False
    assert s["sources"]["A3"] == s["sources"]["A4"] == "body"


def test_no_item_placeholders_scores_as_before():
    problem = {"lang": "en", "question": "C1. A claim.\nFinish with:\nA1: <verdict>\nA2: <mass in g>",
               "parts": [{"label": "A1", "verdict": ["false"]}, {"label": "A2", "number": 2.61, "rel_tol": 0.01}]}
    s = score("**C1:** False — wrong mass.\n\nA1: true\nA2: 2.61 g", problem)
    assert s["parts"] == s["line_parts"] == {"A1": False, "A2": True}
    assert set(s["sources"].values()) == {"line"}


def test_russian_sections_with_verdict_labels():
    item = {"lang": "ru", "question": "Проверь утверждения.\nУ1) Масса 4 г.\nУ2) Объём 2 л.\nЗакончи ответ строками:\n"
                                      "A1: <вердикт для У1>\nA2: <вердикт для У2>\n",
            "parts": [{"label": "A1", "verdict": ["верно"]}, {"label": "A2", "verdict": ["неверно"]}]}
    answer = ("**У1)** Масса 4 г.\nРасчёт: 4,0 г.\n**Вердикт:** верно\n\n**У2)** Объём 2 л.\n**Вывод:** неверно — 3 л.\n\n"
              "A1: неверно\nA2: верно")
    s = score(answer, item)
    assert s["parts"] == {"A1": True, "A2": True} and s["line_parts"] == {"A1": False, "A2": False}


def test_label_that_is_an_item_reads_its_section():
    item = {"lang": "en", "question": "C1. The sum is 12.\nC2. The sample weighs 4 g.\n\nFinish with these lines, in "
                                      "claim order:\nC1: <Correct / Incorrect / Unverified>\nC2: <Correct / Incorrect "
                                      "/ Unverified>\n",
            "parts": [{"label": "C1", "verdict": ["incorrect"]}, {"label": "C2", "verdict": ["correct"]}]}
    body = "### C1 — the sum\nVerdict: Incorrect — it is 11.\n\n### C2 — the mass\nVerdict: Correct\n\n"
    s = score(body + "C1: Correct\nC2: Correct", item)  # the closing lines are no section of their item
    assert s["parts"] == {"C1": True, "C2": True} and s["line_parts"] == {"C1": False, "C2": True}
    assert s["sources"] == {"C1": "body", "C2": "body"}
    lines_only = score("C1: Correct — 11.\n\nC2: Correct", item)  # nothing but the closing lines: read there
    assert lines_only["parts"] == {"C1": False, "C2": True} and set(lines_only["sources"].values()) == {"line"}


def test_quoted_claim_after_the_id_is_no_verdict():
    item = {"lang": "en", "question": 'S1. NaOH + HCl: "neutralisation".\nS2. Fe2O3 + 3 CO → 2 Fe + 3 CO2: "carbon '
                                      'monoxide is the oxidising agent".\nFinish with:\nA1: <the reducing agent in S2>\n',
            "parts": [{"label": "A1", "verdict": ["CO", "carbon monoxide"]}]}
    answer = ('**S2:** "carbon monoxide is the oxidising agent" is **wrong**. CO is the **reducing agent**.\n\n'
              "A1: Fe2O3")
    s = score(answer, item)
    assert s["parts"] == {"A1": False} and s["sources"] == {"A1": "line"}
    after = score('**S2:** "Fe2O3 is the reducing agent" — carbon monoxide, not Fe2O3.\n\nA1: Fe2O3', item)
    assert after["parts"] == {"A1": True} and after["sources"] == {"A1": "body"}  # the text after the quote counts


def test_summary_counts_body_reads_and_line_only_parts():
    s = score("**C1:** Incorrect.\n**C2:** Correct.\n**C3:** Unverified.\n\n" + OFFSET, AUDIT)
    old = {"correct": True, "parts": {"A1": True}, "missing": [], "verdicts": {}}  # scored before body reading
    summary = summarize([log(set="S3", scores={"final": s, "draft": s}), log(scores={"final": old, "draft": old})])
    assert (summary["parts_right"], summary["line_parts_right"], summary["body_verdict_parts"]) == (5, 2, 3)


def test_consistency_fixes_counted():
    r = log(verification={"counts": {}, "consistency": [{"old": "a", "new": "b", "why": ""}]})
    assert summarize([r])["consistency_fixes"] == 1


def test_unsourced_numbers_per_answer():
    assert summarize([log(unsourced=2), log(unsourced=1), log()])["unsourced_per_answer"] == 1.5  # flag off: not counted
    assert summarize([log(), log(unsourced=None)])["unsourced_per_answer"] is None


def test_check_fixes_counted():
    r = log(verification={"counts": {}, "checks": [{"checker": "x", "text": "a", "fixed": "b", "note": ""},
                                                   {"checker": "x", "text": "c", "fixed": None, "note": ""}]})
    assert summarize([r])["check_fixes"] == 1
