"""Measurements of an eval run (eval-runner spec "Stage log", "Time budgets by request type", "Automatic domain
drift", "Measurement summary and report"): the item log's fields, the summary's blocks, the report's tables and the
rescoring of the code checks. Synthetic items and the fake upstream."""

import json

import pytest

from conftest import content, finish, tool_call
from evals.bare import Bare
from evals.measure import item_fields, summary_blocks
from evals.report import report
from evals.rescore import main as rescore_main
from evals.rescore import rescore
from evals.run import read_logs, run_eval, run_one, summarize, write_summary
from general_items import MERGE, coding_item, general_log, item, ready, write_general_run, write_items
from haisenberg.agent import Agent
from haisenberg.l1.store import L1Store
from haisenberg.tools import Knowledge
from test_eval_runs import item_log, write_run

WRONG = MERGE.replace("a <= out[-1][1]", "a < out[-1][0]")


def code(text):
    return f"```python\n{text}\n```"


def tool(name, stage="tool"):
    return {"stage": stage, "kind": "tool", "tool": name, "start_s": 0.1, "seconds": 0.01}


def model(stage, seconds, prompt=None, completion=None):
    return {"stage": stage, "kind": "model", "start_s": 0.0, "seconds": seconds, "prompt_tokens": prompt,
            "completion_tokens": completion}


# --- the item log --------------------------------------------------------------------------------------


def test_budget_follows_the_request_type(cfg):
    chat = ready(item("chat-dev-001", request_type="chat"))
    quick = item_fields(chat, [], 10.0, cfg.evals)
    assert (quick["budget_s"], quick["within_budget"]) == (15, True)
    assert item_fields(chat, [], 20.0, cfg.evals)["within_budget"] is False
    audit = item_fields(ready(item("facts-dev-001", request_type="audit")), [], 400.0, cfg.evals)
    assert (audit["budget_s"], audit["within_budget"]) == (600, True)
    free = item_fields(ready(item("math-dev-001", request_type="analytic")), [], 400.0, cfg.evals)
    assert (free["budget_s"], free["within_budget"]) == (None, None)  # no budget: timed, not judged


def test_item_fields_carry_the_items_attributes(cfg):
    q = ready(item("facts-dev-001", lang="ru", request_type="factual", probes=["drift_bait"], offline_base="no"))
    fields = item_fields(q, [], 1.0, cfg.evals)
    assert {k: fields[k] for k in ("category", "request_type", "lang", "probes", "offline_base", "split")} == {
        "category": "facts", "request_type": "factual", "lang": "ru", "probes": ["drift_bait"], "offline_base": "no",
        "split": "dev"}
    old = item_fields({"id": "f1", "set": "S1", "question": "q"}, [tool("search_knowledge")], 1.0, cfg.evals)
    assert old == {"science_calls": 1, "science_research": 0, "drift_auto": None}  # an older set: counted, never judged
    assert item_fields(q, [tool("search_knowledge")], 1.0, None) == {
        **{k: q[k] for k in ("category", "request_type", "lang", "probes", "offline_base", "split")},
        "science_calls": None, "science_research": None, "drift_auto": None, "budget_s": None, "within_budget": None}


def test_drift_is_a_science_call_of_the_model_on_a_non_science_item(cfg):
    calls = [tool("search_knowledge"), tool("search_knowledge", "research"), tool("search_knowledge", "research"),
             tool("calculate"), model("reasoner", 1.0)]
    coding = item_fields(ready(item("coding-dev-001")), calls, 1.0, cfg.evals)
    assert (coding["science_calls"], coding["science_research"], coding["drift_auto"]) == (1, 2, True)
    checked = item_fields(ready(item("coding-dev-002")), calls[1:], 1.0, cfg.evals)
    assert (checked["science_calls"], checked["science_research"], checked["drift_auto"]) == (0, 2, False)
    science = item_fields(ready(item("science-dev-001")), [tool("lookup_constant")], 1.0, cfg.evals)
    assert (science["science_calls"], science["drift_auto"]) == (1, None)  # expected there: drift does not apply
    math = item_fields(ready(item("math-dev-001")), [tool("calculate"), tool("compute")], 1.0, cfg.evals)
    assert (math["science_calls"], math["science_research"], math["drift_auto"]) == (0, 0, False)


@pytest.mark.anyio
async def test_the_verifiers_re_search_is_no_drift(cfg, fake, built_store):
    q = ready(item("coding-dev-001", request_type="chat", prompt="Where does a Python list start?"))
    fake.scripts = [[content("A Python list starts at index 0, so the third element is at index 2."), finish("stop")]]
    agent = Agent(cfg, fake.client(cfg.model_backend), Knowledge(L1Store(built_store), cfg))
    log = await run_one(agent, q, "config", cfg.evals)
    assert [s["stage"] for s in log["stages"] if s["kind"] == "tool"] == ["research"]  # the model called no tool
    assert log["drift_auto"] is False and (log["science_calls"], log["science_research"]) == (0, 1)


@pytest.mark.anyio
async def test_a_run_logs_stages_drift_and_the_unchanged_prompt(cfg, fake, built_store):
    cfg.reasoning_policy.enable_multi_hop_verification = False
    q = ready(item("coding-dev-001", {"type": "numeric", "value": 1369, "tolerance": 6}, request_type="chat",
                   prompt="Why is my list element not found?"))
    fake.scripts = [[tool_call("search_knowledge", '{"query": "element"}', call_id="c1"), finish("tool_calls")],
                    [content("It is at index 1,370 [S1]."), finish("stop")]]
    agent = Agent(cfg, fake.client(cfg.model_backend), Knowledge(L1Store(built_store), cfg))
    log = await run_one(agent, q, "config", cfg.evals)
    assert fake.bodies[0]["messages"][-1] == {"role": "user", "content": q["prompt"]}  # nothing added for scoring
    assert [(s["stage"], s["kind"]) for s in log["stages"]] == [("reasoner", "model"), ("tool", "tool"), ("reasoner", "model")]
    assert (log["science_calls"], log["drift_auto"], log["category"], log["request_type"]) == (1, True, "coding", "chat")
    assert (log["budget_s"], log["within_budget"], log["finish_reason"]) == (15, True, "stop")
    assert log["check"]["final"] == {"status": "found", "passed": True, "detail": ""} and log["correct"] is True
    assert log["scores"] is None and "key" not in log and "reference" not in log  # no answer lines, no key in the log


@pytest.mark.anyio
async def test_python_tests_run_on_the_answer_of_a_run(cfg, fake, tmp_path):
    items = [ready(coding_item("coding-dev-001")), ready(coding_item("coding-dev-002")), ready(item("chat-dev-001"))]
    fake.scripts = [[content(code(MERGE))], [content(code(WRONG))], [content("Hello!")]]
    summary = await run_eval(Bare(cfg, fake.client(cfg.model_backend)), items, tmp_path / "run", "bare",
                             settings=cfg.evals)
    logs = {r["id"]: r for r in read_logs(tmp_path / "run")}
    assert logs["coding-dev-001"]["check"]["final"]["status"] == "passed" and logs["coding-dev-001"]["correct"] is True
    assert logs["coding-dev-002"]["check"]["final"]["status"] == "failed" and logs["coding-dev-002"]["correct"] is False
    assert logs["chat-dev-001"]["check"] is None and logs["chat-dev-001"]["correct"] is None
    assert summary["all"]["python_tests"] == {"items": 2, "passed": 1, "failed": 1, "no_code": 0, "timeout": 0,
                                              "error": 0, "draft_passed": 1, "broken": 0, "repaired": 0}
    assert summary["general"] is True and set(summary["sets"]) == {"coding", "chat"}
    cfg.evals.python_tests.enabled = False
    fake.scripts = [[content(code(MERGE))]]
    off = await run_one(Bare(cfg, fake.client(cfg.model_backend)), items[0], "bare", cfg.evals)
    assert off["check"] is None and off["correct"] is None  # switched off: nothing was executed


# --- the summary ---------------------------------------------------------------------------------------


def test_request_types_against_their_budgets():
    chat = [general_log(item(f"chat-dev-00{n}", request_type="chat"), total=s, budget_s=15, within_budget=s <= 15)
            for n, s in enumerate((10, 20, 30), 1)]
    audit = general_log(item("facts-dev-001", request_type="audit"), total=400, budget_s=600, within_budget=True)
    free = general_log(item("math-dev-001"), total=50)
    failed = general_log(item("chat-dev-009", request_type="chat"), total=999, error="down", budget_s=15)
    types = summarize([*chat, audit, free, failed])["request_types"]
    assert types["chat"] == {"items": 3, "median_s": 20, "p90_s": 30, "budget_s": 15, "within_budget": 1,
                             "median_within_budget": False}
    assert types["audit"] == {"items": 1, "median_s": 400, "p90_s": 400, "budget_s": 600, "within_budget": 1,
                              "median_within_budget": True}
    assert types["analytic"] == {"items": 1, "median_s": 50, "p90_s": 50, "budget_s": None, "within_budget": None,
                                 "median_within_budget": None}


def test_stage_totals_and_answers_not_given():
    a = general_log(item("chat-dev-001"), total=12.0, stages=[model("reasoner", 4.0, 800, 100), model("review", 2.0, 900, 5)])
    b = general_log(item("chat-dev-002"), "", total=8.0, stages=[model("reasoner", 4.0)], finish_reason="length")
    s = summarize([a, b])
    assert s["stages"] == {"reasoner": {"records": 2, "seconds": 8.0, "prompt_tokens": 800, "completion_tokens": 100},
                           "review": {"records": 1, "seconds": 2.0, "prompt_tokens": 900, "completion_tokens": 5}}
    assert (s["no_answer"], s["truncated"], s["total_s"]) == (1, 1, 20.0)
    assert summary_blocks([general_log(item(), stages=[model("bare", 1.0)])])["stages"]["bare"]["prompt_tokens"] is None


def test_drift_and_check_blocks():
    passed, failed = {"status": "passed", "passed": True}, {"status": "failed", "passed": False}
    found, missing = {"status": "found", "passed": True}, {"status": "missing", "passed": False}
    logs = [
        general_log(item("coding-dev-001"), drift_auto=True, science_calls=3, science_research=4,
                    check={"type": "python_tests", "final": failed, "draft": passed}),
        general_log(item("coding-dev-002"), check={"type": "python_tests", "final": passed, "draft": passed}),
        general_log(item("coding-dev-003"), check={"type": "python_tests", "final": {"status": "no_code", "passed": False},
                                                  "draft": {"status": "no_code", "passed": False}}),
        general_log(item("science-dev-001"), science_calls=2, check={"type": "numeric", "final": found, "draft": found}),
        general_log(item("math-dev-001"), check={"type": "numeric", "final": missing, "draft": found}),
        general_log(item("facts-dev-001"), check={"type": "exact", "final": found, "draft": missing}),
    ]
    s = summarize(logs)
    assert s["drift"] == {"items": 5, "auto": 1, "science_calls": 3, "science_research": 4}  # science: not judged
    assert s["python_tests"] == {"items": 3, "passed": 1, "failed": 1, "no_code": 1, "timeout": 0, "error": 0,
                                 "draft_passed": 2, "broken": 1, "repaired": 0}
    assert s["reference_values"] == {"exact": {"items": 1, "found": 1, "draft_found": 0, "lost": 0, "gained": 1},
                                     "numeric": {"items": 2, "found": 1, "draft_found": 2, "lost": 1, "gained": 0}}


def test_an_older_run_keeps_its_summary():
    old = item_log("i1", {"A1": True})
    s = summarize([old])
    for key in ("stages", "request_types", "drift", "python_tests", "reference_values", "no_answer", "total_s"):
        assert key not in s
    assert summary_blocks([old]) == {} and (s["parts_right"], s["latency_median_s"]) == (1, 2.0)


# --- the report ----------------------------------------------------------------------------------------


@pytest.fixture
def runs(tmp_path):
    """A bare run and a suite run of the same four items, and an older run of answer-line items."""
    items = [coding_item("coding-dev-001"), item("chat-dev-001", request_type="chat"),
             item("math-dev-001", {"type": "numeric", "value": 1369, "tolerance": 6}), item("science-dev-001")]
    path = write_items(tmp_path / "items.yaml", *items)
    passed, failed = {"status": "passed", "passed": True}, {"status": "failed", "passed": False}
    found, missing = {"status": "found", "passed": True}, {"status": "missing", "passed": False}

    def logs(mode, seconds, tests, drift, number, answer="fine"):
        return [general_log(items[0], mode=mode, total=seconds, drift_auto=drift, science_calls=int(drift),
                            check={"type": "python_tests", "final": tests, "draft": passed}, correct=tests["passed"]),
                general_log(items[1], answer, mode=mode, total=seconds, budget_s=15, within_budget=seconds <= 15,
                            stages=[model("reasoner", seconds / 2, 100, 10)]),
                general_log(items[2], mode=mode, total=seconds, check={"type": "numeric", "final": number, "draft": found}),
                general_log(items[3], mode=mode, total=seconds, science_calls=1)]

    bare = write_general_run(tmp_path / "run-bare", "bare", logs("bare", 10.0, passed, False, found, ""), path, write_summary)
    suite = write_general_run(tmp_path / "run-suite", "config", logs("config", 30.0, failed, True, missing), path,
                              write_summary)
    old = write_run(tmp_path / "run-old", "D", [item_log("i1", {"A1": True, "A2": False})])
    return bare, suite, old


def test_report_tables_for_the_measurements(runs):
    bare, suite, old = runs
    text = report([bare, suite, old])
    assert "| run-old | S3 | D | 1 | 0 | 0 | 0/1 (0 %) |" in text  # the answer-line table keeps the older run
    assert "| run-bare | all | bare |" not in text.split("\n\n")[0] and "| run-suite | coding" not in text.split("\n\n")[0]
    # no answer, drift, python tests, reference values
    assert ("| run-bare | all | bare | 4 | 0 | 1 | 0 | 0 of 3 | 1 of 1 | 0 / 0 / 0 / 0 | 0 / 0 | – | 1 of 1 | 0 / 0"
            " | 10 / 10 |") in text
    assert ("| run-suite | all | config | 4 | 0 | 0 | 0 | 1 of 3 | 0 of 1 | 1 / 0 / 0 / 0 | 1 / 0 | – | 0 of 1 | 1 / 0"
            " | 30 / 30 |") in text
    assert "| run-suite | coding | config | 1 | 0 | 0 | 0 | 1 of 1 | 0 of 1 |" in text
    # seconds by request type against the budget
    assert "| run-bare | bare | chat | 1 | 10 | 10 | 15 | 1 of 1 | yes |" in text
    assert "| run-suite | config | chat | 1 | 30 | 30 | 15 | 0 of 1 | NO |" in text
    assert "| run-suite | config | analytic | 3 | 30 | 30 | – | – | – |" in text
    # stages: seconds and their share of the run's 120 s
    assert "| run-suite | config | reasoner | 1 | 15 | 12.5 | 100 | 10 |" in text
    assert "| run-old |" not in text.split("\n\n", 1)[1]  # the older run has no measurement tables


def test_report_pairs_the_arms_of_general_items(runs):
    bare, suite, old = runs
    text = report([bare, suite, old], bare)
    assert "Paired against run-bare (items of the general schema):" in text
    assert "| run-suite | all | 4 | 0 → 1 | 1 of 1 → 0 of 1 | 1 of 1 → 0 of 1 | 1 → 0 | +20 | – | – |" in text
    assert "| run-suite | coding | 1 | 0 → 1 | 1 of 1 → 0 of 1 | – → – | 0 → 0 | +20 | – | – |" in text
    assert "Paired against run-bare:" not in text  # the older run shares no item with the base


# --- rescoring the code checks ---------------------------------------------------------------------------


def test_rescore_runs_the_checks_again(tmp_path, capsys, cfg):
    items = [coding_item("coding-dev-001"), item("math-dev-001", {"type": "numeric", "value": 1369, "tolerance": 6}),
             item("chat-dev-001")]
    path = write_items(tmp_path / "items.yaml", *items)
    stale = {"status": "missing", "passed": False, "detail": "an older rule"}
    logs = [general_log(items[0], code(MERGE), check={"type": "python_tests", "final": {"status": "failed", "passed": False},
                                                     "draft": {"status": "failed", "passed": False}}, correct=False),
            general_log(items[1], "About 1,370 books.", check={"type": "numeric", "final": stale, "draft": stale},
                        correct=False),
            general_log(items[2], "Hello!")]
    folder = write_general_run(tmp_path / "run-g", "bare", logs, path, write_summary)
    counts = rescore(folder, write=False, python=cfg.evals.python_tests)
    assert (counts["checks"], counts["checks_before"], counts["checks_after"]) == (2, 0, 2)
    assert read_logs(folder)[2]["check"]["final"] == stale  # a dry run writes nothing (logs come sorted by id)
    assert rescore(folder, python=None)["checks_after"] == 1  # without the settings the python tests stay as stored
    rescore(folder, python=cfg.evals.python_tests)
    chat, coding, math = read_logs(folder)
    assert coding["check"]["final"]["status"] == "passed" and coding["correct"] is True
    assert math["check"]["final"] == {"status": "found", "passed": True, "detail": ""} and math["correct"] is True
    assert math["check_before_rescore"]["final"] == stale and chat["check"] is None and chat["correct"] is None
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))["all"]
    assert summary["python_tests"]["passed"] == 1 and summary["reference_values"]["numeric"]["found"] == 1
    assert rescore_main([str(folder), "--dry-run"]) == 0
    assert "code checks passed 0 → 2 of 2" in capsys.readouterr().out  # the first scores stay the "before"
