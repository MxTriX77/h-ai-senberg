"""Evidence ledger (evidence-protocol spec): listed items, ledger parsing, the status rules R-a … R-e (with the c7
shapes), coverage repair, the writer, the fallbacks, and the unchanged path when it is off or not triggered."""

import json

import pytest

from conftest import CONNECT_ERROR, content
from haisenberg.agent import DRAFT_MARK, Agent, _tool_results
from haisenberg.evidence import (
    CONTRADICTED, ESTABLISHED, HYPOTHESIS, UNVERIFIED, Entry, Evidence, Settled, key, listed_items, parse_ledger,
    settle,
)
from haisenberg.config import load_config
from haisenberg.tools import Knowledge

WORDS = ["guarantee", "independent", "гарантир", "независим"]
TASK = ("Judge each claim (true / false / unverified):\nC1. The batch weighed 5 kg.\n"
        "C2. One batch guarantees an 85 % yield at any scale.\nC3. 2 × 3 = 7.")
DRAFT = "C1 is true. C2 is unverified. C3 is false: 2 × 3 = 6."


def entry(i="A", status=ESTABLISHED, support=("given",), against=(), **kw):
    return Entry(id=i, statement=kw.pop("statement", "x"), status=status, claimed=status,
                 support=[{"evidence": "e", "from": f} for f in support],
                 against=[{"evidence": "e", "from": f} for f in against], **kw)


def item(i, status, asserts="fact", support=(), against=(), strong=True, **kw):
    return {"id": i, "statement": kw.pop("statement", f"statement {i}"), "asserts": asserts,
            "for": [{"evidence": "e", "from": f} for f in support],
            "against": [{"evidence": "e", "from": f} for f in against],
            "strong_enough": strong, "status": status, "reason": kw.pop("reason", "because"), **kw}


def ledger(*items, skipped=()):
    return json.dumps({"items": list(items), "skipped": list(skipped)})


SETTLED = ledger(item("C1", "established", support=("given",)),
                 item("C2", "contradicted", "proof", support=("given",), strong=False),
                 item("C3", "contradicted", "computed", against=("computation",)))
V2 = ledger(item("C1", "established", support=("given",)),
            item("C2", "unverified", "proof", support=("given",), strong=False, reason="one batch cannot predict"),
            item("C3", "contradicted", "computed", against=("computation",)))


# --- listed items ---------------------------------------------------------------------------------


def test_listed_items_keep_the_ids_as_written():
    assert listed_items(TASK) == [("C1", "The batch weighed 5 kg."),
                                  ("C2", "One batch guarantees an 85 % yield at any scale."), ("C3", "2 × 3 = 7.")]
    assert listed_items("У1) Выход 85 %.\n**У2:** Метод безопасен.\n- У3. Партия одна.") == [
        ("У1", "Выход 85 %."), ("У2", "Метод безопасен."), ("У3", "Партия одна.")]
    assert listed_items("1. a\n2) b\n(3) c\n1.1 not an item") == [("1", "a"), ("2", "b"), ("3", "c")]
    assert listed_items("а) первое\nб) второе") == [("а", "первое"), ("б", "второе")]
    assert listed_items("Claim A: x\nClaim 2. y\nMass m: 5 g") == [("Claim A", "x"), ("Claim 2", "y")]
    assert listed_items("C1. x\nC1: the same id again") == [("C1", "x")]
    # a line that is only a placeholder is a form to fill in, not an item to judge
    assert listed_items("C1. x\nC2. y\nFinish with:\nA1: <verdict for C1>\nA2: <число>.") == [("C1", "x"), ("C2", "y")]


def test_bullets_count_only_without_ids():
    assert listed_items("Check:\n- a\n* b\n• c") == [("1", "a"), ("2", "b"), ("3", "c")]
    assert listed_items("Given:\n- m = 5 g\n- V = 2 L\n- T = 300 K\nC1. x\nC2. y") == [("C1", "x"), ("C2", "y")]
    assert listed_items("What is the boiling point of ethanol?") == []


def test_key():
    assert [key(v) for v in ("Claim A", "a)", " C1. ", "С1", "")] == ["a", "a", "c1", "c1", ""]  # С1: Cyrillic С


# --- ledger ---------------------------------------------------------------------------------------


def test_parse_ledger():
    listed = [("Claim A", "x"), ("C1", "y"), ("C2", "z")]
    text = "```json\n" + ledger(
        {"id": "a", "status": "Established", "asserts": "magic", "for": [{"evidence": "e", "from": "magic"}, "plain"],
         "depends_on": ["c1"], "strong_enough": "yes"},
        {"statement": "no id"}, {"id": "С1", "status": "maybe"}, {"id": "C1", "status": "established"},
        skipped=[{"id": "C2", "why": "data"}, "C9"]) + "\n```"
    entries, skipped = parse_ledger(text, listed)
    a, c1 = entries
    assert (a.id, a.status, a.claimed, a.asserts, a.depends_on, a.strong_enough) == (
        "Claim A", ESTABLISHED, ESTABLISHED, "fact", ["C1"], False)
    assert a.support == [{"evidence": "e", "from": "knowledge"}, {"evidence": "plain", "from": "knowledge"}]
    assert (c1.id, c1.status) == ("C1", UNVERIFIED) and skipped == ["C2", "C9"]  # first C1 (Cyrillic) wins
    assert parse_ledger("no json here", listed) is None and parse_ledger('{"items": 3}', listed) is None
    assert parse_ledger('{"items": [', listed) is None


# --- rules ----------------------------------------------------------------------------------------


def changed(*entries, words=WORDS):
    return [(e.id, e.status, e.rule) for e in settle(list(entries), words)]


def test_ra_hard_evidence_against_contradicts():
    for source in ("given", "source", "computation"):
        e = entry(against=(source,), strong_enough=True)
        assert changed(e) == [("A", CONTRADICTED, "R-a")] and e.claimed == ESTABLISHED
    assert changed(entry(status=UNVERIFIED, support=(), against=("computation",))) == [("A", CONTRADICTED, "R-a")]
    assert changed(entry(status=CONTRADICTED, against=("knowledge",))) == []


def test_rb_claim_of_proof_with_evidence_that_cannot_give_it_is_contradicted():
    assert changed(entry(status=UNVERIFIED, asserts="proof")) == [("A", CONTRADICTED, "R-b")]
    assert changed(entry(status=ESTABLISHED, statement="One run guarantees it")) == [("A", CONTRADICTED, "R-b")]
    assert changed(entry(status=UNVERIFIED, statement="Метод гарантирует чистоту")) == [("A", CONTRADICTED, "R-b")]
    assert changed(entry(asserts="proof", strong_enough=True)) == []  # the evidence gives it all


def test_scope_words_alone_never_make_a_claim_false():
    # "every", "therefore", "поэтому" are not proof words in the shipped list: a plain claim the model can't fully
    # support is only lowered (R-d), never contradicted by R-b
    words = load_config().evidence.strong_words
    for statement in ("Every sample was therefore tested at 25 °C", "Поэтому каждая партия годна"):
        e = entry(status=ESTABLISHED, statement=statement)
        assert settle([e], words) == [e] and (e.status, e.rule) == (UNVERIFIED, "R-d")


def test_rb_claim_of_proof_without_the_material_is_unverified():
    e = entry(status=CONTRADICTED, support=(), asserts="proof", gaps=["the certificate is not attached"])
    assert changed(e) == [("A", UNVERIFIED, "R-b")]
    # the report only mentions the certificate: what it refers to is missing, so not shown, not false
    e = entry(status=CONTRADICTED, asserts="proof", statement="The certificate confirms 99 % purity",
              gaps=["the certificate"])
    assert changed(e) == [("A", UNVERIFIED, "R-b")]
    assert changed(entry(status=UNVERIFIED, support=(), asserts="proof")) == []  # no gaps: the other rules
    assert changed(entry(status=CONTRADICTED, support=(), asserts="proof")) == [("A", UNVERIFIED, "R-c")]


def test_rc_absence_of_evidence_is_not_disproof():
    assert changed(entry(status=CONTRADICTED, support=())) == [("A", UNVERIFIED, "R-c")]


def test_rd_established_needs_evidence_at_full_strength():
    assert changed(entry(support=(), strong_enough=True)) == [("A", UNVERIFIED, "R-d")]
    assert changed(entry(support=("assumption",), strong_enough=True)) == [("A", HYPOTHESIS, "R-d")]
    assert changed(entry(support=("given",))) == [("A", UNVERIFIED, "R-d")]  # not strong_enough
    assert changed(entry(support=("given", "assumption"), strong_enough=True)) == []


def test_re_dependencies_to_a_fixed_point():
    a = entry("A", against=("computation",))
    b = entry("B", depends_on=["A"], strong_enough=True)
    c = entry("C", depends_on=["B", "C", "Z"], strong_enough=True)  # itself and unknown ids are ignored
    assert changed(c, b, a) == [("C", UNVERIFIED, "R-e"), ("B", UNVERIFIED, "R-e"), ("A", CONTRADICTED, "R-a")]
    assert b.why == "it depends on A, which is contradicted"


def test_no_promotion():
    assert changed(entry(status=HYPOTHESIS, support=("computation",), strong_enough=True)) == []
    assert changed(entry(status=UNVERIFIED, support=("given",), strong_enough=True)) == []


# The c7 shapes (wrong verdicts on main, all already wrong in the draft), as the ledger would record them.

def test_c7_v1_not_shown_called_false():
    en = entry("C6", status=CONTRADICTED, support=(), asserts="cause", statement="The supplement lowers the risk",
               gaps=["no controlled comparison"], reason="the study does not prove causation")
    ru = entry("У4", status=CONTRADICTED, support=("given",), asserts="cause", statement="Добавка снижает риск",
               reason="наблюдательное исследование не доказывает причинность")
    assert changed(en, ru) == [("C6", UNVERIFIED, "R-c"), ("У4", UNVERIFIED, "R-c")]


def test_c7_v2_claim_of_proof_called_unverified():
    en = entry("R7", status=UNVERIFIED, asserts="proof", statement="The 1000 h test proves a 10-year life",
               reason="one test cannot predict the lifetime")
    ru = entry("У5", status=UNVERIFIED, statement="Повторный анализ той же выборки: независимая проверка",
               reason="не доказывает")
    assert changed(en, ru) == [("R7", CONTRADICTED, "R-b"), ("У5", CONTRADICTED, "R-b")]


def test_c7_v3_right_calculation_on_a_wrong_input():
    e = entry("C5", support=("computation",), against=("given",), strong_enough=True,
              statement="The plant makes 7,500 t a year", reason="calculation correct despite the wrong input")
    assert changed(e) == [("C5", CONTRADICTED, "R-a")]


def test_c7_ratio_offered_as_proof_of_cause():
    e = entry("C5", support=("computation",), asserts="cause", statement="The new catalyst caused the 20 % rise")
    assert changed(e) == [("C5", UNVERIFIED, "R-d")]


# --- flow -----------------------------------------------------------------------------------------


async def run(cfg, fake, text=TASK, enabled=True):
    cfg.evidence.enabled = enabled
    agent = Agent(cfg, fake.client(cfg.model_backend), Knowledge.open(cfg))  # no store: no tools, no verifier
    items = [i async for i in agent.run([{"role": "user", "content": text}])]
    answer = "".join(i.get("content", "") for i in items)
    trace = "".join(i.get("reasoning_content", "") for i in items)
    return answer, trace, next(i["verification"] for i in items if "verification" in i), items


@pytest.mark.anyio
@pytest.mark.parametrize("text, enabled", [
    (TASK, False), ("Judge:\nC1. x\nC2. y", True), ("What is the boiling point of ethanol?", True),
], ids=["off", "two-items", "no-list"])
async def test_off_or_not_triggered_is_unchanged(cfg, fake, text, enabled):
    fake.scripts = [[content(DRAFT)]]
    *_, items = await run(cfg, fake, text, enabled)
    assert items == [{"reasoning_content": DRAFT_MARK}, {"reasoning_content": DRAFT}, {"draft": DRAFT},
                     {"content": DRAFT}, {"verification": {}}, {"finish_reason": "stop"}]  # main's output
    assert len(fake.requests) == 1


@pytest.mark.anyio
async def test_min_items(cfg, fake):
    cfg.evidence.min_items = 2
    fake.scripts = [[content(DRAFT)], [content(ledger(item("C1", "established", support=("given",)),
                                                      item("C2", "unverified")))]]
    _, trace, report, _ = await run(cfg, fake, "Judge:\nC1. x\nC2. y")
    assert len(fake.requests) == 2 and "Evidence ledger: 2 listed items" in trace and "ledger" in report


@pytest.mark.anyio
async def test_no_change_keeps_the_draft_with_one_extra_call(cfg, fake):
    fake.scripts = [[content(DRAFT)], [content(SETTLED)]]
    answer, trace, report, _ = await run(cfg, fake)
    assert answer == DRAFT and len(fake.requests) == 2
    body = fake.bodies[1]
    assert body["messages"][0]["content"] == cfg.evidence.ledger_prompt
    assert body["response_format"] == {"type": "json_object"} and body["temperature"] == 0.0
    user = body["messages"][1]["content"]
    assert user.startswith(f"Conversation:\nuser: {TASK}\n\nItems listed in the last message:\nC1: The batch weighed")
    assert user.endswith(f"Draft answer:\n{DRAFT}") and "Tool results" not in user
    assert "3 items, no status changed: the answer stays as drafted" in trace
    assert report["ledger"] == {"items": [
        {"id": "C1", "asserts": "fact", "claimed": "established", "status": "established", "rule": ""},
        {"id": "C2", "asserts": "proof", "claimed": "contradicted", "status": "contradicted", "rule": ""},
        {"id": "C3", "asserts": "computed", "claimed": "contradicted", "status": "contradicted", "rule": ""}],
        "skipped": [], "missing": [], "written": False, "fallback": ""}


@pytest.mark.anyio
async def test_answer_written_from_the_settled_ledger(cfg, fake):
    fake.scripts = [[content(DRAFT)], [content(V2)], [content("C1 is true. C2 is false. C3 is false: 2 × 3 = 6.")]]
    answer, trace, report, _ = await run(cfg, fake)
    assert answer == "C1 is true. C2 is false. C3 is false: 2 × 3 = 6." and len(fake.requests) == 3
    writer = fake.bodies[2]
    assert writer["messages"][0]["content"] == cfg.evidence.writer_prompt and "response_format" not in writer
    told = writer["messages"][1]["content"]
    assert told.startswith(f"Request:\n{TASK}\n\nSettled ledger:\nC1: statement C1\n  status: established\n")
    assert ("C2: statement C2\n  status: contradicted (settled by the rules: claim of proof the evidence can't give;"
            " the ledger first said unverified)\n  for (given): e\n") in told
    assert "one batch cannot predict" not in told and told.endswith(f"Draft answer:\n{DRAFT}")
    assert "Evidence: C2 contradicted (claim of proof the evidence can't give; model said unverified)" in trace
    assert report["ledger"]["items"][1] == {"id": "C2", "asserts": "proof", "claimed": "unverified",
                                            "status": "contradicted", "rule": "R-b"}
    assert report["ledger"]["written"] is True


@pytest.mark.anyio
async def test_missing_items_are_asked_for_once(cfg, fake):
    first = ledger(item("C1", "established", support=("given",)), item("C2", "unverified"))
    fake.scripts = [[content(DRAFT)], [content(first)], [content(ledger(item("c3", "unverified")))]]
    answer, trace, report, _ = await run(cfg, fake)
    repair = fake.bodies[2]["messages"]
    assert repair[-2] == {"role": "assistant", "content": first}
    assert repair[-1]["content"] == cfg.evidence.repair_prompt.replace("{missing}", "C3")
    assert "no entry for C3; asking again" in trace and answer == DRAFT and len(fake.requests) == 3
    assert [i["id"] for i in report["ledger"]["items"]] == ["C1", "C2", "C3"] and report["ledger"]["missing"] == []


@pytest.mark.anyio
async def test_still_missing_goes_to_the_writer_as_not_yet_judged(cfg, fake):
    first = ledger(item("C1", "established", support=("given",)), skipped=[{"id": "C2", "why": "data"}])
    fake.scripts = [[content(DRAFT)], [content(first)], [content("sorry")], [content("written")]]
    answer, trace, report, _ = await run(cfg, fake)
    assert answer == "written" and "still no entry for C3 (not yet judged)" in trace
    told = fake.bodies[3]["messages"][1]["content"]
    assert "Skipped (data or instructions): C2\nNot yet judged: C3\n" in told
    assert report["ledger"]["missing"] == ["C3"] and report["ledger"]["skipped"] == ["C2"]


@pytest.mark.anyio
@pytest.mark.parametrize("ledger_reply, writer_reply, why", [
    ([content("I cannot do JSON")], None, "no readable ledger (not valid JSON)"),
    (CONNECT_ERROR, None, "no readable ledger (failed: backend unreachable"),
    ([content(V2)], [content("  ")], "no answer written (empty reply)"),
    ([content(V2)], CONNECT_ERROR, "no answer written (failed: backend unreachable"),
], ids=["unreadable", "ledger-error", "empty-writer", "writer-error"])
async def test_fallbacks_keep_the_draft(cfg, fake, ledger_reply, writer_reply, why):
    fake.scripts = [[content(DRAFT)], ledger_reply, *([writer_reply] if writer_reply else [])]
    answer, trace, report, _ = await run(cfg, fake)
    assert answer == DRAFT and why in trace and report["ledger"]["fallback"].startswith(why)
    assert report["ledger"]["written"] is False and len(fake.requests) == 2 + bool(writer_reply)


@pytest.mark.anyio
async def test_too_long_for_the_context_keeps_the_draft(cfg, fake):
    cfg.evidence.ledger_request = {**cfg.evidence.ledger_request, "max_tokens": cfg.context_tokens - 100}
    fake.scripts = [[content(DRAFT)]]
    answer, trace, report, _ = await run(cfg, fake)
    assert answer == DRAFT and len(fake.requests) == 1 and "no readable ledger (too long for the context)" in trace


def evidence(cfg, fake, checks=False):
    return Evidence(cfg, fake.client(cfg.model_backend), checks)


async def settle_run(ev, draft=DRAFT, tools=""):
    items = [i async for i in ev.run([{"role": "user", "content": TASK}], listed_items(TASK), draft, tools)]
    return next(i for i in items if isinstance(i, Settled)), "".join(i.get("reasoning_content", "")
                                                                     for i in items if isinstance(i, dict))


@pytest.mark.anyio
@pytest.mark.parametrize("checks", [True, False])
async def test_written_answer_is_checked_in_code_as_the_draft(cfg, fake, checks):
    fake.scripts = [[content(V2)], [content("C2 is false. C3 is false: 2 × 3 = 7.")]]
    settled, trace = await settle_run(evidence(cfg, fake, checks))
    assert settled.answer == ("C2 is false. C3 is false: 2 × 3 = " + ("6." if checks else "7."))
    assert ("- arithmetic:" in trace) == checks


@pytest.mark.anyio
async def test_tool_results_go_to_the_ledger_and_give_way_to_the_context(cfg, fake):
    fake.scripts = [[content(SETTLED)], [content(SETTLED)]]
    await settle_run(evidence(cfg, fake), tools="> calculate(expressions=[\"2*3\"])\n2*3 = 6")
    assert "\n\nTool results:\n> calculate(expressions=[\"2*3\"])\n2*3 = 6\n\nDraft answer:\n" in (
        fake.bodies[0]["messages"][1]["content"])
    _, trace = await settle_run(evidence(cfg, fake), tools="x" * 30000)
    assert "tool results left out (context budget)" in trace and "Tool results" not in fake.last_body["messages"][1][
        "content"]


def test_tool_results_text():
    messages = [{"role": "user", "content": "q"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_0", "type": "function",
                     "function": {"name": "calculate", "arguments": '{"expressions": ["2*3"]}'}}]},
                {"role": "tool", "tool_call_id": "call_0", "content": "2*3 = 6"}]
    assert _tool_results(messages) == '> calculate(expressions=["2*3"])\n2*3 = 6'
    assert _tool_results(messages[:1]) == ""
