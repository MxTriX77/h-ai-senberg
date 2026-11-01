"""Evidence ledger (evidence-protocol spec): listed items, ledger parsing, the status rules R-a … R-e (with the c7
shapes), coverage repair, the writer, the fallbacks, the unchanged path when it is off or not triggered, and the
independent judgment per item (R-f)."""

import json

import pytest

from conftest import CONNECT_ERROR, content, reasoning
from haisenberg.agent import DRAFT_MARK, Agent, _tool_results
from haisenberg.evidence import (
    CONTRADICTED, ESTABLISHED, HYPOTHESIS, UNVERIFIED, Entry, Evidence, Settled, compare, key, listed_items,
    parse_entry, parse_ledger, settle,
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


# --- independent judgment (R-f) -------------------------------------------------------------------


def judged(i, status, asserts="fact", support=(), against=(), strong=True, **kw):
    """An independent reply: thinking, a few words, then the entry as fenced JSON."""
    return [reasoning(f"Let me work {i} out {{step by step}}."),
            content(f"About {i}: see below.\n```json\n"
                    + json.dumps(item(i, status, asserts, support, against, strong, **kw)) + "\n```")]


AGREE = [judged("C1", "established", support=("given",)),
         judged("C2", "contradicted", "proof", support=("given",), strong=False),
         judged("C3", "contradicted", "computed", against=("computation",))]


async def run_independent(cfg, fake, text=TASK):
    cfg.evidence.independent = True
    return await run(cfg, fake, text)


@pytest.mark.anyio
async def test_independent_agreement_changes_nothing_and_writes_nothing(cfg, fake):
    fake.scripts = [[content(DRAFT)], [content(SETTLED)], *AGREE]
    answer, trace, report, _ = await run_independent(cfg, fake)
    assert answer == DRAFT and len(fake.requests) == 5  # draft, ledger, one call per item, no writer
    assert "Independent judgment: 3 items, one call each" in trace and "3 judged independently, 0 disagree" in trace
    led = report["ledger"]
    assert led["disagreements"] == 0 and led["independent_calls"] == 3 and led["written"] is False
    assert led["items"][1] == {"id": "C2", "asserts": "proof", "claimed": "contradicted", "status": "contradicted",
                               "rule": "", "independent": {"status": "contradicted", "claimed": "contradicted",
                                                           "rule": ""}}


@pytest.mark.anyio
async def test_independent_disagreement_makes_the_item_unverified_and_the_writer_runs(cfg, fake):
    written = "C1 cannot be confirmed. C2 is false. C3 is false: 2 × 3 = 6."
    fake.scripts = [[content(DRAFT)], [content(SETTLED)], judged("C1", "unverified"), *AGREE[1:], [content(written)]]
    answer, trace, report, _ = await run_independent(cfg, fake)
    assert answer == written and len(fake.requests) == 6
    assert "Evidence: C1 unverified (independent judgment disagrees: unverified; model said established)" in trace
    assert ("C1: statement C1\n  status: unverified (settled by the rules: independent judgment disagrees: unverified;"
            " the ledger first said established)") in fake.bodies[5]["messages"][1]["content"]
    led = report["ledger"]
    assert led["items"][0] == {"id": "C1", "asserts": "fact", "claimed": "established", "status": "unverified",
                               "rule": "R-f", "independent": {"status": "unverified", "claimed": "unverified",
                                                              "rule": ""}}
    assert led["disagreements"] == 1 and led["written"] is True


def test_compare_rf_then_re():
    a, b = entry("A", strong_enough=True), entry("B", depends_on=["A"], strong_enough=True)
    c = entry("C", status=UNVERIFIED, support=())
    own = {"a": entry("A", status=UNVERIFIED, support=()), "b": entry("B", strong_enough=True),
           "c": entry("C", status=CONTRADICTED, against=("given",)), "d": entry("D", status=HYPOTHESIS)}
    entries = [a, b, c]
    disagree, added = compare(entries, own)
    assert (disagree, [e.id for e in added]) == (2, ["D"]) and entries[-1] is own["d"]
    assert [(e.id, e.status, e.rule) for e in (a, b, c)] == [
        ("A", UNVERIFIED, "R-f"), ("B", UNVERIFIED, "R-e"), ("C", UNVERIFIED, "")]  # C was already not settled
    assert a.why == "independent judgment disagrees: unverified"


@pytest.mark.anyio
async def test_a_missing_ledger_item_takes_its_independent_entry(cfg, fake):
    first = ledger(item("C1", "established", support=("given",)),
                   item("C2", "contradicted", "proof", support=("given",), strong=False))
    fake.scripts = [[content(DRAFT)], [content(first)], [content("sorry")], *AGREE, [content("written")]]
    answer, trace, report, _ = await run_independent(cfg, fake)
    assert answer == "written" and len(fake.requests) == 7  # draft, ledger, repair, 3 items, writer
    assert "still no entry for C3 (not yet judged)" in trace and "C3 judged independently only" in trace
    told = fake.bodies[6]["messages"][1]["content"]
    assert "C3: statement C3\n  status: contradicted\n  against (computation): e\n" in told
    assert "Not yet judged" not in told and report["ledger"]["missing"] == []
    assert report["ledger"]["items"][2] == {"id": "C3", "asserts": "computed", "claimed": "contradicted",
                                            "status": "contradicted", "rule": "", "independent": {
                                                "status": "contradicted", "claimed": "contradicted", "rule": ""}}


@pytest.mark.anyio
async def test_independent_calls_never_see_the_draft_or_the_ledger(cfg, fake):
    fake.scripts = [[content(SETTLED)], *AGREE]
    cfg.evidence.independent = True
    tools = "> calculate(expressions=[\"2*3\"])\n2*3 = 6"
    settled, _ = await settle_run(evidence(cfg, fake), tools=tools)
    assert settled.answer == DRAFT and len(fake.requests) == 4
    for n, (iid, text) in enumerate(listed_items(TASK), 1):
        body = fake.bodies[n]
        system, user = (m["content"] for m in body["messages"])
        assert system == cfg.evidence.item_prompt.replace("{id}", iid).replace("{text}", text)
        assert user == f"Conversation:\nuser: {TASK}\n\nTool results:\n{tools}"
        assert DRAFT not in system + user and SETTLED not in system + user and "Draft answer" not in user
        assert "response_format" not in body and "chat_template_kwargs" not in body  # thinking on, no JSON mode
        assert body["temperature"] == cfg.model_backend.temperature  # the model's default request


@pytest.mark.anyio
async def test_an_unreadable_or_failed_independent_reply_is_ignored(cfg, fake):
    fake.scripts = [[content(DRAFT)], [content(SETTLED)], [content("C1 looks fine to me.")], CONNECT_ERROR,
                    judged("C3", "maybe")]
    answer, trace, report, _ = await run_independent(cfg, fake)
    assert answer == DRAFT and len(fake.requests) == 5
    assert "C1: no readable independent judgment (no JSON object with a status); ignored" in trace
    assert "C2: no readable independent judgment (failed: backend unreachable" in trace
    led = report["ledger"]
    assert [i["independent"] for i in led["items"]] == [None, None, None] and led["disagreements"] == 0
    assert led["independent_calls"] == 3


@pytest.mark.anyio
async def test_independent_cap_and_skipped_items(cfg, fake):
    cfg.evidence.independent_max_items = 1
    first = ledger(item("C1", "established", support=("given",)),
                   item("C3", "contradicted", "computed", against=("computation",)), skipped=[{"id": "C2"}])
    fake.scripts = [[content(DRAFT)], [content(first)], AGREE[0]]
    answer, trace, report, _ = await run_independent(cfg, fake)
    assert answer == DRAFT and len(fake.requests) == 3  # C2 is skipped, C3 is over the limit
    assert "Independent judgment: 1 items, one call each\n- C3: over the limit of 1; the ledger's judgment stays" in trace
    assert [i["independent"] is None for i in report["ledger"]["items"]] == [False, True]


@pytest.mark.anyio
async def test_independent_off_is_the_ledger_alone(cfg, fake):
    fake.scripts = [[content(DRAFT)], [content(V2)], [content("written")]]
    off_answer, off_trace, off_report, _ = await run(cfg, fake)
    off_bodies = fake.bodies
    fake.requests.clear()
    fake.scripts = [[content(DRAFT)], [content(V2)], *AGREE, [content("written")]]
    on_answer, on_trace, _, _ = await run_independent(cfg, fake)
    assert len(off_bodies) == 3 and fake.bodies[:2] == off_bodies[:2] and off_answer == on_answer == "written"
    assert "independent_calls" not in off_report["ledger"] and "independent" not in off_report["ledger"]["items"][0]
    section = on_trace[on_trace.index("Independent judgment"):on_trace.index("Evidence: C2")]
    assert "Independent judgment" not in off_trace and on_trace.replace(section, "") == off_trace


@pytest.mark.anyio
async def test_independent_ids_in_russian(cfg, fake):
    task = "Оцените утверждения:\nУ1) Партия весила 5 кг.\nУ2) Одна партия гарантирует выход 85 %.\nУ3) 2 × 3 = 7."
    draft = "У1 верно. У2 верно. У3 неверно."
    first = ledger(item("У1", "established", support=("given",)), item("У2", "established", support=("given",)),
                   item("У3", "contradicted", "computed", against=("computation",)))
    # У2's own judgment: a proof stem and evidence that can't give it → contradicted by R-b; the ledger said
    # established: the two differ, so У2 is not settled. The ids in the replies are misspelt: the item's id wins.
    fake.scripts = [[content(draft)], [content(first)], judged("Y1", "established", support=("given",)),
                    judged("y2", "established", statement="Одна партия гарантирует выход 85 %.", support=("given",),
                           strong=False),
                    judged("У3", "contradicted", "computed", against=("computation",)), [content("переписано")]]
    answer, trace, report, _ = await run_independent(cfg, fake, task)
    assert answer == "переписано" and fake.bodies[3]["messages"][0]["content"].startswith(
        "Judge one item of the user's last message, У2: Одна партия гарантирует выход 85 %.\n")
    assert "Evidence: У2 unverified (independent judgment disagrees: contradicted; model said established)" in trace
    led = report["ledger"]
    assert [i["id"] for i in led["items"]] == ["У1", "У2", "У3"] and led["disagreements"] == 1
    assert led["items"][1]["independent"] == {"status": "contradicted", "claimed": "established", "rule": "R-b"}


def test_parse_entry():
    listed = [("C1", "x"), ("C2", "the batch guarantees it")]
    reply = "Work: {a} and \\frac{1}{2}.\n```json\n" + json.dumps(
        {"id": "nonsense", "status": "Established", "for": [{"evidence": "e", "from": "given"}],
         "depends_on": ["c1"], "strong_enough": True}) + "\n```\nDone."
    e = parse_entry(reply, ("C2", "the batch guarantees it"), listed)
    assert (e.id, e.statement, e.status, e.claimed, e.depends_on, e.strong_enough) == (
        "C2", "the batch guarantees it", ESTABLISHED, ESTABLISHED, ["C1"], True)
    one = parse_entry(ledger(item("C9", "contradicted", against=("given",))), ("C1", "x"), listed)
    assert (one.id, one.status, one.statement) == ("C1", CONTRADICTED, "statement C9")  # a one-item ledger
    for bad in ("no json", '{"status": "maybe"}', '{"items": []}', '{"id": "C1"}', '{"status": "est'):
        assert parse_entry(bad, ("C1", "x"), listed) is None
