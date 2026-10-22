"""Evidence protocol (evidence-protocol spec): ledger parsing, status rules, coverage repair, the answer written
from the ledger and reviewed against it, and the fallbacks."""

import json

import pytest

from conftest import content
from haisenberg.agent import Agent
from haisenberg.evidence import (
    CONTRADICTED, ESTABLISHED, HYPOTHESIS, UNVERIFIED, Entry, apply_rules, missing, norm_id, parse_ledger,
)
from haisenberg.tools import Knowledge

WORDS = ["always", "guarantee", "гарантир"]
TASK = "Check these claims:\nA) the method always works\nB) the total is 5"


def entry(i="A", status=ESTABLISHED, support=("given",), against=(), **kw):
    return Entry(id=i, statement=kw.pop("statement", "x"), status=status, claimed=status,
                 support=[{"evidence": "e", "from": f} for f in support],
                 against=[{"evidence": "e", "from": f} for f in against], conclusion="said", **kw)


def ledger_json(*items, skipped=()):
    return json.dumps({"items": list(items), "skipped": list(skipped)})


ITEM_A = {"id": "A", "statement": "the method always works", "for": [{"evidence": "one run worked", "from": "given"}],
          "against": [], "status": "established", "strong_enough": False, "conclusion": "It works."}
ITEM_B = {"id": "B", "statement": "the total is 5", "for": [],
          "against": [{"evidence": "2 + 2 = 4", "from": "computation"}], "status": "unverified", "conclusion": "Unclear."}


def test_norm_id():
    assert [norm_id(v) for v in ("Claim A", "a)", " 2. ", "", None)] == ["A", "A", "2", "", "NONE"]


def test_parse_ledger():
    text = "```json\n" + ledger_json(
        {"id": "a)", "status": "Established", "for": [{"evidence": "e", "from": "magic"}, "plain text"],
         "depends_on": ["b"], "strong_enough": "yes"},
        {"statement": "no id"},
        {"id": "B", "status": "maybe"},
        skipped=[{"id": "3.", "why": "given data"}]) + "\n```"
    entries, skipped = parse_ledger(text)
    a, b = entries
    assert (a.id, a.status, a.claimed, a.depends_on, a.strong_enough) == ("A", ESTABLISHED, ESTABLISHED, ["B"], False)
    assert a.support == [{"evidence": "e", "from": "knowledge"}, {"evidence": "plain text", "from": "knowledge"}]
    assert (b.id, b.status) == ("B", UNVERIFIED) and skipped == {"3": "given data"}
    assert parse_ledger("no json here") is None and parse_ledger('{"items": 3}') is None
    assert parse_ledger('{"items": [') is None


def test_hard_evidence_against_contradicts():
    for against in ("computation", "given", "source"):
        e = entry(against=(against,))
        apply_rules([e], WORDS)
        assert e.status == CONTRADICTED and e.rule.startswith("established -> contradicted") and e.conclusion == ""
    e = entry(status=UNVERIFIED, support=(), against=("computation",))
    apply_rules([e], WORDS)
    assert e.status == CONTRADICTED  # settled by the computation: not "unconfirmed"


def test_absence_of_evidence_is_unverified():
    e, f = entry(status=CONTRADICTED, support=()), entry(support=())
    apply_rules([e, f], WORDS)
    assert e.status == UNVERIFIED and f.status == UNVERIFIED
    k = entry(status=CONTRADICTED, against=("knowledge",))
    apply_rules([k], WORDS)
    assert k.status == CONTRADICTED and k.rule == ""


def test_no_hypothesis_promotion():
    a, h = entry(support=("assumption",)), entry(status=HYPOTHESIS, support=("computation",))
    apply_rules([a, h], WORDS)
    assert a.status == HYPOTHESIS and h.status == HYPOTHESIS and h.rule == ""


def test_strong_claims_need_strong_evidence():
    weak = entry(statement="It always works")
    strong = entry(statement="It always works", strong_enough=True)
    ru = entry(statement="Метод гарантирует чистоту")
    plain = entry(statement="It works")
    apply_rules([weak, strong, ru, plain], WORDS)
    assert weak.status == UNVERIFIED and "'always'" in weak.rule and ru.status == UNVERIFIED
    assert strong.status == ESTABLISHED and plain.status == ESTABLISHED


def test_dependencies():
    a = entry("A", against=("computation",))
    b = entry("B", depends_on=["A"])
    c = entry("C", depends_on=["B", "C", "Z"])  # itself and unknown ids are ignored
    apply_rules([c, b, a], WORDS)
    assert (a.status, b.status, c.status) == (CONTRADICTED, UNVERIFIED, UNVERIFIED)
    assert "depends on A, which is contradicted" in b.rule


def test_missing():
    assert missing([entry("A")], {"C": "data"}, ["A", "B", "C"]) == ["B"]


async def run(cfg, fake, text=TASK):
    cfg.router.enabled = True
    agent = Agent(cfg, fake.client(cfg.model_backend), Knowledge.open(cfg))  # no store: no tools, no verifier
    items = [i async for i in agent.run([{"role": "user", "content": text}])]
    answer = "".join(i.get("content", "") for i in items)
    trace = "".join(i.get("reasoning_content", "") for i in items)
    return answer, trace, next(i["verification"] for i in items if "verification" in i)


@pytest.mark.anyio
async def test_answer_written_from_the_settled_ledger(cfg, fake):
    fake.scripts = [[content("Draft: A works once; B is 2 + 2 = 4.")], [content(ledger_json(ITEM_A, ITEM_B))],
                    [content("A cannot be confirmed from one run. B is wrong: 2 + 2 = 4.")], [content("OK")]]
    answer, trace, report = await run(cfg, fake)
    assert answer == "A cannot be confirmed from one run. B is wrong: 2 + 2 = 4."
    ledger, writer, review = fake.bodies[1:]
    assert ledger["messages"][0]["content"] == cfg.evidence.ledger_prompt
    assert ledger["response_format"] == {"type": "json_object"}
    assert "Items listed in the task:\nA: the method always works\nB: the total is 5" in ledger["messages"][1]["content"]
    assert writer["messages"][0]["content"] == cfg.evidence.writer_prompt
    told = writer["messages"][1]["content"]
    assert "status: unverified (changed by the rules: established -> unverified: a strong claim ('always')" in told
    assert "status: contradicted (changed by the rules" in told and "Draft:\nDraft: A works once" in told
    assert review["messages"][0]["content"] == cfg.evidence.review_prompt
    assert "Route: analytic" in trace and "Evidence ledger" in trace and "Ledger review: 0 fixes" in trace
    items = {i["id"]: i for i in report["ledger"]["items"]}
    assert (items["A"]["status"], items["A"]["claimed"]) == (UNVERIFIED, ESTABLISHED)
    assert items["B"]["status"] == CONTRADICTED and report["ledger"]["missing"] == []


@pytest.mark.anyio
async def test_missing_items_are_asked_for_once(cfg, fake):
    item_b = {**ITEM_B, "against": [], "status": "established", "for": [{"evidence": "e", "from": "given"}]}
    fake.scripts = [[content("draft")], [content(ledger_json(ITEM_A))], [content(ledger_json(item_b, ITEM_A))],
                    [content("answer")], [content("OK")]]
    _, trace, report = await run(cfg, fake)
    repair = fake.bodies[2]["messages"]
    assert repair[-2]["role"] == "assistant" and repair[-1]["content"] == cfg.evidence.repair_prompt.replace(
        "{missing}", "B")
    assert "no entry for B; asking again" in trace
    assert [i["id"] for i in report["ledger"]["items"]] == ["A", "B"] and report["ledger"]["missing"] == []


@pytest.mark.anyio
async def test_still_missing_is_reported(cfg, fake):
    fake.scripts = [[content("draft")], [content(ledger_json(ITEM_A))], [content("sorry")],
                    [content("answer")], [content("OK")]]
    _, trace, report = await run(cfg, fake)
    assert report["ledger"]["missing"] == ["B"] and "still no entry for B" in trace


@pytest.mark.anyio
async def test_written_answer_fixed_in_code_and_by_the_review(cfg, fake):
    fake.scripts = [[content("draft")], [content(ledger_json(ITEM_A, ITEM_B))],
                    [content("B is wrong: 2 × 3 = 7. A is surely true.")],
                    [content("OLD: A is surely true.\nNEW: A cannot be confirmed.\nWHY: status unverified")]]
    answer, trace, report = await run(cfg, fake)
    assert answer == "B is wrong: 2 × 3 = 6. A cannot be confirmed."
    assert "- arithmetic: 2 * 3 = 7 → 6" in trace
    assert report["ledger"]["fixes"] == [{"old": "A is surely true.", "new": "A cannot be confirmed.",
                                          "why": "status unverified"}]


@pytest.mark.anyio
async def test_unreadable_ledger_keeps_the_draft(cfg, fake):
    fake.scripts = [[content("The draft answer.")], [content("I cannot do JSON")]]
    answer, trace, report = await run(cfg, fake)
    assert answer == "The draft answer." and "no readable ledger (not valid JSON)" in trace
    assert "ledger" not in report and len(fake.requests) == 2


@pytest.mark.anyio
async def test_no_written_answer_reviews_the_draft(cfg, fake):
    fake.scripts = [[content("The draft answer.")], [content(ledger_json(ITEM_A, ITEM_B))], [content(" ")],
                    [content("OK")]]
    answer, trace, _ = await run(cfg, fake)
    assert answer == "The draft answer." and "no answer written (empty reply)" in trace
    assert "Answer:\nThe draft answer." in fake.last_body["messages"][1]["content"]


@pytest.mark.anyio
@pytest.mark.parametrize("ledger, last_system", [
    (ledger_json(ITEM_A, ITEM_B), "review_prompt"),  # the ledger review replaces the consistency review
    ("not json", "consistency_prompt"),  # no readable ledger: the usual consistency review
], ids=["ledger", "no-ledger"])
async def test_with_claim_checks(cfg, fake, built_store, ledger, last_system):
    from haisenberg.l1.store import L1Store
    from test_verifier import DRAFT

    cfg.router.enabled, cfg.verifier.consistency_check = True, True
    agent = Agent(cfg, fake.client(cfg.model_backend), Knowledge(L1Store(built_store), cfg))
    writer = [[content("written")]] if last_system == "review_prompt" else []
    fake.scripts = [[content(DRAFT)], [content("C1: not_said")], [content(ledger)], *writer, [content("OK")]]
    items = [i async for i in agent.run([{"role": "user", "content": TASK}])]
    systems = [b["messages"][0]["content"] for b in fake.bodies]
    assert systems[-1] == getattr(cfg.evidence if last_system == "review_prompt" else cfg.verifier, last_system)
    assert systems.count(cfg.verifier.consistency_prompt) == (last_system == "consistency_prompt")
    report = next(i["verification"] for i in items if "verification" in i)
    assert report["claims"] and ("ledger" in report) == (last_system == "review_prompt")


@pytest.mark.anyio
async def test_protocol_off_or_not_analytic(cfg, fake):
    cfg.evidence.enabled = False
    fake.scripts = [[content("plain")]]
    answer, _, report = await run(cfg, fake)
    assert answer == "plain" and len(fake.requests) == 1 and report == {}
    cfg.evidence.enabled = True
    fake.scripts = [[content("Ethanol boils at 78 °C.")]]
    answer, _, _ = await run(cfg, fake, "What is the boiling point of ethanol?")
    assert answer == "Ethanol boils at 78 °C." and len(fake.requests) == 2  # factual: no ledger call
