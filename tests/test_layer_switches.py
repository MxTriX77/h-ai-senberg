"""The suite's layer switches (measure-suite): the system and tool prompt, tools by name, the per-call thinking
budget, the record check and the claim judge, the writer, and the removal of [S#] markers. Every switch ships on:
the tests of each module cover that state; these tests cover the off state."""

import pytest

from conftest import content, finish, reasoning, tool_call
from haisenberg.agent import Agent, unscaled_room
from haisenberg.evidence import Evidence, Settled
from haisenberg.l1.store import L1Store
from haisenberg.tools import Knowledge, Sources
from haisenberg.verifier import UNVERIFIED, Verification, Verifier
from test_evidence import DRAFT, SETTLED, TASK, V2

pytestmark = pytest.mark.anyio

ASK = [{"role": "user", "content": "What is the molar mass of caffeine?"}]
LOOKUP = [tool_call("lookup_compound", '{"query": "caffeine"}', call_id="c1"), finish("tool_calls")]
KNOWLEDGE_TOOLS = ["search_knowledge", "lookup_compound", "lookup_constant"]


@pytest.fixture
def store(built_store):
    return L1Store(built_store)


async def run(agent, messages=ASK):
    items = [item async for item in agent.run(messages)]
    return ("".join(i.get("content", "") for i in items), "".join(i.get("reasoning_content", "") for i in items),
            next(i["verification"] for i in items if "verification" in i))


def agent(cfg, fake, store=None):
    return Agent(cfg, fake.client(cfg.model_backend), Knowledge(store, cfg))


def names(body):
    return [t["function"]["name"] for t in body.get("tools", [])]


# --- system prompt and tool prompt -------------------------------------------------------------------


async def test_suite_prompt_off_and_no_bare_line_means_no_system_message(cfg, fake):
    cfg.reasoning_policy.use_system_prompt = False
    fake.scripts = [[content("fine")]]
    answer, *_ = await run(agent(cfg, fake))
    assert answer == "fine" and fake.last_body["messages"] == ASK


async def test_bare_line_replaces_the_system_prompt(cfg, fake):
    cfg.reasoning_policy.use_system_prompt, cfg.reasoning_policy.bare_system_prompt = False, "Be brief."
    fake.scripts = [[content("fine")]]
    await run(agent(cfg, fake))
    assert fake.last_body["messages"] == [{"role": "system", "content": "Be brief."}, *ASK]


async def test_tools_without_the_tool_prompt(cfg, fake, store):
    p = cfg.reasoning_policy
    p.use_tool_prompt = p.enable_multi_hop_verification = False
    fake.scripts = [[content("fine")]]
    await run(agent(cfg, fake, store))
    body = fake.last_body
    assert names(body) == [*KNOWLEDGE_TOOLS, "calculate", "compute"]
    assert body["messages"][0]["content"] == f"{p.system_prompt.rstrip()}\n{p.compute_prompt}"
    assert p.tool_prompt.strip() not in body["messages"][0]["content"]


async def test_tool_prompt_alone(cfg, fake, store):
    p = cfg.reasoning_policy
    p.use_system_prompt = p.compute_tool = p.enable_multi_hop_verification = False
    fake.scripts = [[content("fine")]]
    await run(agent(cfg, fake, store))
    assert fake.last_body["messages"][0] == {"role": "system", "content": p.tool_prompt}


async def test_both_prompts_on_is_the_system_message_as_before(cfg, fake, store):
    p = cfg.reasoning_policy
    p.enable_multi_hop_verification = False
    fake.scripts = [[content("fine")]]
    await run(agent(cfg, fake, store))
    assert fake.last_body["messages"][0]["content"] == \
        f"{p.system_prompt.rstrip()}\n\n{p.tool_prompt}".rstrip() + f"\n{p.compute_prompt}"


# --- tools by name -----------------------------------------------------------------------------------


def test_one_tool_switched_off(cfg, store):
    cfg.reasoning_policy.tools_off = ["lookup_compound"]
    k = Knowledge(store, cfg)
    assert [t["function"]["name"] for t in k.tools] == ["search_knowledge", "lookup_constant", "calculate", "compute"]
    text, lines, ids = k.run("lookup_compound", '{"query": "caffeine"}', Sources())
    assert text.startswith("Error: unknown tool 'lookup_compound'; available: search_knowledge, lookup_constant")
    assert ids == [] and k.available


def test_calculator_alone_and_nothing_at_all(cfg, store):
    cfg.reasoning_policy.tools_off, cfg.reasoning_policy.compute_tool = list(KNOWLEDGE_TOOLS), False
    k = Knowledge(store, cfg)
    assert [t["function"]["name"] for t in k.tools] == ["calculate"] and k.calculator and k.available
    assert k.run("calculate", '{"expressions": ["2 * 3"]}', Sources())[0] == "2 * 3 = 6"
    cfg.reasoning_policy.tools_off = [*KNOWLEDGE_TOOLS, "calculate", "compute"]
    k = Knowledge(store, cfg)
    assert k.tools == [] and not k.calculator and not k.computer and k.available  # the store still offers knowledge
    assert k.run("calculate", '{"expressions": ["2 * 3"]}', Sources())[0].startswith("Error: unknown tool")
    assert not Knowledge(None, cfg).available  # no store: nothing on offer


def test_compute_without_the_calculator(cfg, store):
    cfg.reasoning_policy.tools_off = ["calculate"]
    k = Knowledge(store, cfg)
    assert [t["function"]["name"] for t in k.tools] == [*KNOWLEDGE_TOOLS, "compute"] and k.computer and not k.calculator


def test_a_tool_that_is_off_needs_no_text(cfg, store):
    del cfg.tools["lookup_constant"]
    cfg.reasoning_policy.tools_off = ["lookup_constant"]  # set after loading: the loader would ask for its text
    assert "lookup_constant" not in [t["function"]["name"] for t in Knowledge(store, cfg).tools]


def test_the_verifier_may_search_a_store_whose_search_tool_is_off(cfg, store):
    cfg.reasoning_policy.tools_off = ["search_knowledge"]
    k, sources = Knowledge(store, cfg), Sources()
    assert k.run("search_knowledge", '{"query": "ethanol"}', sources)[0].startswith("Error: unknown tool")
    text, _, ids = k.run("search_knowledge", '{"query": "ethanol"}', sources, internal=True)
    assert ids and ids[0] == "S1" and text.startswith("[S1] ")


# --- thinking budget ---------------------------------------------------------------------------------


async def test_thinking_budget_off_sends_no_field_and_the_room_of_a_full_budget(cfg, fake, store):
    tb = cfg.reasoning_policy.thinking_budget
    tb.enabled, cfg.reasoning_policy.enable_multi_hop_verification = False, False
    assert unscaled_room(tb, cfg.model_backend.max_tokens) == tb.max_tokens + tb.answer_tokens == 7168
    fake.scripts = [LOOKUP, [content("194.19 g/mol [S1].")]]
    _, trace, _ = await run(agent(cfg, fake, store))
    assert len(fake.bodies) == 2 and "thinking budget" not in trace
    for body in fake.bodies:
        assert "thinking_budget_tokens" not in body and body["max_tokens"] == 7168


async def test_thinking_budget_off_leaves_the_retry_as_it_was(cfg, fake):
    cfg.reasoning_policy.thinking_budget.enabled = False
    fake.scripts = [[reasoning("thinking only")], [content("the answer")]]
    answer, *_ = await run(agent(cfg, fake))
    retry = fake.bodies[1]
    assert answer == "the answer" and retry["max_tokens"] == cfg.model_backend.max_tokens
    assert "thinking_budget_tokens" not in retry and retry["chat_template_kwargs"] == {"enable_thinking": False}


async def test_without_a_field_the_switch_changes_nothing(cfg, fake):
    tb = cfg.reasoning_policy.thinking_budget
    tb.field, tb.enabled = "", False
    assert unscaled_room(tb, 4096) == 4096  # the old setup: the model's max_tokens
    fake.scripts = [[content("fine")]]
    await run(agent(cfg, fake))
    assert fake.last_body["max_tokens"] == cfg.model_backend.max_tokens and "thinking_budget_tokens" not in fake.last_body


async def test_independent_item_calls_follow_the_switch(cfg, fake):
    tb, ev = cfg.reasoning_policy.thinking_budget, cfg.evidence
    ev.enabled, ev.independent, tb.enabled = True, True, False
    evidence = Evidence(cfg, fake.client(cfg.model_backend), checks=False)
    assert evidence.item_request == {"max_tokens": tb.max_tokens + ev.item_answer_tokens}
    tb.enabled = True
    assert Evidence(cfg, fake.client(cfg.model_backend), checks=False).item_request == {
        "thinking_budget_tokens": tb.max_tokens, "max_tokens": tb.max_tokens + ev.item_answer_tokens}


# --- [S#] markers ------------------------------------------------------------------------------------


@pytest.mark.parametrize("strip, expected", [(False, "total = table[S1] + table[S2]"), (True, "total = table + table")])
async def test_marker_like_text_is_kept_when_the_switch_is_off(cfg, fake, strip, expected):
    cfg.verifier.strip_source_ids = strip
    fake.scripts = [[content("total = table[S1] + table[S2]")]]
    answer, *_ = await run(agent(cfg, fake))
    assert answer == expected


@pytest.mark.parametrize("strip, expected", [(False, "Caffeine is a stimulant found in coffee [S1]. It is bitter."),
                                             (True, "Caffeine is a stimulant found in coffee. It is bitter.")])
async def test_the_switch_holds_after_the_claim_check(cfg, fake, store, strip, expected):
    cfg.verifier.strip_source_ids, cfg.verifier.claim_judge = strip, False
    fake.scripts = [LOOKUP, [content("Caffeine is a stimulant found in coffee [S1]. It is bitter.")]]
    answer, trace, report = await run(agent(cfg, fake, store))
    assert answer == expected and "Verification: 2 claims" in trace and len(report["claims"]) == 2


# --- verifier: the store, the record check, the claim judge -------------------------------------------


async def test_code_checks_run_with_every_tool_switched_off(cfg, fake, store):
    p, v = cfg.reasoning_policy, cfg.verifier
    p.tools_off, v.record_check, v.claim_judge = [*KNOWLEDGE_TOOLS, "calculate", "compute"], False, False
    fake.scripts = [[content("So 2 × 3 = 7 apples.")]]
    answer, trace, report = await run(agent(cfg, fake, store))
    assert answer == "So 2 × 3 = 6 apples." and len(fake.requests) == 1
    assert "tools" not in fake.last_body and "tool_choice" not in fake.last_body
    assert report["claims"] == [] and [a["fixed"] for a in report["arithmetic"]] == ["6"]
    assert "Verification:" not in trace  # no claim stage


async def test_without_a_store_there_is_still_no_verification(cfg, fake):
    fake.scripts = [[content("So 2 × 3 = 7 apples.")]]
    answer, _, report = await run(agent(cfg, fake))
    assert answer == "So 2 × 3 = 7 apples." and report == {}


async def test_judge_off_makes_no_judge_call_and_no_re_search(cfg, fake, store):
    cfg.verifier.claim_judge = False
    fake.scripts = [LOOKUP, [content("Caffeine is a stimulant found in coffee [S1]. It is bitter.")]]
    answer, trace, report = await run(agent(cfg, fake, store))
    assert len(fake.requests) == 2 and "re-search" not in trace and "judge:" not in trace
    assert answer == "Caffeine is a stimulant found in coffee. It is bitter."
    assert [(c["outcome"], c["note"]) for c in report["claims"]] == [(UNVERIFIED, "not judged: the claim judge is off")] * 2
    assert "2 claims not judged (the claim judge is off)" in trace


async def test_record_check_off_sends_the_claim_to_the_judge(cfg, fake, store):
    wrong = "The molar mass of caffeine is 190.00 g/mol [S1]."
    cfg.verifier.max_research = 0
    fake.scripts = [LOOKUP, [content(wrong)]]
    answer, trace, _ = await run(agent(cfg, fake, store))
    assert "194.19" in answer and len(fake.requests) == 2 and "code:" in trace  # corrected from the record, in code
    cfg.verifier.record_check = False
    fake.requests.clear()
    fake.scripts = [LOOKUP, [content(wrong)], [content("C1: not_said")]]
    answer, trace, _ = await run(agent(cfg, fake, store))
    assert answer == wrong.replace(" [S1]", "") and len(fake.requests) == 3 and "code:" not in trace
    assert fake.last_body["messages"][0]["content"] == cfg.verifier.judge_prompt


async def test_claim_stage_off_hands_the_draft_back_unchanged(cfg, fake, store):
    draft = "- \nWater boils at 100 °C [S1].  "
    verifier = Verifier(cfg, fake.client(cfg.model_backend), Knowledge(store, cfg))

    async def verify():
        items = [item async for item in verifier.verify("q", draft, Sources())]
        return next(i for i in items if isinstance(i, Verification)), items

    cfg.verifier.max_research = 0
    on, _ = await verify()
    assert on.answer == "Water boils at 100 °C." and len(on.claims) == 1  # the claim stage renders the draft
    cfg.verifier.record_check = cfg.verifier.claim_judge = False
    off, items = await verify()
    assert off.answer == draft and off.claims == [] and off.report()["counts"] == dict.fromkeys(
        ("supported", "unverified", "corrected", "removed"), 0)
    assert items == [off] and not fake.requests  # nothing said, nothing called


# --- evidence: the writer ----------------------------------------------------------------------------


async def settle(cfg, fake, replies):
    cfg.evidence.enabled = True
    fake.scripts = [[content(r)] for r in replies]
    ev = Evidence(cfg, fake.client(cfg.model_backend), checks=False)
    items = [("C1", "The batch weighed 5 kg."), ("C2", "One batch guarantees an 85 % yield at any scale."),
             ("C3", "2 × 3 = 7.")]
    out = [i async for i in ev.run([{"role": "user", "content": TASK}], items, DRAFT, "")]
    return next(i for i in out if isinstance(i, Settled)), "".join(i["reasoning_content"] for i in out if isinstance(i, dict))


async def test_writer_off_keeps_the_draft_and_reports_the_change(cfg, fake):
    cfg.evidence.writer = False
    settled, trace = await settle(cfg, fake, [V2])
    assert settled.answer == DRAFT and len(fake.requests) == 1  # the ledger call only
    assert "Evidence: C2 contradicted (claim of proof the evidence can't give; model said unverified)" in trace
    assert "writer off: the answer stays as drafted" in trace
    report = settled.report
    assert (report["written"], report["fallback"]) == (False, "writer off")
    assert report["items"][1] == {"id": "C2", "asserts": "proof", "claimed": "unverified", "status": "contradicted",
                                  "rule": "R-b"}


@pytest.mark.parametrize("writer", [True, False])
async def test_nothing_changed_is_the_same_with_the_writer_on_or_off(cfg, fake, writer):
    cfg.evidence.writer = writer
    settled, _ = await settle(cfg, fake, [SETTLED])
    assert settled.answer == DRAFT and len(fake.requests) == 1
    assert (settled.report["written"], settled.report["fallback"]) == (False, "")


async def test_writer_on_rewrites_as_before(cfg, fake):
    settled, _ = await settle(cfg, fake, [V2, "C1 is true. C2 is false. C3 is false: 2 × 3 = 6."])
    assert settled.answer == "C1 is true. C2 is false. C3 is false: 2 × 3 = 6." and settled.report["written"] is True
    assert len(fake.requests) == 2 and fake.bodies[1]["messages"][0]["content"] == cfg.evidence.writer_prompt
