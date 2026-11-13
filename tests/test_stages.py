"""The stage log (measure-suite): one record per model call, tool call and code check of an answer, only while an
in-process caller has a log open."""

import pytest

from conftest import CONNECT_ERROR, content, finish, reasoning, tool_call
from haisenberg import stages
from haisenberg.agent import Agent
from haisenberg.evidence import Evidence
from haisenberg.l1.store import L1Store
from haisenberg.llm import UpstreamError
from haisenberg.tools import Knowledge, Sources
from test_evidence import DRAFT, TASK, V2

pytestmark = pytest.mark.anyio

USER = [{"role": "user", "content": "hi"}]
USAGE = {"choices": [], "usage": {"prompt_tokens": 812, "completion_tokens": 96, "total_tokens": 908}}
LOOKUP = [tool_call("lookup_compound", '{"query": "caffeine"}', call_id="c1"), finish("tool_calls")]


def test_a_stage_is_recorded_only_while_a_log_is_open():
    with stages.stage("check", stages.CHECK) as record:
        assert record is None  # nobody listens: nothing is kept
    with stages.recording() as log:
        with stages.stage("outer", stages.CHECK, note="x") as outer:
            with stages.stage("inner", stages.TOOL):
                pass
        with pytest.raises(ValueError):
            with stages.stage("fails", stages.MODEL):
                raise ValueError("boom")
    assert [(r["stage"], r["kind"]) for r in log] == [("outer", "check"), ("inner", "tool"), ("fails", "model")]
    assert outer is log[0] and log[0]["note"] == "x"
    assert all(r["seconds"] >= 0 and r["start_s"] >= 0 for r in log)  # also for the stage that raised
    assert log[0]["start_s"] <= log[1]["start_s"] <= log[2]["start_s"]
    with stages.stage("after", stages.CHECK) as record:
        assert record is None and len(log) == 3


async def test_a_model_call_records_usage_characters_and_its_own_fields(cfg, fake):
    fake.scripts = [[reasoning("hm"), content("Hello"), content("!"), finish("stop"), USAGE]]
    client = fake.client(cfg.model_backend)
    with stages.recording() as log:
        chunks = [c async for c in client.stream_chat(USER, stage="judge", temperature=0, tools=[{"type": "function"}])]
    assert chunks[-1] == USAGE  # the usage chunk is passed on like any other
    assert len(log) == 1
    record = log[0]
    assert (record["stage"], record["kind"]) == ("judge", "model")
    assert (record["prompt_tokens"], record["completion_tokens"]) == (812, 96)
    assert (record["reasoning_chars"], record["answer_chars"], record["finish_reason"]) == (2, 6, "stop")
    assert record["tools"] == 1 and record["fields"] == {"temperature": 0}
    assert 0 <= record["first_s"] <= record["seconds"]
    assert "stage" not in fake.last_body and fake.last_body["temperature"] == 0  # the name never leaves


async def test_a_backend_without_usage_gives_no_token_counts(cfg, fake):
    fake.scripts = [[content("Hello")]]
    with stages.recording() as log:
        [c async for c in fake.client(cfg.model_backend).stream_chat(USER)]
    assert (log[0]["stage"], log[0]["prompt_tokens"], log[0]["completion_tokens"]) == ("model", None, None)
    assert (log[0]["answer_chars"], log[0]["finish_reason"], log[0]["tools"], log[0]["fields"]) == (5, None, 0, {})


async def test_a_failed_call_is_recorded_and_still_raises(cfg, fake):
    fake.scripts = [CONNECT_ERROR]
    with stages.recording() as log:
        with pytest.raises(UpstreamError):
            [c async for c in fake.client(cfg.model_backend).stream_chat(USER, stage="review")]
    assert len(log) == 1 and log[0]["stage"] == "review" and log[0]["seconds"] is not None
    assert log[0]["first_s"] is None and log[0]["finish_reason"] is None


async def test_nothing_changes_without_a_log(cfg, fake):
    fake.scripts = [[content("Hello"), USAGE], [content("Hello"), USAGE]]
    client = fake.client(cfg.model_backend)
    plain = [c async for c in client.stream_chat(USER, stage="judge", temperature=0)]
    with stages.recording():
        logged = [c async for c in client.stream_chat(USER, stage="judge", temperature=0)]
    assert plain == logged and fake.bodies[0] == fake.bodies[1]


def test_tool_calls_are_recorded_with_their_name_arguments_and_results(cfg, built_store):
    k, sources = Knowledge(L1Store(built_store), cfg), Sources()
    with stages.recording() as log:
        k.run("lookup_compound", '{"query": "caffeine"}', sources)
        k.run("search_knowledge", '{"query": "ethanol"}', sources, 2, internal=True)
        k.run("calculate", '{"expressions": ["2 * 3"]}', sources)
        k.run("lookup_nothing", "{}", sources)
    assert [(r["stage"], r["kind"], r["tool"]) for r in log] == [
        ("tool", "tool", "lookup_compound"), ("research", "tool", "search_knowledge"), ("tool", "tool", "calculate"),
        ("tool", "tool", "lookup_nothing")]
    assert log[0]["arguments"] == '{"query": "caffeine"}' and log[0]["results"] >= 1
    assert 1 <= log[1]["results"] <= 2 and log[2]["results"] == 0 and log[3]["results"] == 0
    long = '{"query": "' + "x" * 500 + '"}'
    with stages.recording() as log:
        k.run("search_knowledge", long, sources)
    assert len(log[0]["arguments"]) == 300


async def test_one_lookup_the_judge_and_the_review_in_order(cfg, fake, built_store):
    cfg.verifier.checkers, cfg.verifier.consistency_check, cfg.verifier.max_research = [], True, 0
    fake.scripts = [LOOKUP, [content("Caffeine is a stimulant found in coffee [S1].")],
                    [content("C1: supports S1")], [content("OK")]]
    agent = Agent(cfg, fake.client(cfg.model_backend), Knowledge(L1Store(built_store), cfg))
    with stages.recording() as log:
        [item async for item in agent.run([{"role": "user", "content": "What is caffeine?"}])]
    assert [r["stage"] for r in log] == ["reasoner", "tool", "reasoner", "arithmetic", "claims", "judge", "review"]
    assert [r["kind"] for r in log] == ["model", "tool", "model", "check", "check", "model", "model"]
    assert log[1]["tool"] == "lookup_compound" and log[1]["results"] == 1
    tb = cfg.reasoning_policy.thinking_budget
    assert log[0]["fields"] == {"thinking_budget_tokens": tb.min_tokens, "max_tokens": tb.min_tokens + tb.answer_tokens}
    assert log[0]["tools"] == 5 and log[5]["fields"] == cfg.verifier.request and log[5]["tools"] == 0
    assert all(a["start_s"] <= b["start_s"] for a, b in zip(log, log[1:])) and all(r["seconds"] is not None for r in log)


async def test_carry_forward_checkers_research_and_retry_have_their_stages(cfg, fake, built_store):
    cfg.verifier.carry_fixes_forward, cfg.verifier.max_research = True, 1
    cfg.verifier.checkers = ["element_balance"]
    draft = "The dose is 2 × 3 = 7 mg. Later we use 7 mg again. Ethanol is a colorless liquid."
    fake.scripts = [[reasoning("thinking only")], [content(draft)], [content("C1: not_said")]]
    agent = Agent(cfg, fake.client(cfg.model_backend), Knowledge(L1Store(built_store), cfg))
    with stages.recording() as log:
        [item async for item in agent.run([{"role": "user", "content": "How much?"}])]
    names = [r["stage"] for r in log]
    assert names[:5] == ["reasoner", "answer_retry", "arithmetic", "carry_forward", "checkers"]
    assert names[5] == "claims" and "research" in names and names[-1] == "research_judge"
    research = log[names.index("research")]
    assert (research["kind"], research["tool"]) == ("tool", "search_knowledge")


async def test_the_ledger_and_the_writer_are_named(cfg, fake):
    cfg.evidence.enabled = True
    fake.scripts = [[content(V2)], [content("C1 is true. C2 is false. C3 is false: 2 × 3 = 6.")]]
    ev = Evidence(cfg, fake.client(cfg.model_backend), checks=False)
    items = [("C1", "The batch weighed 5 kg."), ("C2", "One batch guarantees an 85 % yield at any scale."),
             ("C3", "2 × 3 = 7.")]
    with stages.recording() as log:
        [i async for i in ev.run([{"role": "user", "content": TASK}], items, DRAFT, "")]
    assert [(r["stage"], r["kind"]) for r in log] == [("ledger", "model"), ("writer", "model")]
    assert log[0]["fields"] == cfg.evidence.ledger_request
