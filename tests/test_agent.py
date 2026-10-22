"""The tool loop through the chat endpoint, with the fake upstream and the fixture store (agent-loop spec)."""

import logging

import pytest
from fastapi.testclient import TestClient

from conftest import CONNECT_ERROR, content, finish, reasoning, tool_call
from haisenberg.agent import DRAFT_MARK, Agent
from haisenberg.l1.store import L1Store
from haisenberg.server import create_app
from haisenberg.tools import Knowledge
from test_server import chat, events

CAFFEINE = '{"query": "caffeine"}'


@pytest.fixture
def knowledge(cfg, built_store):
    return Knowledge(L1Store(built_store), cfg)


@pytest.fixture
def client(cfg, fake, knowledge):
    with TestClient(create_app(cfg, fake.client(cfg.model_backend), knowledge)) as c:
        yield c


def deltas(r):
    return [e["choices"][0]["delta"] for e in events(r.text) if e != "[DONE]"]


def reasoning_text(r):
    return "".join(d.get("reasoning_content", "") for d in deltas(r))


def answer(r):
    return "".join(d.get("content", "") for d in deltas(r))


def lookup_round(arguments=CAFFEINE, name="lookup_compound", call_id="c1"):
    return [reasoning("need data"), tool_call(name, arguments, call_id=call_id), finish("tool_calls")]


def test_one_lookup_then_the_answer(client, fake, cfg):
    fake.scripts = [lookup_round(), [content("Caffeine is 194.19 g/mol [S1]."), finish("stop")]]
    r = chat(client, [{"role": "user", "content": "molar mass of caffeine?"}])
    assert len(fake.requests) == 2
    first, second = fake.bodies
    assert first["tools"] == second["tools"] and "tool_choice" not in first and "tool_choice" not in second
    call, result = second["messages"][-2:]
    assert call == {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "lookup_compound", "arguments": CAFFEINE}}]}
    assert result["role"] == "tool" and result["tool_call_id"] == "c1"
    assert result["content"].startswith("[S1] Caffeine (CID 2519) (compound; chemistry)\nCID 2519")
    text = answer(r)
    assert text.startswith("Caffeine is 194.19 g/mol [S1].\n\nSources:\n- [S1] Caffeine (CID 2519): ")
    assert text.endswith("https://pubchem.ncbi.nlm.nih.gov/compound/2519")
    assert events(r.text)[-2]["choices"][0]["finish_reason"] == "stop"


def test_depth_exhausted(client, fake, cfg):
    cfg.reasoning_policy.max_search_depth = 2
    fake.scripts = [lookup_round(call_id="c1"), lookup_round(call_id="c2"), lookup_round(call_id="c3")]
    fake.chunks = [content("never asked")]
    chat(client)
    assert len(fake.requests) == 3
    assert [b.get("tool_choice") for b in fake.bodies] == [None, None, "none"]
    assert all(b["tools"] for b in fake.bodies)


def test_fragmented_tool_call(client, fake):
    fake.scripts = [
        [tool_call("lookup_compound", '{"que', call_id="x1"), tool_call(arguments='ry": "caf'),
         tool_call(arguments='feine"}'), finish("tool_calls")],
        [content("ok [S1]")],
    ]
    chat(client)
    call, result = fake.bodies[1]["messages"][-2:]
    assert call["tool_calls"][0]["function"] == {"name": "lookup_compound", "arguments": CAFFEINE}
    assert "Caffeine" in result["content"]


def test_parallel_calls_in_one_round(client, fake):
    fake.scripts = [
        [tool_call("lookup_compound", CAFFEINE, index=0, call_id="a"),
         tool_call("lookup_constant", '{"query": "Planck constant"}', index=1, call_id="b"), finish("tool_calls")],
        [content("done")],
    ]
    chat(client)
    msgs = fake.bodies[1]["messages"]
    assert [m["role"] for m in msgs[-3:]] == ["assistant", "tool", "tool"]
    assert [m["tool_call_id"] for m in msgs[-2:]] == ["a", "b"]
    assert msgs[-1]["content"].startswith("[S2] Planck constant")


def test_tool_prompt_added_when_tools_offered(client, fake, cfg):
    chat(client)
    policy = cfg.reasoning_policy
    assert fake.last_body["messages"][0] == {
        "role": "system", "content": policy.system_prompt.rstrip() + "\n\n" + policy.tool_prompt,
    }


def test_tool_call_visible_before_the_answer(client, fake):
    fake.scripts = [lookup_round('{"query": "ethanol"}', name="search_knowledge"), [content("Ethanol [S1].")]]
    r = chat(client)
    text = reasoning_text(r)
    assert 'search_knowledge(query="ethanol")' in text and "- [S1] Ethanol" in text
    assert text.index("need data") < text.index("search_knowledge") < text.index(DRAFT_MARK)
    kinds = [next(iter(d)) for d in deltas(r) if d]
    assert kinds.index("content") > max(i for i, k in enumerate(kinds) if k == "reasoning_content")


def test_invented_id_not_listed(client, fake):
    fake.scripts = [lookup_round(), [content("Caffeine [S1], but also [S7].")]]
    text = answer(chat(client))
    assert "- [S1] Caffeine" in text and "- [S7]" not in text


def test_uncited_draft_has_no_source_list(client, fake):
    fake.scripts = [lookup_round(), [content("No sources used.")]]
    assert answer(chat(client)) == "No sources used."


def test_depth_zero(cfg, fake, built_store):
    cfg.reasoning_policy.max_search_depth = 0
    with TestClient(create_app(cfg, fake.client(cfg.model_backend), Knowledge(L1Store(built_store), cfg))) as c:
        chat(c)
    assert len(fake.requests) == 1
    assert "tools" not in fake.last_body
    assert fake.last_body["messages"][0]["content"] == cfg.reasoning_policy.system_prompt


def test_backend_dies_in_round_two(client, fake, caplog):
    fake.scripts = [lookup_round(), CONNECT_ERROR]
    with caplog.at_level(logging.ERROR):
        r = chat(client)
    assert r.status_code == 200 and "[DONE]" not in r.text
    assert "unreachable" in caplog.text


def test_client_tools_ignored(client, fake, knowledge):
    chat(client, tools=[{"type": "function", "function": {"name": "evil"}}], tool_choice="required")
    assert fake.last_body["tools"] == knowledge.tools and "tool_choice" not in fake.last_body


@pytest.mark.anyio
async def test_stop_closes_upstream_and_makes_no_more_calls(cfg, fake, knowledge):
    fake.scripts = [lookup_round()]
    agent = Agent(cfg, fake.client(cfg.model_backend), knowledge)
    run = agent.run([{"role": "user", "content": "hi"}])
    assert await anext(run) == {"reasoning_content": "need data"}
    await run.aclose()
    assert fake.streams[0].closed and len(fake.requests) == 1
