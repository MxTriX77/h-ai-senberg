"""The tool loop through the chat endpoint, with the fake upstream and the fixture store (agent-loop spec)."""

import logging

import pytest
from fastapi.testclient import TestClient

from conftest import CONNECT_ERROR, content, finish, reasoning, tool_call
from haisenberg.agent import CUT_NOTE, DRAFT_MARK, Agent
from haisenberg.llm import estimate_tokens
from haisenberg.l1.store import L1Store
from haisenberg.server import create_app
from haisenberg.tools import Knowledge
from test_server import chat, events

CAFFEINE = '{"query": "caffeine"}'


@pytest.fixture
def knowledge(cfg, built_store):
    cfg.reasoning_policy.enable_multi_hop_verification = False  # the loop alone; tests/test_verifier.py adds it
    return Knowledge(L1Store(built_store), cfg)


@pytest.fixture
def client(cfg, fake, knowledge):
    with TestClient(create_app(cfg, fake.client(cfg.model_backend), knowledge)) as c:
        yield c


@pytest.fixture
def annotated(cfg, fake, knowledge):
    """A client whose answers keep the [S#] ids and the source list (verifier.answer_style: annotated)."""
    cfg.verifier.answer_style = "annotated"
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
    assert answer(r) == "Caffeine is 194.19 g/mol."  # plain: no ids, no source list
    assert "[S1] Caffeine" in reasoning_text(r)  # the lookup stays in the reasoning block
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


def test_cited_sources_listed_when_annotated(annotated, fake):
    fake.scripts = [lookup_round(), [content("Caffeine is 194.19 g/mol [S1]."), finish("stop")]]
    text = answer(chat(annotated))
    assert text.startswith("Caffeine is 194.19 g/mol [S1].\n\nSources:\n- [S1] Caffeine (CID 2519): ")
    assert text.endswith("https://pubchem.ncbi.nlm.nih.gov/compound/2519")


def test_invented_id_not_listed(annotated, fake):
    fake.scripts = [lookup_round(), [content("Caffeine [S1], but also [S7].")]]
    text = answer(chat(annotated))
    assert "- [S1] Caffeine" in text and "- [S7]" not in text


def test_plain_answer_drops_every_id(client, fake):
    fake.scripts = [lookup_round(), [content("Caffeine [S1], but also [S7] and [S1, S2].\n[S1]\nDone.")]]
    assert answer(chat(client)) == "Caffeine, but also and.\n\nDone."


def test_uncited_draft_has_no_source_list(annotated, fake):
    fake.scripts = [lookup_round(), [content("No sources used.")]]
    assert answer(chat(annotated)) == "No sources used."


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
    assert r.status_code == 200 and r.text.endswith("data: [DONE]\n\n")
    assert "could not be completed" in answer(r) and "unreachable" in answer(r)
    assert "unreachable" in caplog.text


# --- context budget ----------------------------------------------------------------------------------


def within_context(cfg, body) -> bool:
    prompt = estimate_tokens(body["messages"], cfg.model_backend.chars_per_token, body.get("tools"))
    return prompt + body.get("max_tokens", cfg.model_backend.max_tokens) <= cfg.context_tokens


def base_tokens(cfg, knowledge) -> int:
    system = f"{cfg.reasoning_policy.system_prompt.rstrip()}\n\n{cfg.reasoning_policy.tool_prompt}"
    return estimate_tokens([{"role": "system", "content": system}], cfg.model_backend.chars_per_token, knowledge.tools)


def test_long_history_trimmed(cfg, fake, knowledge):
    cfg.model_backend.max_tokens = 1000
    cfg.llama_server.ctx_size = 1000 + 2 * base_tokens(cfg, knowledge) + 3000
    turns = [{"role": r, "content": f"{r} turn {i} " + "x" * 600} for i in range(10) for r in ("user", "assistant")]
    fake.scripts = [[content("ok")]]
    with TestClient(create_app(cfg, fake.client(cfg.model_backend), knowledge)) as c:
        chat(c, [*turns, {"role": "user", "content": "the question"}])
    sent = fake.bodies[0]["messages"]
    assert 2 < len(sent) < len(turns) + 2 and sent[1]["role"] == "user"
    assert sent[-1] == {"role": "user", "content": "the question"} and sent[-2]["content"].startswith("assistant turn 9")
    assert within_context(cfg, fake.bodies[0])


def test_worst_case_tool_rounds_stay_within_context(cfg, fake, knowledge, monkeypatch):
    cfg.model_backend.max_tokens = 1000
    cfg.llama_server.ctx_size = 1000 + 2 * base_tokens(cfg, knowledge) + 100
    big = "x" * 4 * base_tokens(cfg, knowledge)  # one result is larger than all the room left
    monkeypatch.setattr(knowledge, "run", lambda name, args, sources, k=None: (big, ["[S1] X"], ["S1"]))
    two_calls = [tool_call("lookup_compound", CAFFEINE, index=0, call_id="c1"),
                 tool_call("lookup_compound", CAFFEINE, index=1, call_id="c2"), finish("tool_calls")]
    fake.scripts = [two_calls, [content("answer")]]  # (a real backend honours tool_choice: "none")
    with TestClient(create_app(cfg, fake.client(cfg.model_backend), knowledge)) as c:
        r = chat(c)
    assert answer(r) == "answer" and len(fake.requests) == 2  # the first round filled the context
    assert all(within_context(cfg, b) for b in fake.bodies)
    tool_results = [m["content"] for m in fake.bodies[1]["messages"] if m["role"] == "tool"]
    assert all(t.endswith(CUT_NOTE) for t in tool_results) and len(tool_results) == 2
    assert fake.bodies[1]["tool_choice"] == "none" and "shortened to fit" in reasoning_text(r)


def test_question_too_long(cfg, fake, knowledge):
    cfg.model_backend.max_tokens = 1000
    cfg.llama_server.ctx_size = 1000 + base_tokens(cfg, knowledge) + 100
    with TestClient(create_app(cfg, fake.client(cfg.model_backend), knowledge)) as c:
        r = chat(c, [{"role": "user", "content": "x" * 5000}])
    assert "too long for the model's context" in answer(r) and not fake.requests
    assert r.text.endswith("data: [DONE]\n\n")


def test_client_tools_ignored(client, fake, knowledge):
    chat(client, tools=[{"type": "function", "function": {"name": "evil"}}], tool_choice="required")
    assert fake.last_body["tools"] == knowledge.tools and "tool_choice" not in fake.last_body


@pytest.mark.anyio
async def test_stop_closes_upstream_and_makes_no_more_calls(cfg, fake, knowledge):
    fake.scripts = [lookup_round()]
    agent = Agent(cfg, fake.client(cfg.model_backend), knowledge)
    run = agent.run([{"role": "user", "content": "hi"}])
    assert await anext(run) == {"route": "factual"}
    assert await anext(run) == {"reasoning_content": "need data"}
    await run.aclose()
    assert fake.streams[0].closed and len(fake.requests) == 1
