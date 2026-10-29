"""The tool loop through the chat endpoint, with the fake upstream and the fixture store (agent-loop spec)."""

import json
import logging

import pytest
from fastapi.testclient import TestClient

from conftest import CONNECT_ERROR, content, finish, reasoning, tool_call
from haisenberg.agent import CUT_NOTE, DRAFT_MARK, NO_ANSWER, Agent, _Markup
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
    cfg.reasoning_policy.max_calc_rounds = 0  # no calculator rounds on top
    cfg.reasoning_policy.empty_answer_retry = False  # the third reply has no content (tests below)
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
    assert await anext(run) == {"reasoning_content": "need data"}
    await run.aclose()
    assert fake.streams[0].closed and len(fake.requests) == 1


@pytest.mark.anyio
async def test_draft_for_in_process_callers_only(cfg, fake, knowledge):
    fake.scripts = [[content("Caffeine [S1].")], [content("Caffeine [S1].")], [content("Caffeine [S1].")]]
    items = [i async for i in Agent(cfg, fake.client(cfg.model_backend), knowledge).run([{"role": "user", "content": "hi"}])]
    assert {"draft": "Caffeine [S1]."} in items and next(i for i in items if "content" in i) == {"content": "Caffeine."}
    assert items.index({"draft": "Caffeine [S1]."}) < next(n for n, i in enumerate(items) if "content" in i)
    with TestClient(create_app(cfg, fake.client(cfg.model_backend), knowledge)) as c:
        assert all("draft" not in d for d in deltas(chat(c)))
        r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
        assert set(r.json()["choices"][0]["message"]) == {"role", "content", "reasoning_content"}
        fake.status = 400  # an upstream 4xx still reaches the non-streaming client before any item
        r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 400


# --- answer robustness: calculator rounds, tool calls written as text, empty answers -----------------

XML_CALL = "<tool_call>\n<function=calculate>\n<parameter=expressions>\n0.48 * 1000\n</parameter>\n</function>\n</tool_call>"
LOOKUP_CALL = "<tool_call>\n<function=lookup_compound>\n<parameter=query>\ncaffeine\n</parameter>\n</function>\n</tool_call>"


def calc_round(expression="2 * 3", call_id="k1"):
    return [tool_call("calculate", json.dumps({"expressions": [expression]}), call_id=call_id), finish("tool_calls")]


def offered(body):
    return [t["function"]["name"] for t in body["tools"]]


def test_calculator_rounds_use_no_lookup_rounds(client, fake, cfg):
    cfg.reasoning_policy.max_search_depth, cfg.reasoning_policy.max_calc_rounds = 2, 1
    fake.scripts = [calc_round(), lookup_round(call_id="c1"), lookup_round(call_id="c2"), [content("done")]]
    assert answer(chat(client)) == "done"
    assert [b.get("tool_choice") for b in fake.bodies] == [None, None, None, "none"]
    assert sum(m["role"] == "tool" for m in fake.bodies[-1]["messages"]) == 3  # both lookups ran after the calculation


def test_lookups_used_up_leave_the_calculator(client, fake, cfg):
    cfg.reasoning_policy.max_search_depth, cfg.reasoning_policy.max_calc_rounds = 1, 2
    fake.scripts = [lookup_round(), calc_round(call_id="k1"), calc_round(call_id="k2"), [content("done")]]
    assert answer(chat(client)) == "done"
    assert "lookup_compound" in offered(fake.bodies[0])
    assert [offered(b) for b in fake.bodies[1:]] == [["calculate"]] * 3  # the same tools up to the last call
    assert [b.get("tool_choice") for b in fake.bodies] == [None, None, None, "none"]


def test_tool_call_written_as_text_after_the_last_round(client, fake, cfg):
    policy = cfg.reasoning_policy
    policy.max_search_depth, policy.max_calc_rounds = 1, 0
    fake.scripts = [lookup_round(), [content(LOOKUP_CALL)], [content("The mass is 480 g.")]]
    r = chat(client)
    assert answer(r) == "The mass is 480 g." and len(fake.requests) == 3
    retry = fake.bodies[2]
    assert retry["messages"] == [*fake.bodies[1]["messages"], {"role": "user", "content": policy.answer_retry_prompt}]
    assert retry["tool_choice"] == "none" and retry["chat_template_kwargs"] == {"enable_thinking": False}
    assert "tool call written as text: removed, final answer asked again" in reasoning_text(r)
    assert "answer kept" not in reasoning_text(r)  # the retry is complete: used as it is


CALC_MSG = {"role": "assistant", "content": None, "tool_calls": [{
    "id": "call_0", "type": "function", "function": {"name": "calculate", "arguments": '{"expressions": "0.48 * 1000"}'}}]}
CALC_RESULT = {"role": "tool", "tool_call_id": "call_0", "content": "0.48 * 1000 = 480"}


def test_calculation_written_after_the_last_round_is_run(client, fake, cfg):
    policy = cfg.reasoning_policy
    policy.max_search_depth, policy.max_calc_rounds = 1, 0
    fake.scripts = [lookup_round(), [content(XML_CALL)], [content("The mass is 480 g.")]]
    r = chat(client)
    assert answer(r) == "The mass is 480 g." and len(fake.requests) == 3  # no extra call: the retry has the result
    retry = fake.bodies[2]
    ask = {"role": "user", "content": policy.answer_retry_prompt}
    assert retry["messages"] == [*fake.bodies[1]["messages"], CALC_MSG, CALC_RESULT, ask]
    assert retry["tool_choice"] == "none" and retry["chat_template_kwargs"] == {"enable_thinking": False}
    assert "tool call written as text: calculation run, final answer asked again" in reasoning_text(r)


QUESTION = [{"role": "user", "content": "Give A1: the mass in g and A2: the energy in J."}]


def test_calculation_written_in_the_retry_is_run_and_asked_once_more(client, fake, cfg):
    policy = cfg.reasoning_policy
    policy.max_search_depth, policy.max_calc_rounds = 1, 0
    fake.scripts = [lookup_round(), [reasoning("all in the thinking")],
                    [content("A1: 480 g\nLet me verify the energy:\n" + XML_CALL)], [content("A1: 480 g\nA2: 12 J")]]
    r = chat(client, QUESTION)
    assert answer(r) == "A1: 480 g\nA2: 12 J" and len(fake.requests) == 4
    last = fake.bodies[3]
    ask = {"role": "user", "content": policy.answer_retry_prompt}
    assert last["messages"] == [*fake.bodies[2]["messages"], {**CALC_MSG, "content": "A1: 480 g\nLet me verify the energy:"},
                                CALC_RESULT, ask]
    assert last["tool_choice"] == "none" and last["chat_template_kwargs"] == {"enable_thinking": False}
    assert "tool call written as text in the retry: calculation run, final answer asked once more" in reasoning_text(r)
    assert "answer kept" not in reasoning_text(r)


def test_one_more_call_at_most_and_the_most_complete_answer_kept(client, fake, cfg):
    policy = cfg.reasoning_policy
    policy.max_search_depth, policy.max_calc_rounds = 1, 0
    fake.scripts = [lookup_round(), [content("A1: 480 g\nLet me check:\n" + XML_CALL)],
                    [content("A1: 480 g\nA2: 12 J\nLet me verify the energy:\n" + XML_CALL)],
                    [content("Let me recompute everything once more, step by step, to be sure.\n" + XML_CALL)]]
    fake.chunks = [content("never asked")]
    r = chat(client, QUESTION)
    assert len(fake.requests) == 4  # the lookup, the draft, the retry and one more call
    assert answer(r) == "A1: 480 g\nA2: 12 J\nLet me verify the energy:"  # the retry: both requested lines
    text = reasoning_text(r)
    assert text.count("calculation run") == 2 and "tool call written as text in the retry: removed\n" in text
    assert "answer kept: the retry (2 requested answer lines, ends at a removed tool call, " in text


def test_lookup_written_in_the_retry_removed_and_the_best_answer_kept(client, fake, cfg):
    cfg.reasoning_policy.max_search_depth, cfg.reasoning_policy.max_calc_rounds = 1, 0
    fake.scripts = [lookup_round(), [content("**A1:** 194.19 g/mol\nA2: C8H10N4O2\n" + LOOKUP_CALL)],
                    [content("I should look the formula up again to be sure of every atom.\n" + LOOKUP_CALL)]]
    r = chat(client, [{"role": "user", "content": "Give A1: the molar mass and A2: the formula of caffeine."}])
    assert answer(r) == "**A1:** 194.19 g/mol\nA2: C8H10N4O2" and len(fake.requests) == 3
    assert "tool call written as text in the retry: removed\n" in reasoning_text(r)
    assert "answer kept: the draft (2 requested answer lines, ends at a removed tool call, " in reasoning_text(r)


def test_complete_answer_kept_over_a_longer_fragment(client, fake):
    unknown = "<function=browse>\n<parameter=url>\nx\n</parameter>\n</function>"
    fake.scripts = [[content(unknown + "\nCaffeine is 194.19 g/mol.")],
                    [content("I should look this up once more before I answer, to be precise.\n" + LOOKUP_CALL)]]
    r = chat(client)
    assert answer(r) == "Caffeine is 194.19 g/mol." and len(fake.requests) == 2
    assert "answer kept: the draft (0 requested answer lines, complete, 25 characters)" in reasoning_text(r)


def test_tool_call_written_as_text_is_run_when_rounds_are_left(client, fake):
    fake.scripts = [[content("I compute it.\n" + XML_CALL)], [content("It is 480.")]]
    r = chat(client)
    assert answer(r) == "It is 480." and len(fake.requests) == 2
    call, result = fake.bodies[1]["messages"][-2:]
    assert call == {"role": "assistant", "content": "I compute it.", "tool_calls": [{
        "id": "call_0", "type": "function", "function": {"name": "calculate", "arguments": '{"expressions": "0.48 * 1000"}'}}]}
    assert result["tool_call_id"] == "call_0" and result["content"].startswith("0.48 * 1000 = 480")
    assert "tool call written as text: run" in reasoning_text(r)


def test_json_tool_call_written_as_text(client, fake):
    fake.scripts = [[content('<tool_call>\n{"name": "lookup_compound", "arguments": {"query": "caffeine"}}\n</tool_call>')],
                    [content("Caffeine [S1].")]]
    assert answer(chat(client)) == "Caffeine."
    assert fake.bodies[1]["messages"][-1]["content"].startswith("[S1] Caffeine")


def test_unknown_tool_written_as_text_removed_and_no_answer_dropped(client, fake):
    unknown = "<function=browse>\n<parameter=url>\nx\n</parameter>\n</function>"
    fake.scripts = [[content("Caffeine is 194.19 g/mol.\n" + unknown)], [content(LOOKUP_CALL)]]
    r = chat(client)
    assert answer(r) == "Caffeine is 194.19 g/mol." and len(fake.requests) == 2  # the retry wrote only a call
    assert "removed, final answer asked again" in reasoning_text(r) and "no answer on the retry" in reasoning_text(r)


def test_empty_answer_asked_again_once(client, fake, cfg):
    fake.scripts = [[reasoning("the whole answer, A1: 42"), finish("stop")], [content("A1: 42")]]
    r = chat(client)
    assert answer(r) == "A1: 42" and len(fake.requests) == 2
    retry = fake.bodies[1]
    assert retry["messages"][-1] == {"role": "user", "content": cfg.reasoning_policy.answer_retry_prompt}
    assert retry["tool_choice"] == "none" and retry["chat_template_kwargs"] == {"enable_thinking": False}
    assert "empty answer: final answer asked again" in reasoning_text(r)


def test_no_answer_said_plainly(cfg, fake, built_store):
    cfg.reasoning_policy.max_search_depth = 0
    fake.scripts = [[reasoning("a"), finish("stop")], [reasoning("b"), finish("stop")]]
    with TestClient(create_app(cfg, fake.client(cfg.model_backend), Knowledge(L1Store(built_store), cfg))) as c:
        r = chat(c)
    assert answer(r) == NO_ANSWER and len(fake.requests) == 2 and r.text.endswith("data: [DONE]\n\n")
    assert "tools" not in fake.bodies[1] and "tool_choice" not in fake.bodies[1]


def test_safety_nets_switched_off(cfg, fake, knowledge):
    cfg.reasoning_policy.text_tool_calls = cfg.reasoning_policy.empty_answer_retry = False
    fake.scripts = [[content(XML_CALL)], [reasoning("only thinking")]]
    with TestClient(create_app(cfg, fake.client(cfg.model_backend), knowledge)) as c:
        assert "<function=calculate>" in answer(chat(c)) and len(fake.requests) == 1
        assert answer(chat(c)) == NO_ANSWER and len(fake.requests) == 2


def test_written_calculations_switched_off(client, fake, cfg):
    policy = cfg.reasoning_policy
    policy.max_search_depth, policy.max_calc_rounds, policy.run_written_calculations = 1, 0, False
    fake.scripts = [lookup_round(), [content(XML_CALL)], [content("It is 480 g.\n" + XML_CALL)]]
    fake.chunks = [content("never asked")]
    r = chat(client)
    assert answer(r) == "It is 480 g." and len(fake.requests) == 3
    assert fake.bodies[2]["messages"] == [*fake.bodies[1]["messages"], {"role": "user", "content": policy.answer_retry_prompt}]
    text = reasoning_text(r)
    assert "tool call written as text: removed, final answer asked again" in text and "calculation run" not in text
    assert "tool call written as text in the retry: removed\n" in text


TEN_LINES = "Check the report's eight statements. Finish with exactly these lines:\n" + "\n".join(
    f"A{i}: <verdict {i}>" for i in range(1, 11))
EIGHT = "Statements 7 and 8 are discussed above.\n" + "\n".join(f"A{i}: true" for i in range(1, 9))


def ask(client, question):
    return chat(client, [{"role": "user", "content": question}])


def test_missing_answer_lines_asked_for_once_and_appended(client, fake, cfg):
    policy = cfg.reasoning_policy
    fake.scripts = [[content(EIGHT), finish("stop")], [content("A9: false\nA10: true")]]
    r = ask(client, TEN_LINES)
    assert answer(r) == EIGHT + "\nA9: false\nA10: true" and len(fake.requests) == 2
    repair = fake.bodies[1]
    assert "tools" not in repair and "tool_choice" not in repair
    assert repair["chat_template_kwargs"] == {"enable_thinking": False}  # answer_retry_request
    assert repair["messages"] == [
        {"role": "system", "content": policy.system_prompt}, {"role": "user", "content": TEN_LINES},
        {"role": "assistant", "content": EIGHT},
        {"role": "user", "content": policy.coverage_repair_prompt.replace("{lines}", "A9: <verdict 9>\nA10: <verdict 10>")}]
    trace = reasoning_text(r)
    assert "answer lines missing: A9, A10; asked for them" in trace and "answer lines added: A9, A10" in trace


@pytest.mark.anyio
async def test_draft_item_has_the_appended_lines(cfg, fake, knowledge):  # what the checks and the evals get
    fake.scripts = [[content(EIGHT)], [content("A9: false\nA10: true")]]
    items = [i async for i in Agent(cfg, fake.client(cfg.model_backend), knowledge).run(
        [{"role": "user", "content": TEN_LINES}])]
    assert {"draft": EIGHT + "\nA9: false\nA10: true"} in items


def test_complete_answer_needs_no_extra_call(client, fake):
    draft = "\n".join(f"- **A{i}:** true" for i in range(1, 11))
    fake.scripts = [[content(draft)]]
    r = ask(client, TEN_LINES)
    assert answer(r) == draft and len(fake.requests) == 1 and "answer lines" not in reasoning_text(r)


def test_only_the_missing_labelled_lines_of_the_reply_are_used(client, fake):
    question = "Claim 1: the mass is 5 g. Claim 2: it is dry. Finish with: A1: <claim 1> A2: <claim 2>"
    fake.scripts = [[content("A1: true")],
                    [content("Here you go.\nA1: false\nClaim 2: true\nB7: x\nA2: <claim 2>\n**A2:** false, no data\nA2: x")]]
    r = ask(client, question)
    assert answer(r) == "A1: true\n**A2:** false, no data" and len(fake.requests) == 2


def test_labels_the_task_does_not_ask_for_are_never_requested(client, fake):
    fake.scripts = [[content("A1: true, the mass matches.")]]
    assert answer(ask(client, "Claim 1: the mass is 5 g. Answer as A1: <claim 1>")) == "A1: true, the mass matches."
    assert len(fake.requests) == 1


@pytest.mark.parametrize("reply, note", [
    (CONNECT_ERROR, "answer line request failed"),
    ([content("I cannot add more.")], "no requested answer line in the reply: answer kept"),
])
def test_repair_that_fails_keeps_the_answer(client, fake, reply, note):
    fake.scripts = [[content(EIGHT)], reply]
    r = ask(client, TEN_LINES)
    assert answer(r) == EIGHT and len(fake.requests) == 2 and note in reasoning_text(r)
    assert r.text.endswith("data: [DONE]\n\n")


def test_coverage_checked_on_the_answer_kept(client, fake, cfg):  # after the most complete reply is chosen
    cfg.reasoning_policy.max_search_depth, cfg.reasoning_policy.max_calc_rounds = 1, 0
    kept = "**A1:** 194.19 g/mol\nA2: C8H10N4O2"
    fake.scripts = [lookup_round(), [content(kept + "\n" + LOOKUP_CALL)],
                    [content("I should look the formula up again to be sure of every atom.\n" + LOOKUP_CALL)],
                    [content("A3: 58-08-2")]]
    r = ask(client, "For caffeine give A1: <molar mass> A2: <formula> A3: <CAS number>")
    assert answer(r) == kept + "\nA3: 58-08-2" and len(fake.requests) == 4
    assert fake.bodies[3]["messages"][-2] == {"role": "assistant", "content": kept}
    text = reasoning_text(r)
    assert text.index("answer kept: the draft") < text.index("answer lines missing: A3") < text.index("added: A3")


def test_coverage_repair_switched_off(client, fake, cfg):
    cfg.reasoning_policy.coverage_repair = False
    fake.scripts = [[content(EIGHT)]]
    assert answer(ask(client, TEN_LINES)) == EIGHT and len(fake.requests) == 1


def test_markup_read_also_when_cut_off(cfg):
    markup = _Markup(cfg.reasoning_policy.tool_call_markup)
    calls, rest = markup.split("Plan.\n<tool_call>\n<function=calculate>\n<parameter=expressions>\n2 * 3")
    assert rest == "Plan." and [c["function"] for c in calls] == [{"name": "calculate", "arguments": '{"expressions": "2 * 3"}'}]
    calls, rest = markup.split("<function=search_knowledge><parameter=query>a b</parameter>"
                               "<parameter=domain>physics</parameter></function>\n</tool_call> x")
    assert rest == "x" and json.loads(calls[0]["function"]["arguments"]) == {"query": "a b", "domain": "physics"}
    assert [c["function"]["name"] for c in markup.split("<tool_call>not json</tool_call>")[0]] == [""]
    assert markup.split("no markup") == ([], "no markup")
    assert markup.cut("Plan.\n" + XML_CALL + "\n ") and markup.cut("Plan.\n<function=calculate><parameter=expressions>2")
    assert not markup.cut(XML_CALL + "\nIt is 480.") and not markup.cut("no markup")
