"""Request router (request-router spec) and the casual path through the loop (agent-loop spec)."""

import pytest
from fastapi.testclient import TestClient

from conftest import content, reasoning
from haisenberg.agent import DRAFT_MARK, Agent
from haisenberg.router import ANALYTIC, CASUAL, FACTUAL, request_items, route
from haisenberg.server import create_app
from test_server import chat, events


@pytest.fixture
def router(cfg):
    cfg.router.enabled = True
    return cfg.router


@pytest.mark.parametrize("message, depth", [
    ("hi, how are you?", CASUAL),
    ("Thanks!", CASUAL),
    ("привет, как дела?", CASUAL),
    ("Tell me a joke", CASUAL),
    ("What is the boiling point of ethanol?", FACTUAL),
    ("explain entropy", FACTUAL),
    ("hi, what is 2 + 2?", FACTUAL),  # digits: not small talk
    ("Thanks, that was great", CASUAL),
    ("Is it ok to mix bleach and ammonia?", FACTUAL),  # a cue word inside a real question
    ("What is the Great Red Spot?", FACTUAL),
    ("Check these claims:\nA) water boils at 100 °C\nB) ice is denser than water\nC) salt lowers the freezing point", ANALYTIC),
    ("Claim 1: the yield is 85 %. Claim 2: the purity is 99 %.", ANALYTIC),
    ("Проверь расчёт: 2,00 г дают 2,61 г продукта. Верно ли это?", ANALYTIC),
    ("Is this correct? The car covers 120 km in 1.5 h, so its speed is 90 km/h.", ANALYTIC),
    ("x" * 700, ANALYTIC),
])
def test_depth_per_message(router, message, depth):
    assert route(message, router)[0] == depth


def test_router_off(cfg):
    assert route("hi", cfg.router) == (FACTUAL, "router off")


def test_request_items():
    assert request_items("Please:\n1. compute the yield\n2. check the purity\n- and the price") == [
        ("1", "compute the yield"), ("2", "check the purity"), ("3", "and the price")]
    assert request_items("Claim A: x. Claim b: y.") == [("A", "x."), ("B", "y.")]
    assert request_items("а) первое\nб) второе") == [("А", "первое"), ("Б", "второе")]


def test_greeting_streams_a_direct_answer(cfg, fake, router):
    fake.chunks = [content("Hello"), content("!")]
    with TestClient(create_app(cfg, fake.client(cfg.model_backend))) as c:
        r = chat(c, [{"role": "user", "content": "hi"}])
    deltas = [e["choices"][0]["delta"] for e in events(r.text) if e != "[DONE]"]
    assert deltas[:2] == [{"content": "Hello"}, {"content": "!"}] and r.text.endswith("data: [DONE]\n\n")
    assert not any("reasoning_content" in d for d in deltas)
    body = fake.last_body
    assert "tools" not in body and body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["messages"][0] == {"role": "system", "content": cfg.reasoning_policy.casual_prompt}
    assert len(fake.requests) == 1


def test_factual_route_line_and_unchanged_loop(cfg, fake, router):
    fake.chunks = [reasoning("t"), content("Caffeine is 194.19 g/mol.")]
    with TestClient(create_app(cfg, fake.client(cfg.model_backend))) as c:
        r = chat(c, [{"role": "user", "content": "What is the molar mass of caffeine?"}])
    deltas = [e["choices"][0]["delta"] for e in events(r.text) if e != "[DONE]"]
    assert deltas[0] == {"reasoning_content": "Route: factual (a question to answer)\n"}
    assert {"reasoning_content": DRAFT_MARK} in deltas and deltas[-2]["content"] == "Caffeine is 194.19 g/mol."
    assert fake.last_body["messages"][0]["content"] == cfg.reasoning_policy.system_prompt


@pytest.mark.anyio
async def test_route_for_in_process_callers(cfg, fake, router):
    fake.chunks = [content("Hi there")]
    items = [i async for i in Agent(cfg, fake.client(cfg.model_backend), _no_tools(cfg)).run(
        [{"role": "user", "content": "hello"}])]
    assert items[0] == {"route": CASUAL} and {"verification": {}} in items


def _no_tools(cfg):
    from haisenberg.tools import Knowledge

    return Knowledge.open(cfg)
