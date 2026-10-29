import asyncio
import json
import logging

import pytest
from fastapi.testclient import TestClient

from conftest import content, reasoning
from haisenberg import server
from haisenberg.agent import DRAFT_MARK, Agent
from haisenberg.server import KEEPALIVE, create_app
from haisenberg.tools import Knowledge

BUDGET = "\n\n> thinking budget: 1024 tokens\n"  # the shipped minimum: every test question is short


@pytest.fixture
def client(cfg, fake):
    with TestClient(create_app(cfg, fake.client(cfg.model_backend))) as c:
        yield c


def events(text: str) -> list:
    """Parse an SSE body into JSON chunks, keeping the literal "[DONE]"."""
    out = []
    for line in text.split("\n"):
        if line.startswith("data: "):
            data = line[6:]
            out.append(data if data == "[DONE]" else json.loads(data))
    return out


def chat(client, messages=None, **extra):
    body = {"stream": True, "messages": messages or [{"role": "user", "content": "hi"}], **extra}
    return client.post("/v1/chat/completions", json=body)


def test_thinking_then_answer(client, fake):
    fake.chunks = [reasoning("a"), reasoning("b"), content("c"), content("d")]
    r = chat(client)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    ev = events(r.text)
    deltas = [e["choices"][0]["delta"] for e in ev[:-2]]
    assert deltas == [  # the draft streams as reasoning; the answer is sent once the model is done
        {"reasoning_content": "a"}, {"reasoning_content": "b"}, {"reasoning_content": DRAFT_MARK},
        {"reasoning_content": "c"}, {"reasoning_content": "d"}, {"reasoning_content": BUDGET}, {"content": "cd"},
    ]
    assert ev[-2]["choices"][0] == {"index": 0, "delta": {}, "finish_reason": "stop"}
    assert ev[-1] == "[DONE]"


def test_same_id_and_model_on_every_chunk(client, cfg, fake):
    fake.chunks = [reasoning("a"), content("b"), content("c")]
    ev = events(chat(client).text)[:-1]
    assert len({e["id"] for e in ev}) == 1
    assert {e["model"] for e in ev} == {cfg.model_backend.model_id}


def test_client_system_message_replaced(client, cfg, fake):
    chat(client, [{"role": "system", "content": "be rude"}, {"role": "user", "content": "hi"}])
    assert fake.last_body["messages"] == [
        {"role": "system", "content": cfg.reasoning_policy.system_prompt},
        {"role": "user", "content": "hi"},
    ]


def test_client_sampling_and_model_ignored(client, cfg, fake):
    chat(client, temperature=2.0, model="other", chat_template_kwargs={"enable_thinking": False})
    body = fake.last_body
    assert body["temperature"] == cfg.model_backend.temperature
    assert body["model"] == cfg.model_backend.model_id
    assert "chat_template_kwargs" not in body


def test_past_reasoning_not_forwarded(client, fake):
    chat(client, [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "thoughts", "timings": {}},
        {"role": "user", "content": "q2"},
    ])
    assert fake.last_body["messages"][2] == {"role": "assistant", "content": "a1"}


def test_non_streaming(client, fake):
    fake.chunks = [reasoning("th"), reasoning("ink"), content("ans"), content("wer")]
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {
        "role": "assistant", "content": "answer", "reasoning_content": "think" + DRAFT_MARK + "answer" + BUDGET,
    }


def test_missing_messages_400(client):
    r = client.post("/v1/chat/completions", json={"stream": True})
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"


def test_non_json_400(client):
    r = client.post("/v1/chat/completions", content=b"not json", headers={"content-type": "application/json"})
    assert r.status_code == 400


def failure_ending(ev: list) -> str:
    """Assert that a stream ended properly after a failure; return the message it showed."""
    assert ev[-1] == "[DONE]" and ev[-2]["choices"][0]["finish_reason"] == "stop"
    message = ev[-3]["choices"][0]["delta"]["content"]
    assert "could not be completed" in message and "send the question again" in message
    return message


def test_upstream_down_streaming_says_so(client, fake):
    fake.connect_error = True
    r = chat(client)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert "unreachable" in failure_ending(events(r.text))


def test_upstream_down_non_streaming_502(client, fake):
    fake.connect_error = True
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 502
    assert r.headers["content-type"].startswith("application/json")
    assert "unreachable" in r.json()["error"]["message"]


def test_upstream_4xx(client, fake):
    fake.status, fake.error_body = 400, {"error": {"message": "context too long"}}
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 400 and r.json()["error"]["message"] == "context too long"
    assert "context too long" in failure_ending(events(chat(client).text))


def test_mid_stream_drop_ends_with_message_and_done(client, fake, caplog):
    fake.chunks = [content("a"), content("b"), content("c")]
    fake.fail_after = 2
    with caplog.at_level(logging.ERROR):
        ev = events(chat(client).text)
    assert [e["choices"][0]["delta"] for e in ev[:3]] == [
        {"reasoning_content": DRAFT_MARK}, {"reasoning_content": "a"}, {"reasoning_content": "b"},
    ]
    failure_ending(ev)
    assert len(ev) == 6 and "mid-stream" in caplog.text


def test_unexpected_error_ends_with_message_and_done(cfg, fake, monkeypatch, caplog):
    class Broken:
        def __init__(self, *args):
            pass

        async def run(self, history):
            yield {"reasoning_content": "thinking"}
            raise RuntimeError("boom")

    monkeypatch.setattr(server, "Agent", Broken)
    with TestClient(create_app(cfg, fake.client(cfg.model_backend))) as c, caplog.at_level(logging.ERROR):
        ev = events(chat(c).text)
    assert ev[0]["choices"][0]["delta"] == {"reasoning_content": "thinking"}
    assert "RuntimeError: boom" in failure_ending(ev) and "chat stream failed" in caplog.text


def test_keep_alive_while_the_backend_is_silent(cfg, fake):
    cfg.app.keepalive_s = 0.05
    fake.delay_s, fake.chunks = 0.5, [reasoning("a"), content("b")]
    with TestClient(create_app(cfg, fake.client(cfg.model_backend))) as c:
        text = chat(c).text
    first_data = text.index("data:")
    assert text[:first_data].count(KEEPALIVE) >= 3  # sent before the first model output
    deltas = [e["choices"][0]["delta"] for e in events(text)[:-2]]
    assert deltas == [{"reasoning_content": "a"}, {"reasoning_content": DRAFT_MARK},
                      {"reasoning_content": "b"}, {"reasoning_content": BUDGET}, {"content": "b"}]


@pytest.mark.anyio
async def test_client_gone_closes_upstream(cfg, fake):
    fake.chunks, fake.gap_s = [reasoning("a"), reasoning("b")], 5
    agent = Agent(cfg, fake.client(cfg.model_backend), Knowledge.open(cfg))
    sse = server._sse("id", 0, "m", agent.run([{"role": "user", "content": "hi"}]), 0.05)
    assert '"reasoning_content": "a"' in await anext(sse)
    assert await anext(sse) == KEEPALIVE  # waiting for "b"
    await sse.aclose()
    for _ in range(20):
        if fake.streams[0].closed:
            break
        await asyncio.sleep(0.01)
    assert fake.streams[0].closed and len(fake.requests) == 1


def test_models(client, cfg):
    assert client.get("/v1/models").json()["data"] == [
        {"id": cfg.model_backend.model_id, "object": "model", "owned_by": "haisenberg"}
    ]


def test_props(client, cfg):
    p = client.get("/props?autoload=false").json()
    assert p["role"] == "model"
    assert p["default_generation_settings"]["params"]
    assert p["default_generation_settings"]["n_ctx"] == cfg.llama_server.ctx_size
    assert p["ui_settings"] == cfg.ui.settings
    assert p["chat_template"] == ""


def test_health_and_stubs(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/tools").json() == []
    assert client.post("/v1/streams/lookup", json={}).json() == []


def test_ui_not_built(client):
    r = client.get("/")
    assert r.status_code == 404 and "build_ui.py" in r.text
    assert chat(client).status_code == 200


def test_ui_built(cfg, fake, tmp_path):
    cfg.ui.dist_dir.mkdir()
    (cfg.ui.dist_dir / "index.html").write_text("<html>ui</html>", encoding="utf-8")
    with TestClient(create_app(cfg, fake.client(cfg.model_backend))) as c:
        assert c.get("/").text == "<html>ui</html>"
        assert c.get("/health").json() == {"status": "ok"}
