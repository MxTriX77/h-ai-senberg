import json

import pytest
from fastapi.testclient import TestClient

from conftest import content, reasoning
from haisenberg.server import create_app


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
    assert deltas == [
        {"reasoning_content": "a"}, {"reasoning_content": "b"}, {"content": "c"}, {"content": "d"},
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
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "answer", "reasoning_content": "think"}


def test_missing_messages_400(client):
    r = client.post("/v1/chat/completions", json={"stream": True})
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"


def test_non_json_400(client):
    r = client.post("/v1/chat/completions", content=b"not json", headers={"content-type": "application/json"})
    assert r.status_code == 400


def test_upstream_down_502(client, fake):
    fake.connect_error = True
    r = chat(client)
    assert r.status_code == 502
    assert r.headers["content-type"].startswith("application/json")
    assert "unreachable" in r.json()["error"]["message"]


def test_upstream_4xx_kept(client, fake):
    fake.status, fake.error_body = 400, {"error": {"message": "context too long"}}
    r = chat(client)
    assert r.status_code == 400 and r.json()["error"]["message"] == "context too long"


def test_mid_stream_drop_ends_without_done(client, fake):
    fake.chunks = [content("a"), content("b"), content("c")]
    fake.fail_after = 2
    ev = events(chat(client).text)
    assert [e["choices"][0]["delta"] for e in ev] == [{"content": "a"}, {"content": "b"}]


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
