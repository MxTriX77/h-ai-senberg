import json

import httpx
import numpy as np
import pytest

from conftest import CONNECT_ERROR, DISCONNECT, UNAVAILABLE, content, reasoning
from haisenberg import llm
from haisenberg.config import load_config
from haisenberg.llm import EmbeddingClient, LLMClient, UpstreamError, estimate_tokens

pytestmark = pytest.mark.anyio

USER = [{"role": "user", "content": "hi"}]


async def drain(client, messages=USER):
    return [c async for c in client.stream_chat(messages)]


async def test_request_body_and_extra_body_override(cfg, fake):
    b = cfg.model_backend
    b.model_id, b.temperature, b.max_tokens = "m", 0.6, 8192
    b.extra_body = {"top_k": 20, "max_tokens": 100}
    await drain(fake.client(b))
    req = fake.requests[-1]
    assert str(req.url) == b.api_base.rstrip("/") + "/chat/completions"
    assert fake.last_body == {
        "model": "m", "messages": USER, "stream": True, "temperature": 0.6, "max_tokens": 100, "top_k": 20,
    }


async def test_per_call_fields_win(cfg, fake):
    b = cfg.model_backend
    b.extra_body = {"top_k": 20}
    tools = [{"type": "function", "function": {"name": "t", "parameters": {"type": "object"}}}]
    [c async for c in fake.client(b).stream_chat(USER, tools=tools, temperature=0)]
    body = fake.last_body
    assert body["tools"] == tools and body["temperature"] == 0 and body["top_k"] == 20
    assert body["model"] == b.model_id and body["max_tokens"] == b.max_tokens


@pytest.mark.parametrize("value, seed", [(None, None), ("", None), ("7", 7)])
async def test_seed_sent_only_when_set(cfg, fake, monkeypatch, value, seed):
    if value is None:
        monkeypatch.delenv("EVAL_SEED", raising=False)
    else:
        monkeypatch.setenv("EVAL_SEED", value)
    await drain(fake.client(load_config(cfg.root / "config" / "config.yaml").model_backend))
    body = fake.last_body
    assert ("seed" in body, body.get("seed")) == (seed is not None, seed)
    assert body["temperature"] == cfg.model_backend.temperature


async def test_no_auth_header_without_key(cfg, fake):
    cfg.model_backend.api_key = ""
    await drain(fake.client(cfg.model_backend))
    assert "authorization" not in fake.requests[-1].headers


async def test_bearer_header_with_key(cfg, fake):
    cfg.model_backend.api_key = "sk-test"
    await drain(fake.client(cfg.model_backend))
    assert fake.requests[-1].headers["authorization"] == "Bearer sk-test"


async def test_reasoning_then_content_in_order(cfg, fake):
    fake.chunks = [reasoning("r1"), reasoning("r2"), content("c1"), content("c2")]
    assert await drain(fake.client(cfg.model_backend)) == fake.chunks


async def test_connection_refused(cfg, fake):
    fake.connect_error = True
    with pytest.raises(UpstreamError) as e:
        await drain(fake.client(cfg.model_backend))
    assert e.value.status is None and "unreachable" in e.value.message


async def test_upstream_400_message(cfg, fake):
    fake.status, fake.error_body = 400, {"error": {"message": "context too long"}}
    with pytest.raises(UpstreamError) as e:
        await drain(fake.client(cfg.model_backend))
    assert e.value.status == 400 and e.value.message == "context too long"


async def test_early_close_closes_upstream(cfg, fake):
    fake.chunks = [content("a"), content("b"), content("c")]
    stream = fake.client(cfg.model_backend).stream_chat(USER)
    assert await anext(stream) == content("a")
    await stream.aclose()
    assert fake.streams[-1].closed


async def test_slow_backend_is_waited_for(cfg, fake):
    fake.delay_s, fake.chunks = 0.3, [content("late")]
    assert await drain(fake.client(cfg.model_backend)) == [content("late")]


async def test_no_read_limit_by_default(cfg):
    assert cfg.model_backend.timeout_s is None
    timeout = LLMClient(cfg.model_backend)._http.timeout
    assert timeout.read is None and timeout.connect == 10.0
    cfg.model_backend.timeout_s = 1
    assert LLMClient(cfg.model_backend)._http.timeout.read == 1


async def test_read_limit_reached(cfg):
    def handler(request):
        raise httpx.ReadTimeout("no answer", request=request)

    client = LLMClient(cfg.model_backend, http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(UpstreamError, match="timed out"):
        await drain(client)


async def test_backend_comes_back(cfg, fake, monkeypatch):
    monkeypatch.setattr(llm, "RETRY_S", 0.01)
    cfg.model_backend.wait_s = 5
    fake.scripts = [CONNECT_ERROR, DISCONNECT, UNAVAILABLE, [content("ok")]]
    assert await drain(fake.client(cfg.model_backend)) == [content("ok")]
    assert len(fake.requests) == 4


async def test_unreachable_after_waiting(cfg, fake, monkeypatch):
    monkeypatch.setattr(llm, "RETRY_S", 0.01)
    cfg.model_backend.wait_s = 0.1
    fake.connect_error = True
    with pytest.raises(UpstreamError, match="unreachable"):
        await drain(fake.client(cfg.model_backend))
    assert len(fake.requests) > 2


async def test_upstream_400_not_retried(cfg, fake):
    cfg.model_backend.wait_s = 5
    fake.status, fake.error_body = 400, {"error": {"message": "context too long"}}
    with pytest.raises(UpstreamError):
        await drain(fake.client(cfg.model_backend))
    assert len(fake.requests) == 1


def embedding_client(cfg, handler):
    return EmbeddingClient(cfg.embedding, http=httpx.Client(transport=httpx.MockTransport(handler)))


def test_embeddings_in_input_order_and_normalised(cfg):
    seen = []

    def handler(request):
        seen.append((str(request.url), json.loads(request.content)))
        return httpx.Response(200, json={"data": [{"index": 1, "embedding": [0, 2]}, {"index": 0, "embedding": [3, 4]}]})

    v = embedding_client(cfg, handler).embed(["a", "b"])
    assert seen == [(cfg.embedding.api_base + "/embeddings", {"model": cfg.embedding.model_id, "input": ["a", "b"]})]
    assert v.dtype == np.float32 and np.allclose(v, [[0.6, 0.8], [0, 1]])


def test_embeddings_error_keeps_status(cfg):
    def handler(request):
        return httpx.Response(400, json={"error": {"message": "input (600 tokens) is larger than the max context size"}})

    with pytest.raises(UpstreamError, match="larger than") as err:
        embedding_client(cfg, handler).embed(["long"])
    assert err.value.status == 400


def test_embeddings_backend_unreachable(cfg):
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(UpstreamError, match="unreachable") as err:
        embedding_client(cfg, handler).embed(["a"])
    assert err.value.status is None


def test_estimate_tokens():
    messages = [{"role": "user", "content": "x" * 100},
                {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "f", "arguments": "y" * 9}}]}]
    assert estimate_tokens(messages, 2.0) == 55 + 2 * llm.PER_MESSAGE
    assert estimate_tokens(messages, 2.0, tools=[{"a": 1}]) > estimate_tokens(messages, 2.0)
