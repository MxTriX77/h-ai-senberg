import pytest

from conftest import content, reasoning
from haisenberg.llm import UpstreamError

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
