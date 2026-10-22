"""Shared fixtures: a scripted fake upstream (no GPU, no network) and a temp copy of the shipped config."""

import json
import shutil
from pathlib import Path

import httpx
import pytest

from haisenberg.config import load_config
from haisenberg.llm import LLMClient

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def anyio_backend():
    return "asyncio"


def reasoning(text):
    return {"choices": [{"index": 0, "delta": {"reasoning_content": text}}]}


def content(text):
    return {"choices": [{"index": 0, "delta": {"content": text}}]}


class ScriptedStream(httpx.AsyncByteStream):
    """SSE body that can drop the connection after `fail_after` events."""

    def __init__(self, events: list[bytes], fail_after: int | None = None):
        self.events, self.fail_after, self.closed = events, fail_after, False

    async def __aiter__(self):
        for i, e in enumerate(self.events):
            if i == self.fail_after:
                raise httpx.ReadError("connection dropped")
            yield e

    async def aclose(self):
        self.closed = True


class FakeUpstream:
    """Records every request and answers with scripted chunks, an HTTP error, or a connect error."""

    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.streams: list[ScriptedStream] = []
        self.chunks: list[dict] = [reasoning("a"), content("b")]
        self.status = 200
        self.error_body: dict | None = None
        self.connect_error = False
        self.fail_after: int | None = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.connect_error:
            raise httpx.ConnectError("connection refused", request=request)
        if self.status != 200:
            return httpx.Response(self.status, json=self.error_body or {"error": {"message": "bad"}})
        events = [f"data: {json.dumps(c)}\n\n".encode() for c in self.chunks] + [b"data: [DONE]\n\n"]
        stream = ScriptedStream(events, self.fail_after)
        self.streams.append(stream)
        return httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})

    def client(self, backend_cfg) -> LLMClient:
        return LLMClient(backend_cfg, http=httpx.AsyncClient(transport=httpx.MockTransport(self.handler)))

    @property
    def last_body(self) -> dict:
        return json.loads(self.requests[-1].content)


@pytest.fixture
def fake():
    return FakeUpstream()


@pytest.fixture
def cfg(tmp_path):
    """The shipped config, loaded from a temp copy (no .env) with an empty UI folder."""
    dest = tmp_path / "config" / "config.yaml"
    dest.parent.mkdir(parents=True)
    shutil.copy(REPO / "config" / "config.yaml", dest)
    c = load_config(dest)
    c.ui.dist_dir = tmp_path / "dist"
    return c
