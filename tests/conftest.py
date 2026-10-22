"""Shared fixtures: a scripted fake upstream (no GPU, no network), a temp copy of the shipped config,
and a Layer 1 store built from the test fixtures."""

import asyncio
import json
import shutil
from pathlib import Path

import httpx
import pytest

from haisenberg.config import load_config
from haisenberg.llm import LLMClient

REPO = Path(__file__).resolve().parents[1]
CONNECT_ERROR = "connect_error"  # FakeUpstream script entries: refuse that request,
DISCONNECT = "disconnect"  # close the connection without a response (a WSL relay with nothing behind it),
UNAVAILABLE = "unavailable"  # or answer 503 (llama-server loading)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def reasoning(text):
    return {"choices": [{"index": 0, "delta": {"reasoning_content": text}}]}


def content(text):
    return {"choices": [{"index": 0, "delta": {"content": text}}]}


def tool_call(name=None, arguments="", index=0, call_id=None):
    """One `delta.tool_calls` fragment; name and id usually come only with the first fragment."""
    frag = {"index": index, "function": {"arguments": arguments}}
    if call_id:
        frag["id"], frag["type"] = call_id, "function"
    if name:
        frag["function"]["name"] = name
    return {"choices": [{"index": 0, "delta": {"tool_calls": [frag]}}]}


def finish(reason):
    return {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]}


class ScriptedStream(httpx.AsyncByteStream):
    """SSE body that can drop the connection after `fail_after` events, or pause `gap_s` between events."""

    def __init__(self, events: list[bytes], fail_after: int | None = None, gap_s: float = 0.0):
        self.events, self.fail_after, self.gap_s, self.closed = events, fail_after, gap_s, False

    async def __aiter__(self):
        for i, e in enumerate(self.events):
            if i == self.fail_after:
                raise httpx.ReadError("connection dropped")
            if i and self.gap_s:
                await asyncio.sleep(self.gap_s)
            yield e

    async def aclose(self):
        self.closed = True


class FakeUpstream:
    """Records every request and answers with scripted chunks, an HTTP error, or a connect error.

    `scripts` answers requests in order (a chunk list, CONNECT_ERROR, DISCONNECT or UNAVAILABLE); once it is
    empty, `chunks` answers. `delay_s`: silence before each answer starts (a queued or slow backend);
    `gap_s`: silence between the events of an answer.
    """

    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.streams: list[ScriptedStream] = []
        self.chunks: list[dict] = [reasoning("a"), content("b")]
        self.scripts: list = []
        self.status = 200
        self.error_body: dict | None = None
        self.connect_error = False
        self.fail_after: int | None = None
        self.delay_s = 0.0
        self.gap_s = 0.0

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        chunks = self.scripts.pop(0) if self.scripts else self.chunks
        if self.connect_error or chunks == CONNECT_ERROR:
            raise httpx.ConnectError("connection refused", request=request)
        if chunks == DISCONNECT:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.", request=request)
        if chunks == UNAVAILABLE:
            return httpx.Response(503, json={"error": {"message": "Loading model"}})
        await asyncio.sleep(self.delay_s)
        if self.status != 200:
            return httpx.Response(self.status, json=self.error_body or {"error": {"message": "bad"}})
        events = [f"data: {json.dumps(c)}\n\n".encode() for c in chunks] + [b"data: [DONE]\n\n"]
        stream = ScriptedStream(events, self.fail_after, self.gap_s)
        self.streams.append(stream)
        return httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})

    def client(self, backend_cfg) -> LLMClient:
        return LLMClient(backend_cfg, http=httpx.AsyncClient(transport=httpx.MockTransport(self.handler)))

    @property
    def last_body(self) -> dict:
        return json.loads(self.requests[-1].content)

    @property
    def bodies(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests]


@pytest.fixture
def fake():
    return FakeUpstream()


@pytest.fixture
def cfg(tmp_path):
    """The shipped config, loaded from a temp copy (no .env) with an empty UI folder and no store,
    so tests get no tools unless they pass a store, even when DATA_DIR points at real data."""
    dest = tmp_path / "config" / "config.yaml"
    dest.parent.mkdir(parents=True)
    shutil.copy(REPO / "config" / "config.yaml", dest)
    c = load_config(dest)
    c.model_backend.wait_s = 0  # fail fast; the retry tests set their own
    c.ui.dist_dir = tmp_path / "dist"
    c.retrieval.layer1_encyclopedia_db = tmp_path / "no-store.db"
    return c


@pytest.fixture(scope="session")
def built_store(tmp_path_factory) -> Path:
    """One Layer 1 store built from the fixtures, shared read-only by the tests."""
    from ingest.build_l1 import build
    from l1.conftest import make_raw
    from l1.l1_helpers import write_manifest

    root = tmp_path_factory.mktemp("l1store")
    db = root / "l1.db"
    build(write_manifest(root / "sources.yaml"), make_raw(root), db)
    return db
