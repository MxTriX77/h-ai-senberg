"""FastAPI app: OpenAI-compatible chat endpoint (D-006), the endpoints the WebUI calls, and the built UI."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .agent import Agent
from .config import Config
from .llm import LLMClient, UpstreamError
from .tools import Knowledge

log = logging.getLogger(__name__)

_FIELDS = ("reasoning_content", "content")
_END = object()
KEEPALIVE = ": keep-alive\n\n"  # SSE comment: keeps proxies and the WebUI (3 s silence check) from dropping us


def create_app(cfg: Config, llm: LLMClient | None = None, knowledge: Knowledge | None = None) -> FastAPI:
    llm = llm or LLMClient(cfg.model_backend)
    agent = Agent(cfg, llm, knowledge or Knowledge.open(cfg))
    model_id = cfg.model_backend.model_id

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await llm.aclose()

    app = FastAPI(title="h-ai-senberg", lifespan=lifespan)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            body = await request.json()
        except ValueError:
            return _error(400, "request body must be JSON", "invalid_request_error")
        messages = body.get("messages") if isinstance(body, dict) else None
        if not isinstance(messages, list) or not messages:
            return _error(400, "`messages` must be a non-empty list", "invalid_request_error")

        # The server owns policy: the agent adds the configured system prompt and tools; client system
        # messages are dropped, history is reduced to role + content, client sampling/model/tools ignored.
        history = [
            {"role": m["role"], "content": m.get("content") or ""}
            for m in messages
            if isinstance(m, dict) and m.get("role") in ("user", "assistant")
        ]

        stream = agent.run(history)
        cid, created = f"chatcmpl-{uuid.uuid4()}", int(time.time())
        if body.get("stream"):  # start at once: waiting for the model (queue, prompt, restart) is covered by keep-alives
            return StreamingResponse(
                _sse(cid, created, model_id, stream, cfg.app.keepalive_s),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )
        try:
            first = await anext(stream, None)
            if first and "route" in first:  # the router's note comes before any model call
                first = await anext(stream, None)
        except UpstreamError as e:
            status = e.status if e.status and 400 <= e.status < 500 else 502
            return _error(status, e.message, "upstream_error")
        try:
            return JSONResponse(await _collect(cid, created, model_id, first, stream))
        except (UpstreamError, httpx.HTTPError) as e:
            return _error(502, f"backend failed mid-response: {e}", "upstream_error")

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "haisenberg"}]}

    @app.get("/props")
    async def props():
        # The subset of llama-server's /props that the b10173 WebUI reads, built from config.
        # Empty chat_template: the WebUI shows no thinking on/off control (thinking is set in config).
        return {
            "role": "model",
            "model_alias": model_id,
            "model_path": "",
            "total_slots": 1,
            "modalities": {"vision": False, "audio": False},
            "chat_template": "",
            "build_info": "h-ai-senberg",
            "ui_settings": cfg.ui.settings,
            "cors_proxy_enabled": False,
            "default_generation_settings": {
                "n_ctx": cfg.llama_server.ctx_size,
                "params": {
                    "temperature": cfg.model_backend.temperature,
                    "max_tokens": cfg.model_backend.max_tokens,
                },
            },
        }

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/tools")
    async def tools():
        return []

    @app.post("/v1/streams/lookup")
    async def streams_lookup():
        return []

    dist = cfg.ui.dist_dir
    if (dist / "index.html").is_file():
        app.mount("/", StaticFiles(directory=dist, html=True), name="ui")
    else:

        @app.get("/")
        async def ui_missing():
            return PlainTextResponse(
                f"The WebUI is not built ({dist} has no index.html). Run: uv run python scripts/build_ui.py",
                status_code=404,
            )

    return app


async def _deltas(first: dict | None, stream: AsyncIterator[dict]):
    """Yield (delta, finish_reason) from the agent's items, keeping only non-empty content and reasoning."""
    try:
        item = first
        while item is not None:
            yield {k: item[k] for k in _FIELDS if item.get(k)}, item.get("finish_reason")
            item = await anext(stream, None)
    finally:
        await stream.aclose()


async def _with_keepalive(stream: AsyncIterator[dict], interval: float):
    """Yield the stream's items, and None whenever `interval` seconds pass without one.
    Closing this generator (client gone) cancels the pending read first, which closes the upstream."""
    pending: asyncio.Task | None = None
    try:
        while True:
            pending = pending or asyncio.ensure_future(anext(stream, _END))
            done, _ = await asyncio.wait({pending}, timeout=interval)
            if not done:
                yield None
                continue
            item, pending = pending.result(), None
            if item is _END:
                return
            yield item
    finally:
        if pending is not None:
            pending.cancel()  # the read unwinds through the agent's cleanup, which closes the upstream request
        else:
            await stream.aclose()


async def _sse(cid: str, created: int, model: str, stream: AsyncIterator[dict], keepalive_s: float):
    def event(delta: dict, finish: str | None = None) -> str:
        chunk = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    finish, items = None, _with_keepalive(stream, keepalive_s)
    try:
        async for item in items:
            if item is None:
                yield KEEPALIVE
                continue
            finish = item.get("finish_reason") or finish
            if delta := {k: item[k] for k in _FIELDS if item.get(k)}:
                yield event(delta)
    except Exception as e:  # status 200 is already sent: say what failed in the answer and end the stream properly
        if isinstance(e, (UpstreamError, httpx.HTTPError)):
            log.error("backend failed mid-stream: %s", e)
        else:
            log.exception("chat stream failed")
        yield event({"content": failure_text(e)})
    finally:
        await items.aclose()  # client gone: stop the agent and its upstream request now, not at garbage collection
    yield event({}, finish or "stop")
    yield "data: [DONE]\n\n"


def failure_text(e: Exception) -> str:
    reason = e.message if isinstance(e, UpstreamError) else f"{type(e).__name__}: {e}"
    return f"\n\n**The answer could not be completed:** {reason}. Please send the question again."


async def _collect(cid: str, created: int, model: str, first: dict | None, stream: AsyncIterator[dict]):
    parts: dict[str, list[str]] = {k: [] for k in _FIELDS}
    finish = None
    async for delta, reason in _deltas(first, stream):
        finish = reason or finish
        for k, v in delta.items():
            parts[k].append(v)
    message: dict[str, Any] = {"role": "assistant", "content": "".join(parts["content"])}
    if parts["reasoning_content"]:
        message["reasoning_content"] = "".join(parts["reasoning_content"])
    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish or "stop"}],
    }


def _error(status: int, message: str, kind: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": kind, "code": status}}, status_code=status)
