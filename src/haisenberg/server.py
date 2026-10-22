"""FastAPI app: OpenAI-compatible chat endpoint (D-006), the endpoints the WebUI calls, and the built UI."""

from __future__ import annotations

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

from .config import Config
from .llm import LLMClient, UpstreamError

log = logging.getLogger(__name__)

_FIELDS = ("reasoning_content", "content")


def create_app(cfg: Config, llm: LLMClient | None = None) -> FastAPI:
    llm = llm or LLMClient(cfg.model_backend)
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

        # The server owns policy: configured system prompt, no client system messages,
        # history reduced to role + content, client sampling and model ignored.
        upstream = [{"role": "system", "content": cfg.reasoning_policy.system_prompt}] + [
            {"role": m["role"], "content": m.get("content") or ""}
            for m in messages
            if isinstance(m, dict) and m.get("role") in ("user", "assistant")
        ]

        stream = llm.stream_chat(upstream)
        try:
            first = await anext(stream, None)
        except UpstreamError as e:
            status = e.status if e.status and 400 <= e.status < 500 else 502
            return _error(status, e.message, "upstream_error")

        cid, created = f"chatcmpl-{uuid.uuid4()}", int(time.time())
        if body.get("stream"):
            return StreamingResponse(
                _sse(cid, created, model_id, first, stream),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )
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
    """Yield (delta, finish_reason) from upstream chunks, keeping only content and reasoning."""
    try:
        chunk = first
        while chunk is not None:
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            yield {k: delta[k] for k in _FIELDS if delta.get(k)}, choice.get("finish_reason")
            chunk = await anext(stream, None)
    finally:
        await stream.aclose()


async def _sse(cid: str, created: int, model: str, first: dict | None, stream: AsyncIterator[dict]):
    def event(delta: dict, finish: str | None = None) -> str:
        chunk = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    finish = None
    try:
        async for delta, reason in _deltas(first, stream):
            finish = reason or finish
            if delta:
                yield event(delta)
    except (UpstreamError, httpx.HTTPError) as e:
        # Status 200 is already sent; ending without [DONE] tells the client the stream broke.
        log.error("backend failed mid-stream: %s", e)
        return
    yield event({}, finish or "stop")
    yield "data: [DONE]\n\n"


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
