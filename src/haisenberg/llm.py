"""The one OpenAI-compatible chat client (D-003). A backend is only config: base URL, key and model."""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .config import ModelBackendConfig


RETRY_S = 2.0  # pause between attempts while the backend is unreachable or loading
PER_MESSAGE = 8  # tokens a chat template adds per message (role markers), for the estimate below


def estimate_tokens(messages: list[dict[str, Any]], chars_per_token: float, tools: list | None = None) -> int:
    """Prompt tokens of a chat request, estimated from characters (no tokenizer: any backend works)."""
    chars = len(json.dumps(tools, ensure_ascii=False)) if tools else 0
    for m in messages:
        c = m.get("content") or ""
        chars += len(c) if isinstance(c, str) else sum(len(p.get("text", "")) for p in c if isinstance(p, dict))
        for call in m.get("tool_calls") or []:
            fn = call.get("function") or {}
            chars += len(fn.get("name") or "") + len(fn.get("arguments") or "")
    return math.ceil(chars / chars_per_token) + PER_MESSAGE * len(messages)


class UpstreamError(Exception):
    """The backend could not be reached, timed out before answering, or answered non-2xx."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.message = message
        self.status = status


class LLMClient:
    def __init__(self, cfg: ModelBackendConfig, http: httpx.AsyncClient | None = None):
        self.cfg = cfg
        self._http = http or httpx.AsyncClient(timeout=httpx.Timeout(cfg.timeout_s, connect=10.0))

    async def aclose(self) -> None:
        await self._http.aclose()

    async def stream_chat(self, messages: list[dict[str, Any]], **fields: Any) -> AsyncIterator[dict[str, Any]]:
        """Yield parsed `chat.completion.chunk` dicts until `[DONE]`.

        `fields` (e.g. tools, tool_choice, sampling) are merged into the body last and win.
        Raises UpstreamError before the first chunk if the call fails; a refused connection or a 503
        (model loading, server restarting) is retried until `wait_s` has passed. Closing the
        iteration early closes the upstream connection.
        """
        cfg = self.cfg
        body = {
            "model": cfg.model_id,
            "messages": messages,
            "stream": True,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            **cfg.extra_body,
            **fields,
        }
        headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
        url = cfg.api_base.rstrip("/") + "/chat/completions"
        response = await self._send(lambda: self._http.build_request("POST", url, json=body, headers=headers))
        try:
            if not response.is_success:
                await response.aread()
                raise UpstreamError(_error_message(response), response.status_code)
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                if data:
                    yield json.loads(data)
        finally:
            await response.aclose()

    async def _send(self, build) -> httpx.Response:
        """Send until the backend answers with anything but 503, retrying refusals until `wait_s` is over.
        No response means nothing was generated, so sending again is safe."""
        api, deadline = self.cfg.api_base, time.monotonic() + self.cfg.wait_s
        while True:
            try:
                response = await self._http.send(build(), stream=True)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError) as e:
                if time.monotonic() >= deadline:  # RemoteProtocolError: e.g. a WSL relay with nothing behind it
                    raise UpstreamError(f"backend unreachable at {api}: {e}") from e
            except httpx.TimeoutException as e:  # only with a configured read limit (timeout_s)
                raise UpstreamError(f"backend at {api} timed out") from e
            except httpx.TransportError as e:
                raise UpstreamError(f"backend unreachable at {api}: {e}") from e
            else:
                if response.status_code != 503 or time.monotonic() >= deadline:
                    return response
                await response.aclose()
            await asyncio.sleep(min(RETRY_S, max(0.0, deadline - time.monotonic())))


def _error_message(response: httpx.Response) -> str:
    try:
        message = response.json()["error"]["message"]
        if isinstance(message, str) and message:
            return message
    except (ValueError, KeyError, TypeError):
        pass
    return response.text[:500] or f"backend answered HTTP {response.status_code}"
