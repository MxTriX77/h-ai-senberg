"""The one OpenAI-compatible chat client (D-003). A backend is only config: base URL, key and model."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .config import ModelBackendConfig


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
        Raises UpstreamError before the first chunk if the call fails; closing the
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
        request = self._http.build_request(
            "POST", cfg.api_base.rstrip("/") + "/chat/completions", json=body, headers=headers
        )
        try:
            response = await self._http.send(request, stream=True)
        except httpx.TimeoutException as e:
            raise UpstreamError(f"backend at {cfg.api_base} timed out") from e
        except httpx.TransportError as e:
            raise UpstreamError(f"backend unreachable at {cfg.api_base}: {e}") from e

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


def _error_message(response: httpx.Response) -> str:
    try:
        message = response.json()["error"]["message"]
        if isinstance(message, str) and message:
            return message
    except (ValueError, KeyError, TypeError):
        pass
    return response.text[:500] or f"backend answered HTTP {response.status_code}"
