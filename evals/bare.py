"""The bare arm: the model alone, as a user of the same backend gets it (evals.run --mode bare).

One request per item through the one client: the conversation as it is, the model's sampling and seed, and room
to think up to the backend's own cap and then answer. An optional neutral system line comes from the config
(`reasoning_policy.bare_system_prompt`, empty by default: no system message). No suite prompt, no tools, no
thinking budget, no retry, no check, no rewrite, no history trim. The loop of the agent is not used here on
purpose (only its draft marker and its room formula), so nothing of the suite can run by accident. A test compares
this request with the agent's when every layer is switched off.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from haisenberg.agent import DRAFT_MARK, unscaled_room
from haisenberg.config import Config
from haisenberg.llm import LLMClient

STAGE = "bare"


def bare_request(cfg: Config) -> dict[str, Any]:
    """What a bare call adds to the client's own body: its system line and its `max_tokens` (the room a suite
    reasoner call gets with a full thinking budget, so thinking up to the backend's cap never eats the answer)."""
    return {"system_prompt": cfg.reasoning_policy.bare_system_prompt,
            "max_tokens": unscaled_room(cfg.reasoning_policy.thinking_budget, cfg.model_backend.max_tokens)}


class Bare:
    """Runs a conversation as one plain model call; yields the items the agent yields, so the eval runner logs
    every arm the same way (reasoning deltas, the draft marker before the first answer token, the draft, the
    answer, an empty verification report, the finish reason)."""

    def __init__(self, cfg: Config, llm: LLMClient):
        self.llm = llm
        request = bare_request(cfg)
        self.system, self.max_tokens = request["system_prompt"], request["max_tokens"]

    async def run(self, history: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
        messages = [{"role": "system", "content": self.system}] if self.system.strip() else []
        reply, finish = "", None
        stream = self.llm.stream_chat([*messages, *history], stage=STAGE, max_tokens=self.max_tokens)
        try:
            async for chunk in stream:
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                if text := delta.get("reasoning_content"):
                    yield {"reasoning_content": text}
                if text := delta.get("content"):
                    if not reply:
                        yield {"reasoning_content": DRAFT_MARK}
                    reply += text
                    yield {"reasoning_content": text}
                finish = choice.get("finish_reason") or finish
        finally:
            await stream.aclose()
        answer = reply.strip()
        yield {"draft": answer}
        yield {"content": answer}
        yield {"verification": {}}
        yield {"finish_reason": finish or "stop"}
