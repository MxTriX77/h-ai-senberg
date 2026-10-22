"""The orchestrator's tool loop (ReAct): look facts up with the generated tools, then answer with sources.

`Agent.run(history)` yields `{"reasoning_content": ...}` deltas (thinking, tool steps, the draft) while
it works, then the answer as one `{"content": ...}`, then `{"finish_reason": ...}`. Only standard
OpenAI chat fields are used (D-003), so any backend with tool support works.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .llm import LLMClient
from .tools import Knowledge, Sources

DRAFT_MARK = "\n\nDraft answer:\n"


@dataclass
class _Call:
    id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class _Reply:
    content: str = ""
    calls: dict[int, _Call] = field(default_factory=dict)
    finish: str | None = None

    def add_fragment(self, frag: dict[str, Any]) -> None:
        call = self.calls.setdefault(frag.get("index", len(self.calls)), _Call())
        call.id = frag.get("id") or call.id
        fn = frag.get("function") or {}
        call.name += fn.get("name") or ""
        call.arguments += fn.get("arguments") or ""

    def tool_calls(self) -> list[dict[str, Any]]:
        return [
            {"id": c.id or f"call_{i}", "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for i, c in sorted(self.calls.items())
        ]


class Agent:
    def __init__(self, cfg: Config, llm: LLMClient, knowledge: Knowledge):
        self.policy = cfg.reasoning_policy
        self.llm = llm
        self.knowledge = knowledge

    async def run(self, history: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
        tools = self.knowledge.tools
        system = self.policy.system_prompt
        if tools:
            system = f"{system.rstrip()}\n\n{self.policy.tool_prompt}"
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}, *history]
        sources = Sources()
        rounds = 0
        while True:
            may_call = bool(tools) and rounds < self.policy.max_search_depth
            fields: dict[str, Any] = {"tools": tools} if tools else {}
            if tools and not may_call:
                fields["tool_choice"] = "none"
            reply = _Reply()
            call = self._call(messages, fields, reply)
            try:
                async for delta in call:
                    yield delta
            finally:
                await call.aclose()
            if not (may_call and reply.calls):
                break
            rounds += 1
            calls = reply.tool_calls()
            messages.append({"role": "assistant", "content": reply.content or None, "tool_calls": calls})
            for c in calls:
                fn = c["function"]
                yield {"reasoning_content": f"\n\n> {fn['name']}({_show_args(fn['arguments'])})\n"}
                result, lines = self.knowledge.run(fn["name"], fn["arguments"], sources)
                yield {"reasoning_content": "".join(f"- {line}\n" for line in lines) + "\n"}
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
        draft = reply.content.strip()
        yield {"content": draft + sources.source_list(draft)}
        yield {"finish_reason": reply.finish or "stop"}

    async def _call(self, messages: list[dict[str, Any]], fields: dict[str, Any], reply: _Reply):
        """Stream one model call: reasoning and the draft go out as reasoning; tool calls are collected."""
        stream = self.llm.stream_chat(messages, **fields)
        try:
            async for chunk in stream:
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                if text := delta.get("reasoning_content"):
                    yield {"reasoning_content": text}
                if text := delta.get("content"):
                    if not reply.content:
                        yield {"reasoning_content": DRAFT_MARK}
                    reply.content += text
                    yield {"reasoning_content": text}
                for frag in delta.get("tool_calls") or []:
                    reply.add_fragment(frag)
                reply.finish = choice.get("finish_reason") or reply.finish
        finally:
            await stream.aclose()


def _show_args(arguments: str) -> str:
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return arguments
    if not isinstance(args, dict):
        return arguments
    return ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items())
