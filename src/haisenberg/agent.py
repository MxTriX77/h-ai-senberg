"""The orchestrator's tool loop (ReAct): look facts up with the generated tools, then answer with sources.

`Agent.run(history)` yields `{"reasoning_content": ...}` deltas (thinking, tool steps, the draft, the
verification) while it works, then the answer as one `{"content": ...}`, a `{"verification": report}`
for in-process callers, then `{"finish_reason": ...}`. Only standard OpenAI chat fields are used (D-003),
so any backend with tool support works.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .evidence import EvidenceProtocol, Ledger
from .llm import LLMClient, estimate_tokens
from .router import ANALYTIC, CASUAL, route
from .tools import Knowledge, Sources
from .verifier import Verification, Verifier, strip_cites

DRAFT_MARK = "\n\nDraft answer:\n"
CUT_NOTE = "\n[... shortened: context budget]"


def _question(history: list[dict[str, Any]]) -> str:
    """The last user message as text (content parts joined)."""
    for m in reversed(history):
        if m["role"] == "user":
            c = m["content"]
            return c if isinstance(c, str) else " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


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
        verify = self.policy.enable_multi_hop_verification and knowledge.tools
        self.verifier = Verifier(cfg, llm, knowledge) if verify else None
        self.annotated = cfg.verifier.answer_style == "annotated"
        self.context = cfg.context_tokens
        self.budget = self.context - cfg.model_backend.max_tokens  # prompt tokens a reasoner call may use
        self.chars_per_token = cfg.model_backend.chars_per_token
        self.router = cfg.router
        self.protocol = EvidenceProtocol(cfg, llm) if cfg.evidence.enabled else None

    async def _casual(self, messages: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
        """Small talk (router): one call, no tools, no checks; the answer streams straight as content."""
        finish = None
        stream = self.llm.stream_chat(messages, **self.router.casual_request)
        try:
            async for chunk in stream:
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                if text := delta.get("reasoning_content"):
                    yield {"reasoning_content": text}
                if text := delta.get("content"):
                    yield {"content": text}
                finish = choice.get("finish_reason") or finish
        finally:
            await stream.aclose()
        yield {"verification": {}}
        yield {"finish_reason": finish or "stop"}

    def _estimate(self, messages: list[dict[str, Any]]) -> int:
        return estimate_tokens(messages, self.chars_per_token, self.knowledge.tools)

    def _fit_history(self, system: dict[str, Any], history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop the oldest turns until system + history use at most half of the prompt budget (the rest is
        for tool results); what remains starts with a user message, and the last message is always kept."""
        history = list(history)
        while len(history) > 1 and self._estimate([system, *history]) > self.budget // 2:
            history.pop(0)
            while len(history) > 1 and history[0]["role"] != "user":
                history.pop(0)
        return history

    async def run(self, history: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
        depth, why = route(_question(history), self.router)
        yield {"route": depth}  # for in-process callers (evals); the server forwards only text fields
        tools = [] if depth == CASUAL else self.knowledge.tools
        system = self.policy.casual_prompt if depth == CASUAL else self.policy.system_prompt
        if tools:
            system = f"{system.rstrip()}\n\n{self.policy.tool_prompt}"
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        messages += self._fit_history(messages[0], history)
        if (need := self._estimate(messages)) > self.budget:
            yield {"content": f"The question is too long for the model's context (about {need} tokens; at most"
                              f" {self.budget} fit). Please shorten it."}
            yield {"verification": {}}
            yield {"finish_reason": "stop"}
            return
        if depth == CASUAL:
            async for item in self._casual(messages):
                yield item
            return
        if self.router.enabled:
            yield {"reasoning_content": f"Route: {depth} ({why})\n"}
        sources = Sources()
        rounds, room_left = 0, True
        while True:
            may_call = bool(tools) and rounds < self.policy.max_search_depth and room_left
            fields: dict[str, Any] = {"tools": tools} if tools else {}
            if tools and not may_call:
                fields["tool_choice"] = "none"
            if (prompt := self._estimate(messages)) > self.budget:  # only by the few chars of a cut note
                fields["max_tokens"] = self.context - prompt
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
                result, lines, _ = self.knowledge.run(fn["name"], fn["arguments"], sources)
                yield {"reasoning_content": "".join(f"- {line}\n" for line in lines) + "\n"}
                message = {"role": "tool", "tool_call_id": c["id"], "content": ""}
                room = int((self.budget - self._estimate([*messages, message])) * self.chars_per_token)
                if len(result) > room:  # keep within the context; no further tool rounds after this one
                    result, room_left = result[: max(0, room - len(CUT_NOTE))] + CUT_NOTE, False
                    yield {"reasoning_content": "- (result shortened to fit the context; no more lookups)\n\n"}
                messages.append({**message, "content": result})
        answer = draft = reply.content.strip()
        question, report = _question(history), {}
        analytic = depth == ANALYTIC and self.protocol is not None and bool(draft)
        if self.verifier and draft:
            check = self.verifier.verify(question, draft, sources, review=not analytic)
            try:
                async for item in check:
                    if isinstance(item, Verification):
                        answer, report = item.answer, item.report()
                    else:
                        yield item
            finally:
                await check.aclose()
        if analytic:  # the evidence protocol replaces the consistency review (D-032)
            ledger, run = None, self.protocol.run(question, answer)
            try:
                async for item in run:
                    if isinstance(item, Ledger):
                        ledger = item
                    else:
                        yield item
            finally:
                await run.aclose()
            if ledger:
                answer, report["ledger"] = ledger.answer, ledger.report()
            elif self.verifier and self.verifier.cfg.consistency_check:  # no readable ledger: the usual review
                review = self.verifier.consistency_review(question, answer)
                try:
                    async for item in review:
                        if isinstance(item, tuple):
                            answer, report["consistency"] = item
                        else:
                            yield item
                finally:
                    await review.aclose()
        if self.annotated:
            yield {"content": answer + sources.source_list(answer)}
        else:  # a plain chat answer: the ids and the checks stay in the reasoning block
            yield {"content": strip_cites(answer)}
        yield {"verification": report}  # for in-process callers (evals); the server forwards only text fields
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
