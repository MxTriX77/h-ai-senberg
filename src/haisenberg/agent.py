"""The orchestrator's tool loop (ReAct): look facts up with the generated tools, then answer with sources.

`Agent.run(history)` yields `{"reasoning_content": ...}` deltas (thinking, tool steps, the draft, the
verification) while it works, a `{"draft": ...}` and, after the answer as one `{"content": ...}`, a
`{"verification": report}` for in-process callers, then `{"finish_reason": ...}`. Only standard OpenAI chat fields are used (D-003),
so any backend with tool support works. As a safety net, tool calls a chat template wrote as text are read with the
configured patterns (`reasoning_policy.tool_call_markup`), and an empty or tool-call-only answer is asked for again once.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from .config import Config, ThinkingBudgetConfig, ToolCallMarkup
from .llm import LLMClient, estimate_tokens
from .tools import CALCULATE, Knowledge, Sources
from .verifier import Verification, Verifier, read_numbers, strip_cites

DRAFT_MARK = "\n\nDraft answer:\n"
CUT_NOTE = "\n[... shortened: context budget]"
NO_ANSWER = "**The answer could not be completed:** the model returned no answer. Please send the question again."


def _question(history: list[dict[str, Any]]) -> str:
    """The last user message as text (content parts joined)."""
    for m in reversed(history):
        if m["role"] == "user":
            c = m["content"]
            return c if isinstance(c, str) else " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


def thinking_budget(question: str, tb: ThinkingBudgetConfig) -> int:
    """Thinking tokens for the reasoner calls of one answer: min_tokens + (max_tokens - min_tokens) x share, where
    the share (at most 1) adds up each signal's weight x level: the question's characters, the numbers written
    in it and its enumerated parts or requested answer lines."""
    counts = ((tb.chars, len(question)), (tb.numbers, len(read_numbers(question))),
              (tb.items, len(re.findall(tb.item_pattern, question))))
    share = min(1.0, sum(signal.weight * signal.level(n) for signal, n in counts))
    return round(tb.min_tokens + (tb.max_tokens - tb.min_tokens) * share)


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


class _Markup:
    """Tool calls the model wrote as text, read with the patterns of reasoning_policy.tool_call_markup."""

    def __init__(self, cfg: ToolCallMarkup):
        self.call, self.function, self.parameter = (re.compile(p, re.S) for p in (cfg.call, cfg.function, cfg.parameter))

    def split(self, text: str) -> tuple[list[dict[str, Any]], str]:
        """(one tool call per call written in the text, the text without them); an unreadable call has no name."""
        calls = []
        for i, m in enumerate(self.call.finditer(text)):
            name, args = self._read(m.group(0))
            calls.append({"id": f"call_{i}", "type": "function",
                          "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
        return calls, self.call.sub("", text).strip()

    def _read(self, call: str) -> tuple[str, dict[str, Any]]:
        if f := self.function.search(call):
            return f.group(1), dict(self.parameter.findall(call, f.end()))
        try:  # JSON {"name": ..., "arguments": {...}}
            data = json.loads(call[call.find("{"):call.rfind("}") + 1])
            name, args = data["name"], data.get("arguments", {})
            args = json.loads(args) if isinstance(args, str) else args
        except (ValueError, TypeError, KeyError, AttributeError):
            return "", {}
        return (name, args) if isinstance(name, str) and isinstance(args, dict) else ("", {})


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
        self.markup = _Markup(self.policy.tool_call_markup) if self.policy.text_tool_calls else None

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
        tools = self.knowledge.tools
        system = self.policy.system_prompt
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
        sources = Sources()
        tb = self.policy.thinking_budget  # one budget for every reasoner round; the retry and the checks get none
        thinking = {tb.field: thinking_budget(_question(history), tb)} if tb.field else {}
        calc_rounds = self.policy.max_calc_rounds
        calc_only = [t for t in tools if t["function"]["name"] == CALCULATE] if calc_rounds else []
        rounds, calcs, room_left, ask_again = 0, 0, True, False
        while True:
            lookups_left = rounds < self.policy.max_search_depth
            calcs_left = bool(calc_only) and calcs < calc_rounds
            may_call = bool(tools) and room_left and (lookups_left or calcs_left)
            offered = tools if lookups_left or not calc_only else calc_only  # lookups used up: the calculator alone
            fields: dict[str, Any] = {"tools": offered} if tools else {}
            if tools and not may_call:
                fields["tool_choice"] = "none"
            if (prompt := self._estimate(messages)) > self.budget:  # only by the few chars of a cut note
                fields["max_tokens"] = self.context - prompt
            reply = _Reply()
            call = self._call(messages, {**fields, **thinking}, reply)
            try:
                async for delta in call:
                    yield delta
            finally:
                await call.aclose()
            calls = reply.tool_calls() if may_call else []
            if not calls and self.markup:
                written, rest = self.markup.split(reply.content)
                if written:  # the template's tool-call text in the reply: run it if allowed, else drop it and ask again
                    names = {t["function"]["name"] for t in offered}
                    run = may_call and all(c["function"]["name"] in names for c in written)
                    note = "run" if run else "removed, final answer asked again"
                    yield {"reasoning_content": f"\n\n> tool call written as text: {note}\n"}
                    calls, reply.content, ask_again = (written if run else []), rest, not run
            if not calls:
                break
            if calcs_left and (not lookups_left or all(c["function"]["name"] == CALCULATE for c in calls)):
                calcs += 1
            else:
                rounds += 1
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
        if thinking:  # after the rounds, so the first reasoning delta is still the model's own
            yield {"reasoning_content": f"\n\n> thinking budget: {thinking[tb.field]} tokens\n"}
        text = reply.content.strip()
        if not text and self.policy.empty_answer_retry and not ask_again:
            yield {"reasoning_content": "\n\n> empty answer: final answer asked again\n"}
            ask_again = True
        if ask_again:  # once, without tools, from the conversation so far
            ask = [*messages, {"role": "user", "content": self.policy.answer_retry_prompt}]
            fields = {**fields, **({"tool_choice": "none"} if tools else {}), **self.policy.answer_retry_request}
            if (prompt := self._estimate(ask)) > self.budget:
                room = self.context - prompt
                fields["max_tokens"] = min(fields.get("max_tokens", room), room)
            retry = _Reply()
            call = self._call(ask, fields, retry)
            try:
                async for delta in call:
                    yield delta
            finally:
                await call.aclose()
            again = self.markup.split(retry.content)[1] if self.markup else retry.content.strip()
            if again:
                text, reply = again, retry
            else:
                yield {"reasoning_content": "\n\n> no answer on the retry\n"}
        answer, draft = text or NO_ANSWER, text
        yield {"draft": draft}  # for in-process callers (evals); the server forwards only text fields
        report: dict[str, Any] = {}
        if self.verifier and draft:
            check = self.verifier.verify(_question(history), draft, sources)
            try:
                async for item in check:
                    if isinstance(item, Verification):
                        answer, report = item.answer, item.report()
                    else:
                        yield item
            finally:
                await check.aclose()
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
