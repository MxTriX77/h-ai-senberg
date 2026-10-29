"""The orchestrator's tool loop (ReAct): look facts up with the generated tools, then answer with sources.

`Agent.run(history)` yields `{"reasoning_content": ...}` deltas (thinking, tool steps, the draft, the
verification) while it works, a `{"draft": ...}` and, after the answer as one `{"content": ...}`, a
`{"verification": report}` for in-process callers, then `{"finish_reason": ...}`. Only standard OpenAI chat fields are used (D-003),
so any backend with tool support works. As a safety net, tool calls a chat template wrote as text are read with the
configured patterns (`reasoning_policy.tool_call_markup`), and an empty or tool-call-only answer is asked for again once
(written calculations are run first); an answer cut off where a call was removed gives way to the most complete one.
Answer lines the task spells out (`A9: <verdict>`) that the draft leaves out are asked for once and appended, and
lines that report on an item of the task (`A4: <verdict for C1>`) are rewritten once from the answer's own analysis.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import Config, ToolCallMarkup
from .llm import LLMClient, UpstreamError, estimate_tokens
from .tools import CALCULATE, Knowledge, Sources
from .verifier import (Verification, Verifier, anchor_lines, answer_labels, item_lines, missing_lines, strip_cites,
                       supplied_lines)

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

    def cut(self, text: str) -> bool:
        """The text ends in a written call: without it, it ends where the call was."""
        ends = [m.end() for m in self.call.finditer(text)]
        return bool(ends) and not text[ends[-1]:].strip()

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

    def _calculations(self, written: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The calculate calls among calls written as text that cannot run as a round: run anyway (plain code)."""
        run = self.policy.run_written_calculations and self.knowledge.calculator
        return [c for c in written if c["function"]["name"] == CALCULATE] if run else []

    def _run_tool(self, messages: list[dict[str, Any]], call: dict[str, Any], sources: Sources) -> tuple[str, bool]:
        """Run one tool call and append its result, shortened to fit the context: (the result lines for the
        reasoning, whether the result fitted)."""
        fn = call["function"]
        result, lines, _ = self.knowledge.run(fn["name"], fn["arguments"], sources)
        shown = "".join(f"- {line}\n" for line in lines) + "\n"
        message = {"role": "tool", "tool_call_id": call["id"], "content": ""}
        room = int((self.budget - self._estimate([*messages, message])) * self.chars_per_token)
        fits = len(result) <= room
        if not fits:
            result = result[: max(0, room - len(CUT_NOTE))] + CUT_NOTE
            shown += "- (result shortened to fit the context; no more lookups)\n\n"
        messages.append({**message, "content": result})
        return shown, fits

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
        asked = self._fit_history(messages[0], history)
        messages += asked
        if (need := self._estimate(messages)) > self.budget:
            yield {"content": f"The question is too long for the model's context (about {need} tokens; at most"
                              f" {self.budget} fit). Please shorten it."}
            yield {"verification": {}}
            yield {"finish_reason": "stop"}
            return
        sources = Sources()
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
            call = self._call(messages, fields, reply)
            try:
                async for delta in call:
                    yield delta
            finally:
                await call.aclose()
            calls = reply.tool_calls() if may_call else []
            cut = final = False
            if not calls and self.markup:
                written, rest = self.markup.split(reply.content)
                if written:  # the template's tool-call text in the reply: run it if allowed, else drop it and ask again
                    names = {t["function"]["name"] for t in offered}
                    run = may_call and all(c["function"]["name"] in names for c in written)
                    calc = [] if run else self._calculations(written)  # run all the same, then ask again
                    note = "run" if run else f"{_fate(written, calc)}, final answer asked again"
                    yield {"reasoning_content": f"\n\n> tool call written as text: {note}\n"}
                    cut, final = self.markup.cut(reply.content), bool(calc)
                    calls, reply.content, ask_again = (written if run else calc), rest, not run
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
                shown, fits = self._run_tool(messages, c, sources)
                room_left = room_left and fits  # a shortened result: no further tool rounds after this one
                yield {"reasoning_content": shown}
            if final:
                break
        text = reply.content.strip()
        if not text and self.policy.empty_answer_retry and not ask_again:
            yield {"reasoning_content": "\n\n> empty answer: final answer asked again\n"}
            ask_again = True
        if ask_again:  # without tools, from the conversation so far; once more after a calculation written in it
            ask = list(messages)
            fields = {**fields, **({"tool_choice": "none"} if tools else {}), **self.policy.answer_retry_request}
            tried = [("the draft", text, cut, reply)]
            for name in ("the retry", "the answer after its calculation"):
                ask.append({"role": "user", "content": self.policy.answer_retry_prompt})
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
                written, again = self.markup.split(retry.content) if self.markup else ([], retry.content.strip())
                calc = self._calculations(written) if len(tried) == 1 else []  # one more call at most
                if written:  # the retry's written calls are never dropped silently
                    note = _fate(written, calc) + (", final answer asked once more" if calc else "")
                    yield {"reasoning_content": f"\n\n> tool call written as text in the retry: {note}\n"}
                tried.append((name, again, bool(written) and self.markup.cut(retry.content), retry))
                if not calc:
                    break
                ask.append({"role": "assistant", "content": again or None, "tool_calls": calc})
                for c in calc:
                    fn = c["function"]
                    yield {"reasoning_content": f"\n\n> {fn['name']}({_show_args(fn['arguments'])})\n"}
                    yield {"reasoning_content": self._run_tool(ask, c, sources)[0]}
            name, text, cut, reply = tried[-1]
            if not text:
                yield {"reasoning_content": "\n\n> no answer on the retry\n"}
            if cut or not text:  # never a fragment that ends where a call was removed: the most complete answer
                question = _question(history)
                if ranked := [((_requested_lines(t[1], question), not t[2], len(t[1])), t) for t in tried if t[1]]:
                    (labelled, complete, chars), (name, text, cut, reply) = max(ranked, key=lambda r: r[0])
                    state = "complete" if complete else "ends at a removed tool call"
                    yield {"reasoning_content": f"\n\n> answer kept: {name} ({labelled} requested answer lines, {state},"
                                                f" {chars} characters)\n"}
        if text and self.policy.coverage_repair:  # requested answer lines the draft left out, before the checks
            cover = self._cover(asked, _question(history), text)
            try:
                async for item in cover:
                    if isinstance(item, str):
                        text = item
                    else:
                        yield item
            finally:
                await cover.aclose()
        if text and self.policy.line_check:  # lines that report on an item of the task, after the coverage repair
            check = self._check_lines(_question(history), text)
            try:
                async for item in check:
                    if isinstance(item, str):
                        text = item
                    else:
                        yield item
            finally:
                await check.aclose()
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

    async def _cover(self, asked: list[dict[str, Any]], question: str, text: str):
        """Answer lines the task asks for that the draft leaves out: asked for in one call without tools, and the
        reply's lines with exactly those labels appended. Yields reasoning deltas, then the new text (if any)."""
        missing = missing_lines(question, text)
        if not missing:
            return
        yield {"reasoning_content": f"\n\n> answer lines missing: {', '.join(missing)}; asked for them\n"}
        prompt = self.policy.coverage_repair_prompt.replace("{lines}", "\n".join(missing.values()))
        ask = [{"role": "system", "content": self.policy.system_prompt}, *asked,
               {"role": "assistant", "content": text}, {"role": "user", "content": prompt}]
        if estimate_tokens(ask, self.chars_per_token) > self.budget:
            yield {"reasoning_content": "\n\n> answer line request too long for the context: answer kept\n"}
            return
        reply = _Reply()
        call = self._call(ask, dict(self.policy.answer_retry_request), reply)
        try:
            async for delta in call:
                yield delta
        except (UpstreamError, httpx.HTTPError) as e:
            yield {"reasoning_content": f"\n\n> answer line request failed ({e}): answer kept\n"}
            return
        finally:
            await call.aclose()
        lines = supplied_lines(reply.content, missing)
        if not lines:
            yield {"reasoning_content": "\n\n> no requested answer line in the reply: answer kept\n"}
            return
        added = [label for label in missing if label in lines]
        yield {"reasoning_content": f"\n\n> answer lines added: {', '.join(added)}\n"}
        yield text.rstrip() + "\n" + "\n".join(lines[label] for label in added)

    async def _check_lines(self, question: str, text: str):
        """Answer lines whose placeholder names an item of the task (item_lines: `A4: <verdict for C1>`): one call
        without tools with the answer and the table label -> item, asking for each line what the answer's analysis
        of its item concluded; anchor_lines decides which lines change. Yields reasoning deltas, then the new text."""
        table = item_lines(question)
        if not table:
            return
        rows = [f"{label} → {item}" for label, item in table.items()]
        yield {"reasoning_content": f"\n\n> line check: {', '.join(rows)}\n"}
        prompt = self.policy.line_check_prompt.replace("{table}", "\n".join(rows)).replace("{answer}", text)
        ask = [{"role": "user", "content": prompt}]
        if estimate_tokens(ask, self.chars_per_token) > self.budget:
            yield {"reasoning_content": "\n\n> line check too long for the context: lines kept\n"}
            return
        reply = _Reply()
        call = self._call(ask, dict(self.policy.answer_retry_request), reply)
        try:
            async for delta in call:
                yield delta
        except (UpstreamError, httpx.HTTPError) as e:
            yield {"reasoning_content": f"\n\n> line check failed ({e}): lines kept\n"}
            return
        finally:
            await call.aclose()
        new, changes = anchor_lines(question, text, reply.content, table)
        yield {"reasoning_content": "\n\n" + "".join(f"> line check: {c}\n" for c in changes or ["no line changed"])}
        if changes:
            yield new

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


def _fate(written: list[dict[str, Any]], calc: list[dict[str, Any]]) -> str:
    """What became of calls written as text that could not run as a tool round."""
    if not calc:
        return "removed"
    return "calculation run" + (", other calls removed" if len(calc) < len(written) else "")


def _requested_lines(text: str, question: str) -> int:
    """How many labelled answer lines of text (`A1: …`) carry a label the question names."""
    return sum(bool(re.search(rf"(?<!\w){re.escape(label)}(?!\w)", question)) for label in answer_labels(text))


def _show_args(arguments: str) -> str:
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return arguments
    if not isinstance(args, dict):
        return arguments
    return ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items())
