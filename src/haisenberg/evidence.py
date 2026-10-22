"""Evidence protocol for analytic requests (D-032). One ledger call lists the request's items with evidence for
and against (with provenance), assumptions, gaps, dependencies and a status; code enforces the status rules and
the coverage of every listed item; the final answer is written from the ledger and reviewed against it.
Nothing here names a subject: prompts and trigger words live in config (`evidence`)."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from .config import Config
from .llm import LLMClient, UpstreamError, estimate_tokens
from .router import request_items
from .verifier import _say, apply_review, complete, fix_arithmetic, run_checkers

log = logging.getLogger(__name__)

ESTABLISHED, CONTRADICTED, UNVERIFIED, HYPOTHESIS = "established", "contradicted", "unverified", "hypothesis"
STATUSES = (ESTABLISHED, CONTRADICTED, UNVERIFIED, HYPOTHESIS)
HARD = ("given", "source", "computation")  # provenance that can settle an item
PROVENANCE = (*HARD, "knowledge", "assumption")


@dataclass
class Entry:
    id: str
    statement: str = ""
    support: list[dict[str, str]] = field(default_factory=list)  # {"evidence", "from"}
    against: list[dict[str, str]] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    status: str = UNVERIFIED
    claimed: str = UNVERIFIED  # the model's status, before the rules
    strong_enough: bool = False  # the evidence supports the statement at its full strength
    conclusion: str = ""
    rule: str = ""  # why the rules changed the status


@dataclass
class Ledger:
    answer: str  # written from the ledger, then fixed in code and by the review
    entries: list[Entry]
    skipped: dict[str, str]  # listed items that are only given data or instructions: id -> why
    missing: list[str]  # listed items still without an entry after the repair round
    fixes: list[dict[str, str]]  # applied review fixes: old, new, why

    def report(self) -> dict[str, Any]:
        return {"items": [asdict(e) for e in self.entries], "skipped": self.skipped, "missing": self.missing,
                "fixes": self.fixes}


def norm_id(value: Any) -> str:
    """"Claim A", "A)", "a" -> "A"."""
    words = re.sub(r"^\W+|\W+$", "", str(value)).split()
    return words[-1].upper() if words else ""


def parse_ledger(text: str) -> tuple[list[Entry], dict[str, str]] | None:
    """Entries and skipped items from the model's JSON (code fences or stray text around it are fine);
    None when there is no readable ledger."""
    start, end = text.find("{"), text.rfind("}")
    try:
        data = json.loads(text[start : end + 1]) if start >= 0 else None
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        return None
    entries = [e for raw in data["items"] if (e := _entry(raw))]
    skipped = {norm_id(s["id"]): str(s.get("why") or "") for s in data.get("skipped") or []
               if isinstance(s, dict) and norm_id(s.get("id", ""))}
    return entries, skipped


def _entry(raw: Any) -> Entry | None:
    if not isinstance(raw, dict) or not (i := norm_id(raw.get("id", ""))):
        return None
    status = str(raw.get("status") or "").strip().lower()
    status = status if status in STATUSES else UNVERIFIED
    return Entry(
        id=i, statement=str(raw.get("statement") or ""), support=_evidence(raw.get("for")),
        against=_evidence(raw.get("against")), assumptions=_strings(raw.get("assumptions")),
        gaps=_strings(raw.get("gaps")), depends_on=[norm_id(d) for d in _strings(raw.get("depends_on"))],
        status=status, claimed=status, strong_enough=raw.get("strong_enough") is True,
        conclusion=str(raw.get("conclusion") or ""),
    )


def _evidence(value: Any) -> list[dict[str, str]]:
    out = []
    for v in value if isinstance(value, list) else []:
        text, source = (v.get("evidence"), str(v.get("from") or "").lower()) if isinstance(v, dict) else (v, "")
        if text:
            out.append({"evidence": str(text), "from": source if source in PROVENANCE else "knowledge"})
    return out


def _strings(value: Any) -> list[str]:
    return [str(v) for v in value if v] if isinstance(value, list) else []


def strong_word(text: str, words: list[str]) -> str | None:
    """The first trigger word of a strong claim ("always", "guarantees", ...) in the text, matched at a word start."""
    low = text.lower()
    return next((w for w in words if re.search(rf"(?<!\w){re.escape(w.lower())}", low)), None)


def apply_rules(entries: list[Entry], strong_words: list[str]) -> None:
    """Settle each status by the evidence rules (in code, whatever the model said):
    - evidence against from a computation, the given data or a source -> contradicted;
    - contradicted with nothing against it -> unverified (absence of evidence is not disproof);
    - established with nothing for it -> unverified; resting only on assumptions -> hypothesis;
    - established strong claim ("always", ...) without evidence for its full strength -> unverified;
    - established but built on an item that is not established -> unverified.
    The rules only lower a status (a hypothesis is never promoted)."""
    for e in entries:
        hard = next((a["from"] for a in e.against if a["from"] in HARD), None)
        if hard:
            _set(e, CONTRADICTED, f"evidence against it from {hard}")
        elif e.status == CONTRADICTED and not e.against:
            _set(e, UNVERIFIED, "nothing against it: absence of evidence is not disproof")
        elif e.status == ESTABLISHED:
            if not e.support:
                _set(e, UNVERIFIED, "no evidence for it")
            elif all(s["from"] == "assumption" for s in e.support):
                _set(e, HYPOTHESIS, "it rests only on assumptions")
            elif (word := strong_word(e.statement, strong_words)) and not e.strong_enough:
                _set(e, UNVERIFIED, f"a strong claim ({word!r}) without evidence for its full strength")
    by_id = {e.id: e for e in entries}
    changed = True
    while changed:  # only lowers statuses, so it ends
        changed = False
        for e in entries:
            weak = [d for d in e.depends_on if d != e.id and d in by_id and by_id[d].status != ESTABLISHED]
            if e.status == ESTABLISHED and weak:
                _set(e, UNVERIFIED, f"it depends on {weak[0]}, which is {by_id[weak[0]].status}")
                changed = True


def _set(e: Entry, status: str, why: str) -> None:
    if e.status != status:
        e.rule, e.status, e.conclusion = f"{e.status} -> {status}: {why}", status, ""  # the conclusion was for the old status


def missing(entries: list[Entry], skipped: dict[str, str], listed: list[str]) -> list[str]:
    """Listed item ids with no entry and not skipped."""
    have = {e.id for e in entries} | set(skipped)
    return [i for i in listed if i not in have]


def render(entries: list[Entry], skipped: dict[str, str]) -> str:
    """The settled ledger as text, for the writer, the review and the reasoning block."""
    out = []
    for e in entries:
        out.append(f"{e.id}: {e.statement}")
        out.append(f"  status: {e.status}" + (f" (changed by the rules: {e.rule})" if e.rule else ""))
        out += [f"  for ({s['from']}): {s['evidence']}" for s in e.support]
        out += [f"  against ({a['from']}): {a['evidence']}" for a in e.against]
        if e.assumptions:
            out.append("  assumptions: " + "; ".join(e.assumptions))
        if e.gaps:
            out.append("  gaps: " + "; ".join(e.gaps))
        if e.depends_on:
            out.append("  depends on: " + ", ".join(e.depends_on))
        if e.conclusion:
            out.append(f"  conclusion: {e.conclusion}")
    out += [f"{i}: skipped ({why})" for i, why in skipped.items()]
    return "\n".join(out)


class EvidenceProtocol:
    def __init__(self, cfg: Config, llm: LLMClient):
        self.cfg = cfg.evidence
        self.verifier = cfg.verifier  # arithmetic and checker settings, reused on the written answer
        self.llm = llm
        self.context = cfg.context_tokens
        self.max_tokens = cfg.model_backend.max_tokens
        self.chars_per_token = cfg.model_backend.chars_per_token

    def _budget(self, request: dict[str, Any]) -> int:
        return self.context - (request.get("max_tokens") or self.max_tokens)

    async def _ask(self, messages: list[dict[str, Any]], request: dict[str, Any]) -> tuple[str | None, str]:
        """(reply, problem): the reply text, or None and why there is none."""
        if estimate_tokens(messages, self.chars_per_token) > self._budget(request):
            return None, "too long for the context"
        try:
            return await complete(self.llm, messages, request), ""
        except (UpstreamError, httpx.HTTPError) as e:
            log.error("evidence protocol call failed: %s", e)
            return None, f"failed: {e}"

    async def run(self, question: str, draft: str) -> AsyncIterator[dict | Ledger]:
        """Yield reasoning lines, then the Ledger. Without a readable ledger it yields no Ledger (the caller
        falls back to the consistency review)."""
        listed = request_items(question)
        items = "".join(f"{i}: {t}\n" for i, t in listed)
        user = f"Task:\n{question}\n\n" + (f"Items listed in the task:\n{items}\n" if items else "") + f"Draft:\n{draft}"
        messages = [{"role": "system", "content": self.cfg.ledger_prompt}, {"role": "user", "content": user}]
        yield _say("\n\nEvidence ledger\n")
        reply, problem = await self._ask(messages, self.cfg.ledger_request)
        parsed = parse_ledger(reply) if reply is not None else None
        if parsed is None:
            yield _say(f"- no readable ledger ({problem or 'not valid JSON'})\n")
            return
        entries, skipped = parsed
        if gone := missing(entries, skipped, [i for i, _ in listed]):
            yield _say(f"- no entry for {', '.join(gone)}; asking again\n")
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": self.cfg.repair_prompt.replace("{missing}", ", ".join(gone))}]
            more, _ = await self._ask(messages, self.cfg.ledger_request)
            if more and (extra := parse_ledger(more)):
                entries += [e for e in extra[0] if e.id in gone]
                skipped |= {i: why for i, why in extra[1].items() if i in gone}
            gone = missing(entries, skipped, gone)
            if gone:
                yield _say(f"- still no entry for {', '.join(gone)}\n")
        apply_rules(entries, self.cfg.strong_words)
        ledger = render(entries, skipped)
        yield _say(ledger + "\n")

        yield _say("\n\nWriting the answer from the ledger\n")
        messages = [{"role": "system", "content": self.cfg.writer_prompt},
                    {"role": "user", "content": f"Task:\n{question}\n\nLedger:\n{ledger}\n\nDraft:\n{draft}"}]
        written, problem = await self._ask(messages, self.cfg.request)
        answer = (written or "").strip()
        if not answer:
            yield _say(f"- no answer written ({problem or 'empty reply'}); reviewing the draft instead\n")
            answer = draft
        fixed: list[dict[str, str]] = []
        if self.verifier.arithmetic_check:
            answer, fixed, _ = fix_arithmetic(answer, self.verifier.arithmetic_rel_tol)
        answer, checks = run_checkers(answer, self.verifier.checkers)
        lines = [f"- arithmetic: {a['expression']} = {a['stated']} → {a['fixed']}\n" for a in fixed]
        lines += [f"- {c['checker']}: {c['text']!r} → {c['fixed']!r}\n" for c in checks if c["fixed"] is not None]
        if lines:
            yield _say("".join(lines))

        messages = [{"role": "system", "content": self.cfg.review_prompt},
                    {"role": "user", "content": f"Task:\n{question}\n\nLedger:\n{ledger}\n\nAnswer:\n{answer}"}]
        fixes: list[dict[str, str]] = []
        review = apply_review(self.llm, self.cfg.request, self._budget(self.cfg.request), self.chars_per_token,
                              messages, answer, "Ledger review")
        try:
            async for item in review:
                if isinstance(item, tuple):
                    answer, fixes = item
                else:
                    yield item
        finally:
            await review.aclose()
        yield Ledger(answer, entries, skipped, gone, fixes)
