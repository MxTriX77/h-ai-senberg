"""Evidence ledger (D-032, c9): verdicts that follow from the recorded evidence.

When the last user message lists at least `evidence.min_items` items to judge, one JSON call records per item
what it asserts, the evidence for and against it (with where each piece comes from), assumptions, gaps,
dependencies and a status. Rules in code then settle each status from that record, whatever the model said
(`settle`). When the rules change nothing and every listed item has an entry, the draft stays the answer;
otherwise one writer call writes the answer from the settled ledger and the draft. Without a readable ledger,
on an upstream error or an empty writer reply the draft stays, unchanged. Nothing here names a subject or holds
language data: prompts and trigger words live in config (`evidence`)."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import Config
from .llm import LLMClient, UpstreamError, estimate_tokens
from .verifier import _say, complete, fix_arithmetic, run_checkers

log = logging.getLogger(__name__)

ESTABLISHED, CONTRADICTED, UNVERIFIED, HYPOTHESIS = "established", "contradicted", "unverified", "hypothesis"
STATUSES = (ESTABLISHED, CONTRADICTED, UNVERIFIED, HYPOTHESIS)
PROOF = "proof"
ASSERTS = ("fact", "cause", "forecast", "computed", PROOF)
HARD = ("given", "source", "computation")  # where evidence can settle an item from
PROVENANCE = (*HARD, "knowledge", "assumption")

# A listed item: a line that starts with an id (`1.`, `(2)`, `a)`, `C1.`, `У3)`, `S4:`, also after a bullet,
# Markdown emphasis or one word: `Claim A:`), or a bulleted line without one (`- …`, `* …`, `• …`).
_ID = r"\d{1,2}|[^\W\d_]{1,2}\d{1,2}|[^\W\d_]"
_LISTED = re.compile(rf"^[ \t>]*(?:[-*•][ \t]+)?[*_]*(?:\((?P<p>{_ID})\)|(?:(?P<word>[^\W\d_]+)[ \t]+)?"
                     rf"(?P<id>{_ID})[*_]*[.):：])[*_]*[ \t]+(?P<text>\S.*)$", re.M)
_BULLET = re.compile(r"^[ \t>]*[-*•][ \t]+(?P<text>\S.*)$", re.M)
_LOOKALIKE = str.maketrans("АВСЕНКМОРТХавсенкмортх", "ABCEHKMOPTXabcehkmoptx")  # Cyrillic for Latin in ids


def listed_items(text: str) -> list[tuple[str, str]]:
    """(id as the message writes it, text) of every item the message lists, first occurrence of each id. Lines
    with an id win; without any, bulleted lines are numbered 1, 2, … in order. With a word before the id
    (`Claim A:`), the id has a digit or is one capital letter, so `Mass m: 5 g` is not an item."""
    found = []
    for m in _LISTED.finditer(text):
        iid = m["p"] or m["id"]
        if m["word"] and not (re.search(r"\d", iid) or (len(iid) == 1 and iid.isupper())):
            continue
        found.append((f"{m['word']} {iid}" if m["word"] else iid, m["text"].strip()))
    if not found:
        found = [(str(n), m["text"].strip()) for n, m in enumerate(_BULLET.finditer(text), 1)]
    seen, items = set(), []
    for iid, t in found:
        if (k := key(iid)) not in seen:
            seen.add(k)
            items.append((iid, t))
    return items


def key(value: Any) -> str:
    """An id for matching: its last word, case and Cyrillic look-alike letters ignored ("Claim A", "a)" → "a")."""
    words = re.findall(r"\w+", str(value))
    return words[-1].translate(_LOOKALIKE).casefold() if words else ""


@dataclass
class Entry:
    id: str
    statement: str = ""
    asserts: str = "fact"
    support: list[dict[str, str]] = field(default_factory=list)  # "for": {"evidence", "from"}
    against: list[dict[str, str]] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    strong_enough: bool = False  # the evidence for supports the statement at its full strength
    status: str = UNVERIFIED
    claimed: str = UNVERIFIED  # the model's status, before the rules
    reason: str = ""
    rule: str = ""  # the rule that changed the status (R-a … R-e), with why
    why: str = ""


def parse_ledger(text: str, listed: list[tuple[str, str]]) -> tuple[list[Entry], list[str]] | None:
    """(entries, skipped ids) from the model's JSON, code fences or text around it allowed; None when there is no
    readable ledger. Ids that match a listed item become that item's id as the message writes it."""
    start, end = text.find("{"), text.rfind("}")
    try:
        data = json.loads(text[start : end + 1]) if 0 <= start < end else None
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        return None
    ids = {key(i): i for i, _ in listed}
    entries, seen = [], set()
    for raw in data["items"]:
        if isinstance(raw, dict) and (e := _entry(raw, ids)) and key(e.id) not in seen:
            seen.add(key(e.id))
            entries.append(e)
    skipped = []
    for s in data.get("skipped") or []:
        if (i := _id(s.get("id") if isinstance(s, dict) else s, ids)) and key(i) not in seen:
            seen.add(key(i))
            skipped.append(i)
    return entries, skipped


def _id(value: Any, ids: dict[str, str]) -> str:
    """The listed id the value names (any of its words, last first), else the value itself, stripped."""
    words = re.findall(r"\w+", str(value or ""))
    return next((ids[k] for w in reversed(words) if (k := key(w)) in ids), " ".join(words))


def _entry(raw: dict[str, Any], ids: dict[str, str]) -> Entry | None:
    if not (i := _id(raw.get("id"), ids)):
        return None
    status = str(raw.get("status") or "").strip().lower()
    status = status if status in STATUSES else UNVERIFIED
    asserts = str(raw.get("asserts") or "").strip().lower()
    return Entry(
        id=i, statement=str(raw.get("statement") or ""), asserts=asserts if asserts in ASSERTS else "fact",
        support=_evidence(raw.get("for")), against=_evidence(raw.get("against")),
        assumptions=_strings(raw.get("assumptions")), gaps=_strings(raw.get("gaps")),
        depends_on=[_id(d, ids) for d in _strings(raw.get("depends_on"))],
        strong_enough=raw.get("strong_enough") is True, status=status, claimed=status,
        reason=str(raw.get("reason") or ""),
    )


def _evidence(value: Any) -> list[dict[str, str]]:
    out = []
    for v in value if isinstance(value, list) else []:
        text, source = (v.get("evidence"), str(v.get("from") or "").strip().lower()) if isinstance(v, dict) else (v, "")
        if text:
            out.append({"evidence": str(text), "from": source if source in PROVENANCE else "knowledge"})
    return out


def _strings(value: Any) -> list[str]:
    return [str(v) for v in value if v] if isinstance(value, list) else []


def strong_word(text: str, words: list[str]) -> str | None:
    """The first of the words that starts a word of the text (case ignored), or None."""
    low = text.lower()
    return next((w for w in words if re.search(rf"(?<!\w){re.escape(w.lower())}", low)), None)


def settle(entries: list[Entry], strong_words: list[str]) -> list[Entry]:
    """Settle each status from the recorded evidence, first rule that applies (R-a … R-d), then R-e to a fixed
    point; return the entries whose status changed. No rule makes anything established.
    - R-a: evidence against from the given material, a source or a computation → contradicted.
    - R-b: a claim of proof (`asserts: proof`, or a strong word in the statement) without evidence for its full
      strength: with evidence for it that cannot give what it claims → contradicted; with nothing for it and
      gaps saying material is missing → unverified.
    - R-c: contradicted with nothing against it → unverified (not shown is not false).
    - R-d: established with nothing for it → unverified; resting only on assumptions → hypothesis; without
      evidence for its full strength → unverified.
    - R-e: established but built on an item that is not established → unverified."""
    for e in entries:
        hard = next((a["from"] for a in e.against if a["from"] in HARD), None)
        proof = (e.asserts == PROOF or strong_word(e.statement, strong_words)) and not e.strong_enough
        if hard:
            _set(e, CONTRADICTED, "R-a", f"evidence against it from {_FROM[hard]}")
        elif proof and e.support:
            _set(e, CONTRADICTED, "R-b", "claim of proof the evidence can't give")
        elif proof and e.gaps:
            _set(e, UNVERIFIED, "R-b", "claim of proof; the material that would settle it is missing")
        elif e.status == CONTRADICTED and not e.against:
            _set(e, UNVERIFIED, "R-c", "nothing against it: not shown is not false")
        elif e.status == ESTABLISHED and not e.support:
            _set(e, UNVERIFIED, "R-d", "nothing for it")
        elif e.status == ESTABLISHED and all(s["from"] == "assumption" for s in e.support):
            _set(e, HYPOTHESIS, "R-d", "it rests only on assumptions")
        elif e.status == ESTABLISHED and not e.strong_enough:
            _set(e, UNVERIFIED, "R-d", "the evidence falls short of the statement's full strength")
    by_key = {key(e.id): e for e in entries}
    changed = True
    while changed:  # only lowers established, so it ends
        changed = False
        for e in entries:
            weak = [d for d in e.depends_on if (k := key(d)) != key(e.id) and k in by_key
                    and by_key[k].status != ESTABLISHED]
            if e.status == ESTABLISHED and weak:
                d = by_key[key(weak[0])]
                _set(e, UNVERIFIED, "R-e", f"it depends on {d.id}, which is {d.status}")
                changed = True
    return [e for e in entries if e.status != e.claimed]


_FROM = {"given": "the given material", "source": "a source", "computation": "a computation"}


def _set(e: Entry, status: str, rule: str, why: str) -> None:
    if e.status != status:
        e.status, e.rule, e.why = status, rule, why


def missing(entries: list[Entry], skipped: list[str], listed: list[tuple[str, str]]) -> list[str]:
    """Listed ids with neither an entry nor a skipped mark."""
    have = {key(e.id) for e in entries} | {key(s) for s in skipped}
    return [i for i, _ in listed if key(i) not in have]


def render(entries: list[Entry], skipped: list[str], gone: list[str]) -> str:
    """The settled ledger as text for the writer."""
    out = []
    for e in entries:
        out.append(f"{e.id}: {e.statement}")
        out.append(f"  status: {e.status}" + (f" (settled by the rules: {e.why}; the ledger first said {e.claimed})"
                                             if e.status != e.claimed else ""))
        out += [f"  for ({s['from']}): {s['evidence']}" for s in e.support]
        out += [f"  against ({a['from']}): {a['evidence']}" for a in e.against]
        if e.assumptions:
            out.append("  assumptions: " + "; ".join(e.assumptions))
        if e.gaps:
            out.append("  gaps: " + "; ".join(e.gaps))
        if e.reason and e.status == e.claimed:
            out.append(f"  reason: {e.reason}")
    if skipped:
        out.append("Skipped (data or instructions): " + ", ".join(skipped))
    if gone:
        out.append("Not yet judged: " + ", ".join(gone))
    return "\n".join(out)


@dataclass
class Settled:
    answer: str
    report: dict[str, Any]  # verification.ledger


def _report(entries: list[Entry], skipped: list[str], gone: list[str], written: bool, fallback: str = "") -> dict:
    return {"items": [{"id": e.id, "asserts": e.asserts, "claimed": e.claimed, "status": e.status, "rule": e.rule}
                      for e in entries],
            "skipped": skipped, "missing": gone, "written": written, "fallback": fallback}


def text_of(content: Any) -> str:
    """A message's content as text (content parts joined)."""
    return content if isinstance(content, str) else " ".join(
        p.get("text", "") for p in content or [] if isinstance(p, dict))


class Evidence:
    def __init__(self, cfg: Config, llm: LLMClient, checks: bool):
        self.cfg = cfg.evidence
        self.verifier = cfg.verifier  # arithmetic and checker settings, run on a written answer as on the draft
        self.checks = checks  # the draft was checked (the verifier is on)
        self.llm = llm
        self.context = cfg.context_tokens
        self.max_tokens = cfg.model_backend.max_tokens
        self.chars_per_token = cfg.model_backend.chars_per_token

    def triggered(self, question: str) -> list[tuple[str, str]]:
        """The items the last user message lists, when there are enough to run the ledger; else []."""
        items = listed_items(question)
        return items if len(items) >= self.cfg.min_items else []

    def _fits(self, messages: list[dict[str, Any]], request: dict[str, Any]) -> bool:
        return estimate_tokens(messages, self.chars_per_token) <= self.context - (
            request.get("max_tokens") or self.max_tokens)

    async def _ask(self, messages: list[dict[str, Any]], request: dict[str, Any]) -> tuple[str | None, str]:
        """(reply, problem): the reply text, or None and why there is none."""
        if not self._fits(messages, request):
            return None, "too long for the context"
        try:
            return await complete(self.llm, messages, request), ""
        except (UpstreamError, httpx.HTTPError) as e:
            log.error("evidence ledger call failed: %s", e)
            return None, f"failed: {e}"

    async def run(self, turns: list[dict[str, Any]], items: list[tuple[str, str]], draft: str,
                  tools: str) -> AsyncIterator[dict | Settled]:
        """Yield reasoning lines, then Settled (the answer and the report). `turns`: the conversation the model
        saw; `items`: the listed items; `draft`: the checked draft; `tools`: the tool results as text."""
        question = next((text_of(m["content"]) for m in reversed(turns) if m["role"] == "user"), "")
        conversation = "\n\n".join(f"{m['role']}: {text_of(m.get('content'))}" for m in turns)
        listed = "".join(f"{i}: {t}\n" for i, t in items)
        head = f"Conversation:\n{conversation}\n\nItems listed in the last message:\n{listed}\n"
        user = head + (f"Tool results:\n{tools}\n\n" if tools else "") + f"Draft answer:\n{draft}"
        messages = [{"role": "system", "content": self.cfg.ledger_prompt}, {"role": "user", "content": user}]
        yield _say(f"\n\nEvidence ledger: {len(items)} listed items\n")
        if tools and not self._fits(messages, self.cfg.ledger_request):
            messages[1]["content"] = head + f"Draft answer:\n{draft}"
            yield _say("- tool results left out (context budget)\n")
        reply, problem = await self._ask(messages, self.cfg.ledger_request)
        parsed = parse_ledger(reply, items) if reply is not None else None
        if parsed is None:
            why = f"no readable ledger ({problem or 'not valid JSON'})"
            yield _say(f"- {why}; the answer stays as drafted\n")
            yield Settled(draft, _report([], [], [], False, why))
            return
        entries, skipped = parsed
        if gone := missing(entries, skipped, items):
            yield _say(f"- no entry for {', '.join(gone)}; asking again\n")
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": self.cfg.repair_prompt.replace("{missing}", ", ".join(gone))}]
            more, _ = await self._ask(messages, self.cfg.ledger_request)
            wanted = {key(i) for i in gone}
            if more and (extra := parse_ledger(more, items)):
                entries += [e for e in extra[0] if key(e.id) in wanted]
                skipped += [s for s in extra[1] if key(s) in wanted]
            if gone := missing(entries, skipped, items):
                yield _say(f"- still no entry for {', '.join(gone)} (not yet judged)\n")
        if changed := settle(entries, self.cfg.strong_words):
            yield _say("".join(f"Evidence: {e.id} {e.status} ({e.why}; model said {e.claimed})\n" for e in changed))
        elif not gone:
            yield _say(f"- {len(entries)} items, no status changed: the answer stays as drafted\n")
            yield Settled(draft, _report(entries, skipped, gone, False))
            return

        messages = [{"role": "system", "content": self.cfg.writer_prompt},
                    {"role": "user", "content": f"Request:\n{question}\n\nSettled ledger:\n"
                                                f"{render(entries, skipped, gone)}\n\nDraft answer:\n{draft}"}]
        written, problem = await self._ask(messages, self.cfg.writer_request)
        answer = (written or "").strip()
        if not answer:
            why = f"no answer written ({problem or 'empty reply'})"
            yield _say(f"- {why}; the answer stays as drafted\n")
            yield Settled(draft, _report(entries, skipped, gone, False, why))
            return
        yield _say("- answer written from the settled ledger\n")
        if self.checks:
            lines, v = [], self.verifier
            if v.arithmetic_check:
                answer, fixed, _ = fix_arithmetic(answer, v.arithmetic_rel_tol, v.arithmetic_prefixes,
                                                  question if v.keep_quotes else "", v.arithmetic_strict_parse,
                                                  v.arithmetic_max_change)
                lines += [f"- arithmetic: {a['expression']} = {a['stated']} → {a['fixed']}\n"
                          for a in fixed if "note" not in a]
            answer, found = run_checkers(answer, v.checkers, question, v.keep_quotes, v.checkers_clear_only)
            lines += [f"- {c['checker']}: {c['text']!r} → {c['fixed']!r}\n" for c in found if c["fixed"] is not None]
            if lines:
                yield _say("".join(lines))
        yield Settled(answer, _report(entries, skipped, gone, True))
