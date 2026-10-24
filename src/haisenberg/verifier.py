"""Claim verification (ARCHITECTURE.md "Verification"): split the draft into claims, check numbers and
formulas in code against structured records, judge the rest with one batched model call, re-search what
is not supported, then label, fix or remove each claim. Nothing here names a source or kind (D-021):
records say what can be checked about them (`values()`, `formulas()`, `names()`).
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import Config
from .calc import Arithmetic, find_arithmetic, prefixed
from .checkers import CHECKERS
from .l1.text import parse_formula, to_hill
from .llm import LLMClient, UpstreamError, estimate_tokens
from .tools import SEARCH, Knowledge, Source, Sources

log = logging.getLogger(__name__)

SUPPORTED, UNVERIFIED, CORRECTED, REMOVED, NO_CLAIM = "supported", "unverified", "corrected", "removed", "no_claim"
OUTCOMES = (SUPPORTED, UNVERIFIED, CORRECTED, REMOVED)

# Same-length normalisation for reading numbers and formulas, so spans stay valid in the original text.
_NORM = str.maketrans("−⁻⁺⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉  \xa0", "--+01234567890123456789   ")
_NUM = re.compile(
    r"(?<![\w.^/\\-])(-?)(\d{1,3}(?:(?:[ ,]|\\,)\d{3})+(?!\d)|\d+)"  # sign, integer part (maybe grouped in threes)
    r"(?:\.(\d+(?: \d{1,3}(?!\d))*))?"  # decimals (maybe grouped, CODATA style)
    r"(?:\s*(?:[eE]|(?:[×x*·⋅]|\\times|\\cdot)\s*10\s*(?:\^|\*\*)?)\s*\{?\(?([-+]?\d+)\)?\}?)?"  # exponent
    r"(?!\w)"
)
_FORMULA = re.compile(r"(?<![\w()\[\]-])((?:[A-Z][a-z]?\d*)+)(?![\w()\[\]-])")
_CITES = re.compile(r"\s*\[S\d+(?:\s*[,;]\s*S?\d+)*\]")
_CITE_MARKS = re.compile(r"[ \t]*\[S\d+(?:\s*[,;]\s*S?\d+)*\]")  # the same, never across a line break
_END = re.compile(r"[.!?]+(?:\s*\[S\d+(?:\s*[,;]\s*S?\d+)*\])*(?=\s|$)")
_ABBREV = {"e.g", "i.e", "approx", "ca", "vs", "etc", "cf", "fig", "eq", "no", "al"}
_MARKERS = re.compile(r"\s*(?:(?:[-*+]|\d+[.)])\s+|>\s*)*")
_VERDICT = re.compile(r"^\W*C(\d+)\W+(supports|contradicts|not[_ ]said|no[_ ]claim)\b(.*)$", re.I)
# A label (an id such as A1, C or (b), maybe after one word: "Claim C", "Claim 3") and a colon starts the line.
_LABELLED = re.compile(r"[*_`]*(?:(?:[^\W\d_]+\s+)?\(?[^\W\d_]\d{0,2}\)?|[^\W\d_]+\s+\d{1,2})[*_`\s]*[:：][*_`]*(?:\s|$)")
# An equation, comparison or operation with numbers: a computation, not a sourceable fact.
_COMPUTED = re.compile(r"[=≈<>≤≥]|\d\s*(?:[×*/÷·⋅]|\\times|\\cdot)\s*[-−(]?\s*\d|\\frac")


@dataclass
class Claim:
    n: int  # C1, C2, ... in draft order
    start: int  # span in the draft
    end: int
    text: str
    cites: list[str]
    outcome: str = ""
    ids: list[str] = field(default_factory=list)  # supporting (or correcting) source ids
    fixed: str | None = None  # the corrected text
    note: str = ""


@dataclass
class Verification:
    answer: str  # the draft rendered by verifier.answer_style (see render), after the consistency review
    claims: list[Claim]
    consistency: list[dict[str, str]] = field(default_factory=list)  # applied review fixes: old, new, why
    arithmetic: list[dict[str, str]] = field(default_factory=list)  # fixed results: expression, stated, fixed
    checks: list[dict[str, Any]] = field(default_factory=list)  # domain checker findings: checker, text, fixed, note

    def report(self) -> dict[str, Any]:
        claims = [c for c in self.claims if c.outcome != NO_CLAIM]
        return {
            "claims": [{"text": c.text, "outcome": c.outcome, "ids": c.ids, "note": c.note} for c in claims],
            "counts": {o: sum(c.outcome == o for c in claims) for o in OUTCOMES},
            "consistency": self.consistency,
            "arithmetic": self.arithmetic,
            "checks": self.checks,
        }


# --- claims ---------------------------------------------------------------------------------------


def split_claims(draft: str, skip_labelled: bool = False) -> list[Claim]:
    """Sentences of paragraphs and list items; not headings, code, display math, tables or lead-ins, and with
    `skip_labelled` not labelled answer or verdict lines (`A1: …`, `Claim C: Incorrect`)."""
    claims: list[Claim] = []
    pos, fence = 0, False
    for line in draft.splitlines(keepends=True):
        start, pos = pos, pos + len(line)
        stripped = line.strip()
        if stripped.startswith("```"):
            fence = not fence
            continue
        if fence or not stripped or stripped.startswith(("#", "|", "$$", "\\[", "\\]")):
            continue
        body_at = start + _MARKERS.match(line).end()
        body = draft[body_at : start + len(line.rstrip("\r\n"))]
        if skip_labelled and _LABELLED.match(body):
            continue
        for a, b in _sentences(body):
            text = body[a:b]
            lead = len(text) - len(text.lstrip())
            text = text.strip()
            if _is_claim(text):
                at = body_at + a + lead
                claims.append(Claim(len(claims) + 1, at, at + len(text), text, cites(text)))
    return claims


def _sentences(body: str) -> list[tuple[int, int]]:
    out, start = [], 0
    for m in _END.finditer(body):
        before = re.search(r"[\w.]+$", body[start : m.start()])
        word = before.group().lower().rstrip(".") if before else ""
        if word in _ABBREV:
            continue
        out.append((start, m.end()))
        start = m.end()
    if body[start:].strip():
        out.append((start, len(body)))
    return out


def plain(text: str) -> str:
    """The claim without citation markers and markdown emphasis."""
    return re.sub(r"[*`]", "", _CITES.sub("", text)).strip()


def _is_claim(text: str) -> bool:
    p = plain(text)
    if not re.search(r"[A-Za-z]", p) or p.endswith((":", "?")):
        return False
    return len(p.split()) >= 3 or bool(re.search(r"\d", p))


def cites(text: str) -> list[str]:
    return [f"S{n}" for m in _CITES.finditer(text) for n in re.findall(r"\d+", m.group())]


# --- numbers and formulas -----------------------------------------------------------------------


@dataclass
class Number:
    value: float
    half: float  # half a unit in the last written digit
    start: int
    end: int
    decimals: int
    scientific: bool


def read_numbers(text: str) -> list[Number]:
    norm = text.translate(_NORM)
    out = []
    for m in _NUM.finditer(norm):
        sign, integer, decimals, exp = m.groups()
        digits, dec = re.sub(r"[ ,\\]", "", integer), (decimals or "").replace(" ", "")
        e = int(exp) if exp else 0
        value = float(f"{sign}{digits}.{dec or '0'}") * 10.0**e
        out.append(Number(value, 0.5 * 10.0 ** (e - len(dec)), m.start(), m.end(), len(dec), exp is not None))
    return out


def read_formulas(text: str, need_digit: bool = True) -> list[tuple[str, int, int]]:
    """(Hill formula, start, end) of every token that parses as a formula (H2O, CH3CH2OH); tokens without
    a digit (NaCl, but also SI or He) only when need_digit is False."""
    norm = text.translate(_NORM)
    out = []
    for m in _FORMULA.finditer(norm):
        token = m.group(1)
        if (re.search(r"\d", token) or not need_digit) and parse_formula(token) and (hill := to_hill(token)):
            out.append((hill, m.start(), m.end()))
    return out


def _matches(n: Number, ref: float, rel_tol: float) -> bool:
    return abs(n.value - ref) <= max(n.half, rel_tol * abs(ref))


def _format(ref: float, like: Number) -> str:
    if like.scientific or ref == 0 or not 1e-3 <= abs(ref) < 1e6:
        e = math.floor(math.log10(abs(ref))) if ref else 0
        return f"{ref / 10.0**e:.{max(like.decimals, 3)}f} × 10^{e}"
    return f"{ref:.{like.decimals}f}"


_UNIT_AFTER = re.compile(r"(?:[ \t]|\\[,;! ]|\\(?:text|mathrm)\{)*([A-Za-zА-Яа-яЁёµμΩ]+(?:[·*/⋅∙][A-Za-zА-Яа-яЁёµμΩ]+)*)")
_NEXT_RESULT = re.compile(r"[ \t}$]*=[ \t$]*[-−]?\d(?:[\d.,]|[ \xa0  ]\d)*")
_SCALES = (1.0, 1e3, 1e-3, 1e6, 1e-6, 1e9, 1e-9)


def _result_units(tail: str) -> str:
    """The unit right after a result and the one after the next `= number` of its chain (`= 2452 W = 2.45 kW`)."""
    units = []
    if m := _UNIT_AFTER.match(tail):
        units.append(m[1])
        tail = tail[m.end() :]
    if (nxt := _NEXT_RESULT.match(tail)) and (m := _UNIT_AFTER.match(tail, nxt.end())):
        units.append(m[1])
    return " ".join(units)


def fix_arithmetic(draft: str, rel_tol: float, prefixes: bool = True) -> tuple[str, list[dict[str, str]], int]:
    """(the draft with every wrong `expression = result` fixed in place, the fixes, how many were checked).
    A result is wrong only when it is wrong under every reading of its % signs (a% as a/100 or as a, a
    result in % as itself or as a fraction) and, with `prefixes`, of the SI prefixes in its chain (a unit
    with a prefix on an operand, right after the result or after the next `=`: the result may be scaled by
    10^±3, ±6, ±9): off by more than half a unit of its last digit and rel_tol. It is rewritten only when the
    right value is clear; otherwise it is left as written, and so is a number that cannot be read as one."""
    decimal_comma = bool(re.search("[А-Яа-яЁё]", draft))  # Russian text: every comma between digits
    fixes, checked = [], 0
    for a in find_arithmetic(draft, decimal_comma):
        window = draft[a.result_at : a.result_at + 80]
        comma = r"(\d),(\d)" if decimal_comma else r"(\d),(\d{1,2}(?!\d)|\d{4,})"
        window = re.sub(comma, r"\1.\2", window)  # same length: spans stay valid
        numbers = read_numbers(window)
        if not numbers or numbers[0].start != 0:
            continue
        if re.match(r"(?:\\[,;! ]|[ \xa0  ,.'])\d", window[numbers[0].end :]):
            continue  # "4\,900": digits read_numbers did not join, so not one reading
        n, checked = numbers[0], checked + 1
        in_percent = a.as_percent or bool(re.match(r"\s*\\?%", window[n.end :]))
        stated = [(n.value, n.half)] + ([(n.value / 100, n.half / 100)] if in_percent else [])
        scaled = prefixes and not in_percent and prefixed(f"{a.expression} {_result_units(window[n.end :])}")
        scales = _SCALES if scaled else (1.0,)
        if any(abs(v * s - lhs) <= max(half * s, rel_tol * abs(lhs))
               for lhs in (a.fraction, a.plain) for v, half in stated for s in scales):
            continue
        right = _right_result(a, in_percent)
        if right is not None and scaled:  # the scale meant: the one within a factor of ten of the stated value
            near = [right / s for s in scales if right * n.value > 0 and abs(math.log10(right / s / n.value)) < 1]
            right = near[0] if len(near) == 1 else None
        if right is None:
            continue
        start, end = a.result_at + n.start, a.result_at + n.end
        fixed = _format(right, n)
        fixes.append({"expression": a.expression, "stated": draft[start:end], "fixed": fixed, "at": (start, end)})
    for f in sorted(fixes, key=lambda f: f["at"], reverse=True):
        start, end = f.pop("at")
        draft = draft[:start] + f["fixed"] + draft[end:]
    return draft, fixes, checked


def run_checkers(draft: str, names: list[str]) -> tuple[str, list[dict[str, Any]]]:
    """Run the configured domain checkers (D-034); apply their fixes in place, later spans first, skipping a fix
    that overlaps one already applied. Returns the draft and every finding (fixed or only reported)."""
    found = [(name, f) for name in names for f in CHECKERS[name](draft)]
    applied_from = len(draft) + 1
    for _, f in sorted(found, key=lambda nf: nf[1].start, reverse=True):
        if f.fixed is not None and f.end <= applied_from:
            draft = draft[: f.start] + f.fixed + draft[f.end :]
            applied_from = f.start
    return draft, [{"checker": name, "text": f.text, "fixed": f.fixed, "note": f.note} for name, f in found]


def _right_result(a: Arithmetic, in_percent: bool) -> float | None:
    """The value to write for a wrong result, or None when the intended reading is not clear."""
    if not in_percent:
        return a.fraction if a.fraction == a.plain else None  # "15% × 200 = 40": 30 or 3000? leave it
    if a.as_percent or (a.fraction <= 1.5 and not a.times_100):
        return 100 * a.fraction  # a fraction written as a percent
    if a.times_100 and not a.has_percent:
        return a.plain  # "x / y × 100 = r %": the value is already a percent
    return None


def _mentions(text: str, names: list[str]) -> bool:
    low = text.lower()
    return any(len(n) >= 3 and re.search(rf"(?<!\w){re.escape(n.lower())}(?!\w)", low) for n in names)


def code_check(claim: Claim, sources: Sources, rel_tol: float) -> tuple[str, list[str], str | None, str] | None:
    """(outcome, ids, fixed text, note) from the structured records, or None when code cannot decide."""
    norm = claim.text.translate(_NORM)
    nums, forms = read_numbers(claim.text), read_formulas(claim.text)
    cited = [s for i in claim.cites if (s := sources.get(i))]
    says_formula = "formula" in norm.lower()

    for s in cited:  # contradictions need a cited record and a clear quantity
        values, fixes = s.record.values(), []
        for n in nums:
            if any(_matches(n, v.number, rel_tol) for v in values):
                continue
            for v in values:
                same_unit = v.unit and re.match(v.unit, norm[n.end :], re.I)
                same_size = not v.unit and n.value * v.number > 0 and abs(math.log10(n.value / v.number)) < 1
                if same_unit or same_size:
                    fixes.append((n.start, n.end, _format(v.number, n), f"{norm[n.start:n.end]} ≠ {v.number:g}"))
                    break
        hills = {to_hill(f) for f in s.record.formulas()}
        if hills and forms and says_formula and not any(h in hills for h, _, _ in forms):
            _, a, b = forms[0]
            fixes.append((a, b, s.record.formulas()[0], f"{norm[a:b]} ≠ {s.record.formulas()[0]}"))
        if fixes:
            text = claim.text
            for a, b, new, _ in sorted(fixes, reverse=True):
                text = text[:a] + new + text[b:]
            return CORRECTED, [s.id], text, "; ".join(f[3] for f in fixes) + f" ({s.id})"

    ids, notes = [], []  # support: every number matches a record, or a stated formula does
    for n in nums:
        hit = _support(n, cited, sources, norm, rel_tol)
        if hit is None:
            ids = []
            break
        ids.append(hit[0])
        notes.append(f"{norm[n.start:n.end]} ≈ {hit[1]:g} ({hit[0]})")
    if not ids and says_formula:
        for s in [*cited, *(x for x in sources.items if x not in cited)]:
            hills = {to_hill(f) for f in s.record.formulas()}
            if hills and (s in cited or _mentions(norm, s.record.names())) and any(h in hills for h, _, _ in forms):
                ids, notes = [s.id], [f"formula = {s.record.formulas()[0]} ({s.id})"]
                break
    if ids:
        return SUPPORTED, list(dict.fromkeys(ids)), None, ", ".join(notes)
    return None


def _support(n: Number, cited: list[Source], sources: Sources, norm: str, rel_tol: float) -> tuple[str, float] | None:
    for s in [*cited, *(x for x in sources.items if x not in cited)]:
        named = s in cited or _mentions(norm, s.record.names())
        for v in s.record.values():
            if _matches(n, v.number, rel_tol) and (named or not v.unit):
                return s.id, v.number
    return None


def results(draft: str) -> list[float]:
    """Every number written right after `=` or `≈` in the draft and not followed by an operator: values the
    answer computed."""
    out = []
    for m in re.finditer(r"[=≈][ \t$]*", draft):
        text = draft[m.end() : m.end() + 40]
        nums = read_numbers(text)
        if nums and nums[0].start == 0 and not re.match(r"\s*(?:[×*/·⋅÷+^−-]|\\times|\\cdot)", text[nums[0].end :]):
            out.append(nums[0].value)
    return out


def is_computation(claim: Claim, given: list[float], computed: list[float], rel_tol: float) -> bool:
    """The claim is a calculation: an equation, comparison or operation with numbers, or a sentence whose
    numbers are all the task's (`given`) or the answer's results (`computed`), at least one a result."""
    nums = read_numbers(claim.text)
    if not nums:
        return False
    if _COMPUTED.search(claim.text):  # the claim as written: plain() would drop a "*" operator
        return True
    is_result = [any(_matches(n, v, rel_tol) for v in computed) for n in nums]
    return any(is_result) and all(r or any(_matches(n, v, rel_tol) for v in given) for n, r in zip(nums, is_result))


# --- judge ------------------------------------------------------------------------------------------


def parse_fixes(text: str) -> list[dict[str, str]]:
    """The consistency review's OLD/NEW/WHY blocks (an `OK` reply has none). Markdown emphasis around the
    keys is ignored; a block needs an OLD and a NEW line."""
    blocks: list[dict] = []
    for line in text.splitlines():
        m = re.match(r"^\W*(OLD|NEW|WHY)\W*?:\s?(.*)$", line.strip(), re.I)
        if not m:
            continue
        key, value = m.group(1).lower(), m.group(2).strip().lstrip("*_").strip()
        if len(value) > 1 and (value[0], value[-1]) in (('"', '"'), ("'", "'"), ("`", "`"), ("«", "»"), ("“", "”")):
            value = value[1:-1]  # a quote around the text is not part of it
        if key == "old":
            blocks.append({"old": value, "new": None, "why": ""})
        elif blocks:
            blocks[-1][key] = value
    return [b for b in blocks if b["old"] and b["new"] is not None]


def parse_verdicts(text: str, given: set[str]) -> dict[int, tuple[str, list[str], str | None]]:
    """Claim number -> (verdict, decisive ids among `given`, correction) from the judge's reply lines."""
    out: dict[int, tuple[str, list[str], str | None]] = {}
    for line in text.splitlines():
        m = _VERDICT.match(line.strip())
        if not m:
            continue
        verdict = m.group(2).lower().replace(" ", "_")
        ids_part, _, correction = m.group(3).partition("|")
        ids = [i for i in dict.fromkeys(f"S{x}" for x in re.findall(r"S(\d+)", ids_part)) if i in given]
        if verdict in ("supports", "contradicts") and not ids:
            verdict = "not_said"
        out.setdefault(int(m.group(1)), (verdict, ids, correction.strip() or None))
    return out


class Verifier:
    def __init__(self, cfg: Config, llm: LLMClient, knowledge: Knowledge):
        self.cfg = cfg.verifier
        self.llm = llm
        self.knowledge = knowledge
        judge_max = self.cfg.request.get("max_tokens") or cfg.model_backend.max_tokens
        self.budget = cfg.context_tokens - judge_max  # prompt tokens a judge call may use
        self.chars_per_token = cfg.model_backend.chars_per_token

    async def verify(self, question: str, draft: str, sources: Sources) -> AsyncIterator[dict | Verification]:
        """Yield reasoning deltas while checking, then the Verification."""
        arithmetic, checked = [], 0
        if self.cfg.arithmetic_check:
            draft, arithmetic, checked = fix_arithmetic(draft, self.cfg.arithmetic_rel_tol,
                                                        self.cfg.arithmetic_prefixes)
        if checked:
            yield _say(f"\n\nArithmetic: {checked} calculation{'s' if checked != 1 else ''} recomputed,"
                       f" {len(arithmetic)} fixed\n"
                       + "".join(f"- {a['expression']} = {a['stated']} → {a['fixed']}\n" for a in arithmetic))
        draft, checks = run_checkers(draft, self.cfg.checkers)
        for name in self.cfg.checkers:
            found = [c for c in checks if c["checker"] == name]
            if found:
                yield _say(f"\n\nCheck {name}: {len(found)} finding{'s' if len(found) != 1 else ''}\n" + "".join(
                    f"- {c['text']!r} → {c['fixed']!r} ({c['note']})\n" if c["fixed"] is not None
                    else f"- {c['text']!r}: {c['note']}\n" for c in found))
        derived = self.cfg.skip_derived_claims
        claims = split_claims(draft, skip_labelled=derived)
        yield _say(f"\n\nVerification: {len(claims)} claims\n")
        pending = []
        given, computed = [n.value for n in read_numbers(question)], results(draft)
        for c in claims:
            if found := code_check(c, sources, self.cfg.rel_tol):
                c.outcome, c.ids, c.fixed, c.note = found
                c.note = f"code: {c.note}"
                yield _say(f"- C{c.n} {c.outcome} ({c.note})\n")
            elif derived and is_computation(c, given, computed, self.cfg.rel_tol):
                c.outcome, c.note = UNVERIFIED, "computation: checked in code, not against sources"
                yield _say(f"- C{c.n} {c.outcome} ({c.note})\n")
            else:
                pending.append(c)

        cited = [s for s in sources.items if any(s.id in c.cites for c in pending)]
        if pending and cited:
            async for line in self._judge(question, pending, cited):
                yield line
        research = [c for c in pending if not c.outcome]
        for c in research[self.cfg.max_research :]:
            c.outcome, c.note = UNVERIFIED, "not supported; no re-search left"
        research = research[: self.cfg.max_research]

        found_for: dict[int, list[str]] = {}
        for c in research:
            query = plain(c.text)
            yield _say(f"\n> {SEARCH}(query={json.dumps(query, ensure_ascii=False)})  [re-search C{c.n}]\n")
            _, lines, ids = self.knowledge.run(SEARCH, json.dumps({"query": query}), sources, self.cfg.research_k)
            yield _say("".join(f"- {line}\n" for line in lines))
            found_for[c.n] = ids
        again = [c for c in research if found_for.get(c.n) or c.cites]
        if again:
            wanted = {i for c in again for i in [*found_for.get(c.n, []), *c.cites]}
            async for line in self._judge(question, again, [s for s in sources.items if s.id in wanted]):
                yield line
        for c in research:
            if not c.outcome:
                c.outcome, c.note = UNVERIFIED, c.note or "not supported after re-search"
        for c in claims:
            if c.outcome == CORRECTED:
                yield _say(f"- C{c.n} corrected from {', '.join(c.ids)}: {plain(c.fixed or '')}\n")
            elif c.outcome == REMOVED:
                yield _say(f"- C{c.n} removed (contradicted by {', '.join(c.ids)})\n")
        answer, fixes = render(draft, claims, self.cfg.answer_style == "annotated"), []
        if self.cfg.consistency_check:
            review = self._consistency(question, answer)
            try:
                async for item in review:
                    if isinstance(item, tuple):
                        answer, fixes = item
                    else:
                        yield item
            finally:
                await review.aclose()
        yield Verification(answer, claims, consistency=fixes, arithmetic=arithmetic, checks=checks)

    async def _consistency(self, question: str, answer: str):
        """One review call over the whole answer and the task. Yields reasoning lines, then (answer, fixes)."""
        messages = [{"role": "system", "content": self.cfg.consistency_prompt},
                    {"role": "user", "content": f"Task:\n{question}\n\nAnswer:\n{answer}"}]
        if estimate_tokens(messages, self.chars_per_token) > self.budget:
            yield _say("\n\nConsistency review skipped: the answer is too long for the context\n")
            yield answer, []
            return
        try:
            reply = await self._complete(messages)
        except (UpstreamError, httpx.HTTPError) as e:
            log.error("consistency review failed: %s", e)
            yield _say(f"\n\nConsistency review failed: {e}\n")
            yield answer, []
            return
        blocks, applied = parse_fixes(reply), []
        yield _say(f"\n\nConsistency review: {len(blocks)} fix{'es' if len(blocks) != 1 else ''} proposed\n")
        for f in blocks:
            count = answer.count(f["old"]) if f["old"] else 0
            if count == 1 and f["new"] != f["old"]:
                answer = answer.replace(f["old"], f["new"])
                applied.append(f)
                yield _say(f"- {f['old']!r} → {f['new']!r}" + (f" ({f['why']})" if f["why"] else "") + "\n")
            else:
                why = "not in the answer" if count == 0 else "unchanged" if count == 1 else f"found {count} times"
                yield _say(f"- skipped {f['old']!r} ({why})\n")
        yield answer, applied

    async def _judge(self, question: str, claims: list[Claim], passages: list[Source]):
        """One judge call for these claims; sets their outcome. Yields reasoning lines."""
        passages = list(passages)
        while True:  # leave passages out from the end until the call fits the context
            user = (
                f"Question: {question}\n\nPassages:\n"
                + "\n\n".join(f"[{s.id}] {s.record.label()}\n{s.passage}" for s in passages)
                + "\n\nClaims:\n"
                + "\n".join(f"C{c.n}: {plain(c.text)}" for c in claims)
            )
            messages = [{"role": "system", "content": self.cfg.judge_prompt}, {"role": "user", "content": user}]
            if not passages or estimate_tokens(messages, self.chars_per_token) <= self.budget:
                break
            yield _say(f"- {passages.pop().id} left out of the judge call (context budget)\n")
        given = {s.id for s in passages}
        try:
            reply = await self._complete(messages)
        except (UpstreamError, httpx.HTTPError) as e:
            log.error("judge call failed: %s", e)
            yield _say(f"- judge failed: {e}\n")
            return
        verdicts = parse_verdicts(reply, given)
        for c in claims:
            verdict, ids, correction = verdicts.get(c.n, ("not_said", [], None))
            c.note = f"judge: {verdict}" + (f" {' '.join(ids)}" if ids else "")
            if verdict == "supports":
                c.outcome, c.ids = SUPPORTED, ids
            elif verdict == "contradicts" and not correction and self.cfg.keep_uncorrected:
                c.outcome, c.note = UNVERIFIED, f"{c.note}, no correction: kept"  # no value stated against it
            elif verdict == "contradicts":
                c.outcome, c.ids, c.fixed = (CORRECTED, ids, correction) if correction else (REMOVED, ids, None)
            elif verdict == "no_claim":
                c.outcome = NO_CLAIM
            yield _say(f"- C{c.n} {c.note}\n")

    async def _complete(self, messages: list[dict[str, Any]]) -> str:
        """The judge's answer text; its reasoning tokens, if any, are ignored."""
        parts: list[str] = []
        stream = self.llm.stream_chat(messages, **self.cfg.request)
        try:
            async for chunk in stream:
                parts.append(((chunk.get("choices") or [{}])[0].get("delta") or {}).get("content") or "")
        finally:
            await stream.aclose()
        return "".join(parts)


# --- answer -----------------------------------------------------------------------------------------


def strip_cites(text: str) -> str:
    """The text without `[S#]` citation markers."""
    return _CITE_MARKS.sub("", text)


def render(draft: str, claims: list[Claim], annotated: bool = True) -> str:
    """The draft with every claim rewritten by its outcome. Annotated: labels, ids and the summary line.
    Plain: fixes applied and contradicted claims without a fix dropped, silently; no ids, no summary."""
    out = draft
    for c in sorted(claims, key=lambda c: c.start, reverse=True):
        end = c.end + (1 if c.outcome == REMOVED and out[c.end : c.end + 1] == " " else 0)
        out = out[: c.start] + (_render(c) if annotated else _plain(c)) + out[end:]
    out = re.sub(r"(?m)^[ \t]*(?:[-*+]|\d+[.)])[ \t]*(?:\r?\n|$)", "", out)  # list items left empty by removals
    if not annotated:
        return strip_cites(out).rstrip()
    counted = [c for c in claims if c.outcome != NO_CLAIM]
    if not counted:
        return f"{out.rstrip()}\n\n*Verification: no checkable claims.*"
    n = {o: sum(c.outcome == o for c in counted) for o in OUTCOMES}
    summary = (
        f"*Verification: {len(counted)} claim{'s' if len(counted) != 1 else ''}: {n[SUPPORTED]} supported,"
        f" {n[UNVERIFIED]} unverified, {n[CORRECTED]} corrected, {n[REMOVED]} removed.*"
    )
    return f"{out.rstrip()}\n\n{summary}"


def _render(c: Claim) -> str:
    if c.outcome == NO_CLAIM:
        return _CITES.sub("", c.text)  # a sentence that states nothing is supported by nothing
    if c.outcome == REMOVED:
        return ""
    body = _CITES.sub("", c.fixed if c.outcome == CORRECTED and c.fixed else c.text).rstrip()
    m = re.match(r"(.*?)([.!?;]*)$", body, re.S)
    core, punct = m.group(1).rstrip(), m.group(2)
    tags = "".join(f"[{i}]" for i in c.ids)
    if c.outcome == SUPPORTED:
        return f"{core} {tags}{punct}"
    if c.outcome == CORRECTED:
        return f"{core} *(corrected from {tags})*{punct}"
    return f"{core} *(unverified)*{punct}"


def _plain(c: Claim) -> str:
    if c.outcome == REMOVED:
        return ""
    return c.fixed if c.outcome == CORRECTED and c.fixed else c.text


def _say(text: str) -> dict[str, str]:
    return {"reasoning_content": text}
