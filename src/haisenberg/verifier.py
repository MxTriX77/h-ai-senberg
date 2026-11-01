"""Claim verification (ARCHITECTURE.md "Verification"): split the draft into claims, check numbers and
formulas in code against structured records, judge the rest with one batched model call, re-search what
is not supported, then label, fix or remove each claim. Nothing here names a source or kind (D-021):
records say what can be checked about them (`values()`, `formulas()`, `names()`).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

import httpx

from .config import Config
from .calc import Arithmetic, computations, converted, find_arithmetic, prefixed
from .checkers import CHECKERS
from .l1.store import STOPWORDS, Value
from .l1.text import parse_formula, to_hill
from .l1.units import parse_unit, read_unit, same_unit
from .llm import LLMClient, UpstreamError, estimate_tokens
from .tools import SEARCH, Knowledge, Source, Sources, term_share

log = logging.getLogger(__name__)

SUPPORTED, UNVERIFIED, CORRECTED, REMOVED, NO_CLAIM = "supported", "unverified", "corrected", "removed", "no_claim"
OUTCOMES = (SUPPORTED, UNVERIFIED, CORRECTED, REMOVED)

# Same-length normalisation for reading numbers and formulas, so spans stay valid in the original text.
_NORM = str.maketrans("−⁻⁺⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉  \xa0", "--+01234567890123456789   ")
# Numbers keep their superscripts: `× 10³` must not read like `× 103` or `× 100`.
_SUPER = "⁻⁺⁰¹²³⁴⁵⁶⁷⁸⁹"
_NUM_NORM = {k: v for k, v in _NORM.items() if chr(k) not in _SUPER}
_FROM_SUPER = str.maketrans(_SUPER, "-+0123456789")
_NUM = re.compile(
    r"(?<![\w.^/\\-])(-?)(\d{1,3}(?:(?:[ ,]|\\,)\d{3})+(?!\d)|\d+)"  # sign, integer part (maybe grouped in threes)
    r"(?:\.(\d+(?: \d{1,3}(?!\d))*(?:(?<=[. ]\d{3}) \d{4}(?!\d))?))?"  # decimals (maybe grouped, CODATA style:
    # threes, the last group maybe four: 9.109 383 7139)
    r"(?:\s*(?:[eE]\s*\{?\(?([-+]?\d+)\)?\}?"  # exponent: e-notation, or × 10 with ^ or **, with superscript
    r"|(?:[×x*·⋅]|\\times|\\cdot)\s*10(?:\s*(?:\^|\*\*)\s*\{?\(?([-+]?\d+)\)?\}?"  # digits, or with a minus
    r"|([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)|(-\d+))))?"  # right after the 10 (`× 10-19`); `× 100` has no exponent
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
_BARE_LABEL = re.compile(r"\W*[^\W\d_]\d{1,2}\W*")  # S3. / C2. / A1): a label cut off its line, not a claim
# Quotations of the task: runs this long (after _quote_text) that the answer shares with the task.
_QUOTE_MIN = 20
_QUOTE_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻₀₁₂₃₄₅₆₇₈₉₊₋", "0123456789+-0123456789+-")
_QUOTE_DROP = frozenset("*_`\"'“”‘’«»„")  # Markdown emphasis and quote marks


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
    consistency: list[dict[str, str]] = field(default_factory=list)  # applied review fixes: old, new, why, conflict
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
    if not re.search(r"[A-Za-z]", p) or p.endswith((":", "?")) or _BARE_LABEL.fullmatch(p):
        return False
    return len(p.split()) >= 3 or bool(re.search(r"\d", p))


def cites(text: str) -> list[str]:
    return [f"S{n}" for m in _CITES.finditer(text) for n in re.findall(r"\d+", m.group())]


# --- quotations of the task -----------------------------------------------------------------------


def _quote_text(text: str) -> tuple[str, list[int]]:
    """The text as compared for quotations (sub/superscripts as plain digits and signs, no emphasis or quote
    marks, whitespace runs as one space) and the position in `text` of each of its characters."""
    out: list[str] = []
    at: list[int] = []
    for i, ch in enumerate(text.translate(_QUOTE_MAP)):
        if ch in _QUOTE_DROP or (ch.isspace() and out and out[-1] == " "):
            continue
        out.append(" " if ch.isspace() else ch)
        at.append(i)
    return "".join(out), at


def quoted(text: str, task: str) -> list[bool]:
    """Per character of `text`: part of a run of at least _QUOTE_MIN characters that also occurs in the task
    (both read by _quote_text), that is a quotation of the material under review. Whitespace after the first
    of a run, emphasis and quote marks are never marked."""
    mask = [False] * len(text)
    norm, at = _quote_text(text)
    hay = _quote_text(task)[0]
    windows = {hay[i : i + _QUOTE_MIN] for i in range(len(hay) - _QUOTE_MIN + 1)}
    for i in range(len(norm) - _QUOTE_MIN + 1):
        if norm[i : i + _QUOTE_MIN] in windows:
            for j in at[i : i + _QUOTE_MIN]:
                mask[j] = True
    return mask


def is_quotation(text: str, mask: list[bool], start: int, end: int) -> bool:
    """text[start:end] quotes the task (see quoted) and, apart from the quotation and its citation markers, has
    no number and at most two words: a label or a verdict mark (`✗ WRONG`), not a statement of its own."""
    if not any(mask[start:end]):
        return False
    rest = plain("".join(" " if mask[i] else text[i] for i in range(start, end)))
    return not read_numbers(rest) and len(_words(rest)) <= 2


def rewrites_quote(answer: str, task: str, old: str, new: str) -> bool:
    """Replacing `old` (once in the answer) by `new` changes quoted text: the part of `old` that differs from
    `new` (after their common beginning and end) is quoted in part, or, when `new` only inserts text, a
    character next to the insertion is."""
    mask, at = quoted(answer, task), answer.find(old)
    head = len(os.path.commonprefix([old, new]))
    tail = len(os.path.commonprefix([old[head:][::-1], new[head:][::-1]]))
    lo, hi = at + head, at + len(old) - tail
    if lo == hi:
        lo, hi = lo - 1, hi + 1
    return any(mask[max(lo, 0) : hi])


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
    norm = text.translate(_NUM_NORM)
    out = []
    for m in _NUM.finditer(norm):
        sign, integer, decimals, *exps = m.groups()
        exp = next((x.translate(_FROM_SUPER) for x in exps if x), None)
        digits, dec = re.sub(r"[ ,\\]", "", integer), (decimals or "").replace(" ", "")
        e = int(exp) if exp else 0
        try:
            value, half = float(f"{sign}{digits}.{dec or '0'}") * 10.0**e, 0.5 * 10.0 ** (e - len(dec))
        except OverflowError:  # `1e999`: no number a check could use
            continue
        out.append(Number(value, half, m.start(), m.end(), len(dec), exp is not None))
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


def fix_arithmetic(draft: str, rel_tol: float, prefixes: bool = True, task: str = "", strict: bool = True,
                   max_change: float = 0.0) -> tuple[str, list[dict[str, str]], int]:
    """(the draft with every wrong `expression = result` fixed in place, the fixes, how many were checked).
    A result is wrong only when it is wrong under every reading of its % signs (a% as a/100 or as a, a
    result in % as itself or as a fraction) and, with `prefixes`, of the SI prefixes in its chain (a unit
    with a prefix on an operand, right after the result or after the next `=`: the result may be scaled by
    10^±3, ±6, ±9): off by more than half a unit of its last digit and rel_tol. It is rewritten only when the
    right value is clear; otherwise it is left as written, and so is a number that cannot be read as one.
    With the `task`, a wrong result that is quoted from it (the task writes the same expression with the same
    result, and the result lies in a quotation, see quoted) is left as written and listed with a `note`.
    With `strict`, only whole left sides are read (calc.find_arithmetic) and a wrong result that is a unit
    conversion (calc.converted) is left as written. With `max_change`, a fix that changes the stated number by
    that factor or more, flips its sign or goes to or from zero (at the stated precision), and a division whose
    stated result is the reciprocal of its value, are not applied either and listed with a `note`."""
    decimal_comma = bool(re.search("[А-Яа-яЁё]", draft))  # Russian text: every comma between digits
    fixes, checked = [], 0
    in_task = _written_results(task, decimal_comma, strict) if task else set()
    mask = quoted(draft, task) if task else []
    for a in find_arithmetic(draft, decimal_comma, strict):
        window = _decimal_points(draft[a.result_at : a.result_at + 80], decimal_comma)
        numbers = read_numbers(window)
        if not numbers or numbers[0].start != 0:
            continue
        if re.match(r"(?:\\[,;! ]|[ \xa0  ,.'])\d", window[numbers[0].end :]):
            continue  # "4\,900": digits read_numbers did not join, so not one reading
        n, checked = numbers[0], checked + 1
        in_percent = a.as_percent or bool(re.match(r"\s*\\?%", window[n.end :]))
        stated = [(n.value, n.half)] + ([(n.value / 100, n.half / 100)] if in_percent else [])
        units = _result_units(window[n.end :])
        scaled = prefixes and not in_percent and prefixed(f"{a.expression} {units}")
        scales = _SCALES if scaled else (1.0,)
        if any(abs(v * s - lhs) <= max(half * s, rel_tol * abs(lhs))
               for lhs in (a.fraction, a.plain) for v, half in stated for s in scales):
            continue
        if strict and converted(a.expression, units):
            continue  # "10 Gbit/s × 60 s = 75 GB": a unit conversion, not plain arithmetic
        right = _right_result(a, in_percent)
        if right is not None and scaled:  # the scale meant: the one within a factor of ten of the stated value
            near = [right / s for s in scales if right * n.value > 0 and abs(math.log10(right / s / n.value)) < 1]
            right = near[0] if len(near) == 1 else None
        if right is None:
            continue
        start, end = a.result_at + n.start, a.result_at + n.end
        fix = {"expression": a.expression, "stated": draft[start:end], "fixed": _format(right, n), "at": (start, end)}
        if (a.expression, n.value) in in_task and all(mask[start:end]):
            fix["note"] = "quoted from the task, left as written"
        elif max_change and "/" in a.expression and any(
                lhs and abs(v - 1 / lhs) <= max(half, rel_tol / abs(lhs)) for lhs in (a.fraction, a.plain)
                for v, half in stated):  # "€1224 / €4200 = 3.43 years" meant 4200 / 1224
            fix["note"] = "result is the reciprocal: division written the other way round?"
        elif max_change and (right * n.value < 0 or max(abs(right), abs(n.value))
                             >= max_change * min(abs(right), abs(n.value))
                             or re.fullmatch(r"-?0(?:\.0+)?", fix["fixed"])):  # 0.5 written as "0"
            fix["note"] = "not fixed: change too large to be a slip"
        fixes.append(fix)
    for f in sorted(fixes, key=lambda f: f["at"], reverse=True):
        start, end = f.pop("at")
        if "note" not in f:
            draft = draft[:start] + f["fixed"] + draft[end:]
    return draft, fixes, checked


def _written_results(text: str, decimal_comma: bool, strict: bool) -> set[tuple[str, float]]:
    """(expression, stated result) of every `expression = result` written in the text."""
    out = set()
    for a in find_arithmetic(text, decimal_comma, strict):
        numbers = read_numbers(_decimal_points(text[a.result_at : a.result_at + 80], decimal_comma))
        if numbers and numbers[0].start == 0:
            out.add((a.expression, numbers[0].value))
    return out


def fix_answer_lines(draft: str, fixes: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    """(the draft with each labelled answer line (`A1: …`, as in split_claims) that states exactly the old wrong
    result of a fixed computation, once, given the fixed value too; those fixes, marked `on: answer line`). A
    result fixed to two different values is left alone, and so is one left as written (a `note`: quoted)."""
    by_old: dict[str, list[dict[str, str]]] = {}
    for f in fixes:
        if "note" not in f:
            by_old.setdefault(f["stated"], []).append(f)
    out, done = [], []
    for line in draft.splitlines(keepends=True):
        if _LABELLED.match(line[_MARKERS.match(line).end() :]):
            spans = []
            for old, fs in by_old.items():
                hits = list(re.finditer(rf"(?<![\w.,]){re.escape(old)}(?!\w|[.,]\d)", line))
                if len(hits) == 1 and len({f["fixed"] for f in fs}) == 1:
                    spans.append((hits[0].start(), hits[0].end(), fs[0]))
            for a, b, f in sorted(spans, key=lambda s: s[0], reverse=True):
                line = line[:a] + f["fixed"] + line[b:]
                done.append({"expression": f["expression"], "stated": f["stated"], "fixed": f["fixed"],
                             "on": "answer line"})
        out.append(line)
    return "".join(out), done


def _decimal_points(text: str, decimal_comma: bool) -> str:
    """Commas between digits as decimal points (same length: spans stay valid): every one with
    `decimal_comma` (Russian text), otherwise not before exactly three digits (1,900)."""
    comma = r"(\d),(\d)" if decimal_comma else r"(\d),(\d{1,2}(?!\d)|\d{4,})"
    return re.sub(comma, r"\1.\2", text)


def run_checkers(draft: str, names: list[str], task: str = "", keep_quotes: bool = True,
                 clear_only: bool = True) -> tuple[str, list[dict[str, Any]]]:
    """Run the configured domain checkers (D-034) on the draft and the task; apply their fixes in place, later
    spans first, skipping a fix that overlaps one already applied. With `clear_only`, a fix the checker doubts is
    only reported; with `keep_quotes`, so is a fix whose whole span is quoted from the task. Returns the draft and
    every finding (fixed or only reported)."""
    found = [(name, f) for name in names for f in CHECKERS[name](draft, task)]
    for _, f in found:
        if clear_only and f.doubt and f.fixed is not None:
            f.note, f.fixed = f"left as written: {f.doubt}; {f.note}: {f.fixed}", None
    if task and keep_quotes:
        mask = quoted(draft, task)
        for _, f in found:
            span = [i for i in range(f.start, f.end) if not draft[i].isspace() and draft[i] not in _QUOTE_DROP]
            if f.fixed is not None and all(mask[i] for i in span):
                f.note, f.fixed = f"quoted from the task, left as written; {f.note}: {f.fixed}", None
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


def _property(norm: str, n: Number, v: Value) -> bool | None:
    """Whether the property words nearest the number (before it, else after it) are the value's own (True) or
    another property's (False); None when the claim names neither."""
    pattern = f"(?P<own>{v.words})" + (f"|{v.unlike}" if v.unlike else "")
    found = list(re.finditer(rf"(?<!\w)(?:{pattern})", norm, re.I))
    before = [m for m in found if m.end() <= n.start]
    near = before[-1] if before else next((m for m in found if m.start() >= n.end), None)
    return None if near is None else near["own"] is not None


def _fits(n: Number, v: Value, norm: str, cited: bool, named: bool, units: bool) -> tuple[bool, bool]:
    """(a match of the number with value v supports the claim; a mismatch contradicts it, v's record being cited).
    With `units` both need the same quantity: no other unit after the number and no other property named nearest
    it. Support also needs the record cited or named, or v's unit written when v is its record (a constant). A
    contradiction also needs v's unit written after the number, or v's property (for a constant: its name) named
    and no word after the number, and a number within a factor of ten of v. Without `units`, numbers alone:
    support needs the record cited or named unless v is its record, a contradiction v's unit written or, when v
    is its record, a number within a factor of ten."""
    size = n.value * v.number > 0 and abs(math.log10(n.value / v.number)) < 1
    written, word = read_unit(norm[n.end :])
    record = parse_unit(v.unit)
    unit = None if written is None else record is not None and same_unit(written, record)
    if not units:
        return cited or named or not v.words, (unit is True) if v.words else size
    prop = _property(norm, n, v) if v.words else (named or None)
    same = unit is not False and prop is not False
    return (same and (cited or named or (unit is True and not v.words)),
            same and size and (unit is True or (prop is True and not word)))


def code_check(claim: Claim, sources: Sources, rel_tol: float,
               units: bool = True) -> tuple[str, list[str], str | None, str] | None:
    """(outcome, ids, fixed text, note) from the structured records, or None when code cannot decide. With
    `units`, a number counts only as the same quantity as a record value (see _fits)."""
    norm = claim.text.translate(_NORM)
    nums, forms = read_numbers(claim.text), read_formulas(claim.text)
    cited = [s for i in claim.cites if (s := sources.get(i))]
    named = {s.id: _mentions(norm, s.record.names()) for s in sources.items}
    says_formula = "formula" in norm.lower()

    for s in cited:  # contradictions need a cited record and the same quantity
        fixes = []
        for n in nums:
            if any(_matches(n, v.number, rel_tol) and _fits(n, v, norm, True, named[c.id], units)[0]
                   for c in cited for v in c.record.values()):
                continue  # the value of a cited record
            for v in s.record.values():
                if _fits(n, v, norm, True, named[s.id], units)[1]:
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
        hit = _support(n, cited, sources, norm, rel_tol, named, units)
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


def _support(n: Number, cited: list[Source], sources: Sources, norm: str, rel_tol: float,
             named: dict[str, bool], units: bool) -> tuple[str, float] | None:
    for s in [*cited, *(x for x in sources.items if x not in cited)]:
        for v in s.record.values():
            if _matches(n, v.number, rel_tol) and _fits(n, v, norm, s in cited, named[s.id], units)[0]:
                return s.id, v.number
    return None


# A unit right after a number (normalised text): % or a word, maybe after ° and joined by / or ·, each part maybe
# with a power (kg/m3, J·K-1, °C). A stopword after a number is not a unit.
_UNIT = re.compile(r"[ \t]*(%|°?[^\W\d_]+(?:-?\d)?(?:[/·⋅∙*][^\W\d_]+(?:-?\d)?)*)")


def _quantities(text: str) -> list[tuple[Number, str]]:
    """Every number of the text that a unit follows, with that unit (normalised)."""
    norm = text.translate(_NORM)
    return [(n, m[1]) for n in read_numbers(text)
            if (m := _UNIT.match(norm, n.end)) and m[1].lower() not in STOPWORDS]


def reference_conflict(claim: Claim, sources: Sources, rel_tol: float) -> str:
    """'' or why a claim that only literature passages support (claim.ids) is not supported: a reference record
    retrieved for the answer (any source that is not literature) gives another value for one of its quantities (a
    number with a unit). The record gives that quantity in a line or sentence of its passage with a number in the
    same unit within a factor of two, when that line or sentence with the record's title holds at least half the
    claim's words (units aside, see term_share); another value: none of those values matches the claim's number."""
    supporting = [s for i in claim.ids if (s := sources.get(i))]
    if not supporting or not all(s.literature for s in supporting):
        return ""
    text = plain(claim.text)
    quantities = _quantities(text)
    units = {w for _, u in quantities for w in _words(u)}
    words = [w for w in _words(text) if w not in units and w not in STOPWORDS]
    if not words:
        return ""
    query = " ".join(words)
    segments = [(s, seg) for s in sources.items if not s.literature
                for seg in (line[a:b] for line in s.passage.splitlines() for a, b in _sentences(line))
                if term_share(query, f"{s.record.label()} {seg}") >= 0.5]
    found = []
    for n, unit in quantities:
        values = [(s, seg, m) for s, seg in segments for m, u in _quantities(seg)
                  if u == unit and n.value and 0.5 <= m.value / n.value <= 2]
        if values and not any(abs(n.value - m.value) <= max(n.half, m.half, rel_tol * abs(m.value))
                              for _, _, m in values):
            s, seg, m = values[0]
            found.append(f"{text[n.start:n.end]} {unit}, but reference {s.id} gives {seg[m.start:m.end]} {unit}")
    return "; ".join(found)


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
    """The consistency review's OLD/NEW/CONFLICT/WHY blocks (an `OK` reply has none). Markdown emphasis around
    the keys is ignored; a block needs an OLD and a NEW line."""
    blocks: list[dict] = []
    for line in text.splitlines():
        m = re.match(r"^\W*(OLD|NEW|CONFLICT|WHY)\W*?:\s?(.*)$", line.strip(), re.I)
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


def _words(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", text.casefold())


def _answer_line(text: str, start: int, end: int) -> bool:
    """A line that text[start:end] touches is a labelled answer or verdict line (`A1: …`, as in split_claims)."""
    stop = text.find("\n", end)
    lines = text[text.rfind("\n", 0, start) + 1 : stop if stop >= 0 else len(text)].splitlines()
    return any(_LABELLED.match(line[_MARKERS.match(line).end() :]) for line in lines)


def answer_labels(text: str) -> set[str]:
    """The labels of the labelled answer or verdict lines in text (`**A1:** …` -> `A1`, as in split_claims)."""
    found = (_LABELLED.match(line, _MARKERS.match(line).end()) for line in text.splitlines())
    return {m.group(0).strip("*_`:： \t") for m in found if m}


def unconfirmed(answer: str, fix: dict[str, str]) -> str:
    """Why a review fix whose OLD text occurs once in the answer does not resolve a conflict inside the answer,
    or "" when it does: its CONFLICT quote is elsewhere in the answer, NEW adds something, every number NEW adds
    is stated in the quote (within the rounding of either) or is the value of a computation written there, and
    every word NEW adds is in the quote too when it adds no number or OLD is on a labelled answer/verdict line."""
    old, new, quote = fix["old"], fix["new"], fix.get("conflict", "")
    at = answer.find(old)
    if not quote:
        return "no conflicting passage quoted"
    if quote not in f"{answer[:at]}\0{answer[at + len(old) :]}":
        return "conflicting passage not elsewhere in the answer"
    labelled = _answer_line(answer, at, at + len(old))
    if labelled and not _answer_line(f"{answer[:at]}{new}{answer[at + len(old) :]}", at, at + len(new)):
        return "the answer line would lose its label"
    comma = bool(re.search("[А-Яа-яЁё]", answer))  # Russian text: every comma between digits is a decimal comma
    stated, computed = read_numbers(_decimal_points(quote, comma)), computations(quote, comma)
    had = {n.value for n in read_numbers(_decimal_points(old, comma))}
    numbers = [n for n in read_numbers(_decimal_points(new, comma)) if n.value not in had]
    for n in numbers:
        if not (any(abs(n.value - s.value) <= max(n.half, s.half) for s in stated)
                or any(abs(n.value - v) <= n.half for v in computed)):
            return f"{new[n.start : n.end]} is not stated in the quoted passage"
    words = [w for w in _words(new) if w not in set(_words(old))]
    if not numbers and not words:
        return "it adds nothing from the quoted passage"
    missing = [w for w in words if w not in set(_words(quote))]
    if missing and (labelled or not numbers):
        return f"{'answer line' if labelled else 'no value changed'}: {', '.join(missing)} not in the quoted passage"
    return ""


_EDIT_TERMS = 3  # terms one edit of a judge correction may replace and put in: a value with its unit, a short name
_STEM = 4  # letters of the shorter word for two words, one inside the other, to be one word with an affix


def _terms(text: str) -> list[tuple[str, int, int, Number | None]]:
    """(key, start, end, number) of the numbers and other terms of the text in order, citation markers aside: a
    number keyed by its value, any other run of letters and digits (a word, a formula such as H2O, a unit symbol)
    by its casefolded text. Commas between digits are decimal points as in unconfirmed."""
    blank = _CITES.sub(lambda m: " " * len(m.group()), text)
    points = _decimal_points(blank, bool(re.search("[А-Яа-яЁё]", text)))
    norm, numbers = points.translate(_NORM), read_numbers(points)  # read_numbers keeps the superscripts
    out = [(f"#{n.value!r}", n.start, n.end, n) for n in numbers]
    for n in numbers:
        norm = norm[: n.start] + " " * (n.end - n.start) + norm[n.end :]
    out += [(m.group().casefold(), m.start(), m.end(), None) for m in re.finditer(r"\w+", norm)]
    return sorted(out, key=lambda t: t[1])


def minimal_edit(text: str, correction: str, passages: str) -> tuple[str | None, str]:
    """The claim `text` with a judge correction put in as a minimal edit, or None and why not. Claim and correction
    are compared as terms (see _terms); each place they differ is an edit. Minimal: at least one edit, each replacing
    1 to _EDIT_TERMS terms by 1 to _EDIT_TERMS terms (nothing only added or dropped: an explanation, a `not`) and no
    word by one that contains it or that it contains (balanced → unbalanced: a changed verdict). Stated: every term
    an edit puts in is in the passages, a number within half a unit of the last written digit of either. Only the
    edited terms change: the rest of the claim, its emphasis and citation markers stay as written."""
    old, new, said = _terms(text), _terms(correction), _terms(passages)
    stated, words = [n for *_, n in said if n], {key for key, *_, n in said if not n}
    edits = [op for op in SequenceMatcher(None, [t[0] for t in old], [t[0] for t in new], autojunk=False).get_opcodes()
             if op[0] != "equal"]
    if not edits:
        return None, "not a minimal edit: it changes no value or term"
    for _, i1, i2, j1, j2 in edits:
        was = text[old[i1][1] : old[i2 - 1][2]] if i2 > i1 else ""
        put = correction[new[j1][1] : new[j2 - 1][2]] if j2 > j1 else ""
        if not was or not put:
            return None, f"not a minimal edit: {f'adds {put!r}' if put else f'drops {was!r}'}"
        if max(i2 - i1, j2 - j1) > _EDIT_TERMS:
            return None, f"not a minimal edit: rewrites {was!r} as {put!r}"
        for w in (t[0] for t in old[i1:i2] if t[0].isalpha()):
            for v in (t[0] for t in new[j1:j2] if t[0].isalpha()):
                if w != v and min(len(w), len(v)) >= _STEM and (w in v or v in w):
                    return None, f"not a minimal edit: changes {w!r} to {v!r}"
        for key, a, b, n in new[j1:j2]:
            if not (any(abs(n.value - s.value) <= max(n.half, s.half) for s in stated) if n else key in words):
                return None, f"not stated in the source: {correction[a:b]}"
    out = text
    for _, i1, i2, j1, j2 in reversed(edits):
        out = out[: old[i1][1]] + correction[new[j1][1] : new[j2 - 1][2]] + out[old[i2 - 1][2] :]
    return out, ""


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


_ECHO_RUN = 5  # words in a row shared with the judge's instructions that make a correction an echo of them


def echoes(text: str, instructions: str) -> bool:
    """The text repeats the instructions: all its words, or a run of five of them, appear in a row in them
    (case ignored; ids such as S1 or C2 count as one same word, so `agrees with S3` repeats `agrees with S1`)."""

    def words(s: str) -> list[str]:
        return re.findall(r"\w+", re.sub(r"\b[SC]\d+\b", "ID", s).lower())

    said, given = words(text), f" {' '.join(words(instructions))} "
    runs = [said[i : i + _ECHO_RUN] for i in range(max(1, len(said) - _ECHO_RUN + 1))]
    return bool(said) and any(f" {' '.join(run)} " in given for run in runs)


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
        arithmetic, checked, lines = [], 0, []
        if self.cfg.arithmetic_check:
            draft, arithmetic, checked = fix_arithmetic(draft, self.cfg.arithmetic_rel_tol,
                                                        self.cfg.arithmetic_prefixes,
                                                        question if self.cfg.keep_quotes else "",
                                                        self.cfg.arithmetic_strict_parse,
                                                        self.cfg.arithmetic_max_change)
        kept = [a for a in arithmetic if "note" in a]  # quoted from the task: only in the trace
        arithmetic = [a for a in arithmetic if "note" not in a]
        if self.cfg.arithmetic_answer_lines and arithmetic:
            draft, lines = fix_answer_lines(draft, arithmetic)
        if checked:
            yield _say(f"\n\nArithmetic: {checked} calculation{'s' if checked != 1 else ''} recomputed,"
                       f" {len(arithmetic)} fixed\n"
                       + "".join(f"- {a['expression']} = {a['stated']} → {a['fixed']}\n" for a in arithmetic)
                       + "".join(f"- {a['expression']} = {a['stated']}: {a['note']} (would be {a['fixed']})\n"
                                 for a in kept)
                       + "".join(f"- answer line: {a['stated']} → {a['fixed']} (as {a['expression']})\n"
                                 for a in lines))
        arithmetic += lines
        draft, checks = run_checkers(draft, self.cfg.checkers, question, self.cfg.keep_quotes,
                                     self.cfg.checkers_clear_only)
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
        mask = quoted(draft, question) if self.cfg.keep_quotes else []
        for c in claims:
            if is_quotation(draft, mask, c.start, c.end):  # before the code checks: they would rewrite it too
                c.outcome, c.note = UNVERIFIED, "quoted from the task: not checked"
                yield _say(f"- C{c.n} {c.outcome} ({c.note})\n")
            elif found := code_check(c, sources, self.cfg.rel_tol, self.cfg.unit_checks):
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
        if self.cfg.reference_over_literature:
            for c in claims:
                if c.outcome == SUPPORTED and (why := reference_conflict(c, sources, self.cfg.rel_tol)):
                    c.outcome, c.ids, c.note = UNVERIFIED, [], f"{c.note}; only literature supports {why}"
                    yield _say(f"- C{c.n} {c.outcome} (only literature supports {why})\n")
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
            if count != 1 or f["new"] == f["old"]:
                why = "not in the answer" if count == 0 else "unchanged" if count == 1 else f"found {count} times"
            elif self.cfg.keep_quotes and rewrites_quote(answer, question, f["old"], f["new"]):
                why = "quoted from the task"
            elif self.cfg.consistency_keep_answer_lines and _answer_line(
                    answer, at := answer.find(f["old"]), at + len(f["old"])):
                why = f"answer line kept; conflict: {f.get('conflict') or f['why'] or 'not named'!r}"
            else:
                why = unconfirmed(answer, f) if self.cfg.consistency_quotes else ""
            if why:
                yield _say(f"- skipped {f['old']!r} ({why})\n")
                continue
            answer = answer.replace(f["old"], f["new"])
            applied.append(f)
            yield _say(f"- {f['old']!r} → {f['new']!r}" + (f" ({f['why']})" if f["why"] else "") + "\n")
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
            if (verdict == "contradicts" and correction and self.cfg.ignore_prompt_echo
                    and echoes(correction, self.cfg.judge_prompt)):
                correction, c.note = None, f"{c.note}, correction repeats the judge instructions: ignored"
            fixed, why = correction, ""
            if verdict == "contradicts" and correction and self.cfg.judge_minimal_edits:
                fixed, why = minimal_edit(c.text, correction, "\n".join(
                    f"{s.record.label()}\n{s.passage}" for s in passages if s.id in ids))
            if verdict == "supports":
                c.outcome, c.ids = SUPPORTED, ids
            elif why:  # a rewrite, or a value no passage states: the claim stays as written
                c.outcome, c.note = UNVERIFIED, f"{c.note}, correction not applied ({why}): {correction}"
            elif verdict == "contradicts" and not correction and self.cfg.keep_uncorrected:
                c.outcome, c.note = UNVERIFIED, f"{c.note}, no correction: kept"  # no value stated against it
            elif verdict == "contradicts":
                c.outcome, c.ids, c.fixed = (CORRECTED, ids, fixed) if fixed else (REMOVED, ids, None)
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
