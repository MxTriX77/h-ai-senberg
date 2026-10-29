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
from collections import Counter
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

import httpx

from .config import Config, StatusWords
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
_NUM = re.compile(
    r"(?<![\w.^/\\-])(-?)(\d{1,3}(?:(?:[ ,]|\\,)\d{3})+(?!\d)|\d+)"  # sign, integer part (maybe grouped in threes)
    r"(?:\.(\d+(?: \d{1,3}(?!\d))*(?:(?<=[. ]\d{3}) \d{4}(?!\d))?))?"  # decimals (maybe grouped, CODATA style:
    # threes, the last group maybe four: 9.109 383 7139)
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
_LABEL = r"(?:(?:[^\W\d_]+\s+)?\(?[^\W\d_]\d{0,2}\)?|[^\W\d_]+\s+\d{1,2})"
_LABELLED = re.compile(rf"[*_`]*{_LABEL}[*_`\s]*[:：][*_`]*(?:\s|$)")
# An answer line the task asks for: a label, a colon and a placeholder in angle brackets (`A9: <verdict>`), anywhere
# in the text; a placeholder starts with no space, digit, -, = or bracket (`A: <0.1 %`, `A: <-> B` are no placeholders).
_REQUESTED = re.compile(rf"(?<!\w)[*_`]*({_LABEL})[*_`\s]*[:：][*_`\s]*(<(?![\s\d<>=-])[^<>\n]+>)")
_ID = re.compile(r"\(?[^\W\d_]\d{0,2}\)?")  # the id of a label (A1, C, (b)), without a word before it
_LOOKALIKE = str.maketrans("АВСЕНКМОРТХавсенкмортх", "ABCEHKMOPTXabcehkmoptx")  # Cyrillic letters seen in labels
_ITEM = re.compile(r"(?<!\w)[^\W\d_]+\d+(?!\w)")  # an item id: letters, then digits (C3, У2, Q1)
_ITEM_START = re.compile(r"[*_`]*([^\W\d_]+\d+)[*_`]*[.):：][*_`]*(?:\s|$)")  # a list item with one: C3. / У2) / S4:
_LEAD = re.compile(r"[#*>\s-]*")  # Markdown markers before a section's item id (`### C3.`, `- **У2)**`)
# A line label and its value (`**Verdict:** Incorrect — …`): up to three words, the first starting with a letter.
_FIELD = re.compile(r"[#*>\s_`-]*([^\W\d_][^\W_]*(?: [^\W_]+){0,2})[*_`\s]*[:：](.*)")
_REASON = re.compile(r" [—–-] |[.;,] | \(")  # where a verdict's reason starts: ` — `, `. `, `, `, ` (` …
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


def _label_id(label: str) -> str:
    """A label's id: its last word when that is an id (`Claim C` -> `C`, `with A1` -> `A1`), else the label."""
    words = label.split()
    return words[-1] if _ID.fullmatch(words[-1]) else " ".join(words)


def _label_key(label: str) -> str:
    """A label as compared: its id, case and Cyrillic look-alike letters ignored (`А1` = `a1`, `(b)` = `b`)."""
    return _label_id(label).translate(_LOOKALIKE).casefold().strip("()")


def missing_lines(task: str, answer: str) -> dict[str, str]:
    """The answer lines the task asks for (_REQUESTED) that have no labelled line in the answer (answer_labels):
    label id -> the line as the task writes it, with that id (`A9: <verdict>`), in the task's order."""
    seen = {_label_key(label) for label in answer_labels(answer)}
    missing: dict[str, str] = {}
    for m in _REQUESTED.finditer(task):
        if (key := _label_key(m.group(1))) not in seen:
            seen.add(key)
            label = _label_id(m.group(1))
            missing[label] = f"{label}: {m.group(2)}"
    return missing


def supplied_lines(reply: str, labels: Iterable[str]) -> dict[str, str]:
    """The reply's labelled lines (as in answer_labels) for these labels, the first per label, each with text
    after the colon that is not only a placeholder (`<…>`): label -> the line."""
    wanted = {_label_key(label): label for label in labels}
    found: dict[str, str] = {}
    for line in reply.splitlines():
        m = _LABELLED.match(line, _MARKERS.match(line).end())
        label = wanted.get(_label_key(m.group(0).strip("*_`:： \t"))) if m else None
        text = line[m.end() :].strip() if m else ""
        if label and label not in found and text and not (text.startswith("<") and text.endswith(">")):
            found[label] = line.strip()
    return found


def item_lines(task: str) -> dict[str, str]:
    """The answer lines the task asks for (_REQUESTED) whose placeholder names exactly one enumerated item of the
    task (`A4: <verdict for C1>`): an id of letters and digits that also starts a line of the task as a list item
    (`C1.`, `**У2)**`, `S4:`) and is no requested label itself. Label id -> the item id as the task's list writes
    it, in the task's order; {} unless the task asks for at least two such lines."""
    requested = [(_label_id(m.group(1)), m.group(2)) for m in _REQUESTED.finditer(task)]
    labels = {_label_key(label) for label, _ in requested}
    items: dict[str, str] = {}
    for line in task.splitlines():
        if (m := _ITEM_START.match(line, _MARKERS.match(line).end())) and _label_key(m.group(1)) not in labels:
            items.setdefault(_label_key(m.group(1)), m.group(1))
    table: dict[str, str] = {}
    for label, placeholder in requested:
        named = {_label_key(t) for t in _ITEM.findall(placeholder)} & items.keys()
        if len(named) == 1 and label not in table:
            table[label] = items[named.pop()]
    return table if len(table) >= 2 else {}


def _lines(text: str) -> list[tuple[str, str, int]]:
    """Each line of text: (its label key as in answer_labels, "" when unlabelled; the line; where its value starts)."""
    out = []
    for line in text.splitlines():
        m = _LABELLED.match(line, _MARKERS.match(line).end())
        out.append((_label_key(m.group(0).strip("*_`:： \t")), line, m.end()) if m else ("", line, 0))
    return out


def anchor_lines(task: str, answer: str, reply: str, table: dict[str, str]) -> tuple[str, list[str]]:
    """The answer with its lines for the labels of table (label -> item id, item_lines) set to the reply's value
    for that label, and the changes (`A4 old → new (item C1)`). A line changes only when the reply has its label
    on exactly one line with a value that is not a placeholder, the answer has it on exactly one line with another
    value, and the item id occurs in the answer outside its requested lines (the lines block). Nothing else changes."""
    requested = {_label_key(m.group(1)) for m in _REQUESTED.finditer(task)}
    lines = _lines(answer)
    named = {_label_key(t) for key, line, _ in lines if key not in requested for t in _ITEM.findall(line)}
    counts = Counter(key for key, _, _ in lines)
    values: dict[str, list[str]] = {}
    for key, line, at in _lines(reply):
        values.setdefault(key, []).append(line[at:].strip())
    wanted = {_label_key(label): (label, item) for label, item in table.items()}
    out, changes = [], []
    for key, line, at in lines:
        label, item = wanted.get(key, ("", ""))
        given = values.get(key, [])
        new, old = given[0] if len(given) == 1 else "", line[at:].strip()
        if (label and counts[key] == 1 and new and not (new.startswith("<") and new.endswith(">"))
                and _label_key(item) in named and _value_words(new) != _value_words(old)):
            line = f"{line[:at].rstrip()} {new}"
            changes.append(f"{label} {old} → {new} (item {item})")
        out.append(line)
    return ("\n".join(out), changes) if changes else (answer, [])


def _value_words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.casefold())


def _verdict(value: str) -> str:
    """A value up to its first separator (_REASON), trimmed of emphasis: `**Incorrect** — 4 g` -> `Incorrect`."""
    return _REASON.split(value, maxsplit=1)[0].strip(" \t*_`.")


def _sections(task: str, lines: list[tuple[str, str, int]],
              ends: Callable[[str], bool] | None = None) -> dict[str, list[int]]:
    """Item key -> the indices in lines (_lines of an answer) of its sections: a section starts at a line that begins,
    after _LEAD, with an item the task lists (_ITEM_START) and ends at the next such line, at a requested line or at
    a line that `ends` accepts."""
    requested = {_label_key(m.group(1)) for m in _REQUESTED.finditer(task)}
    listed = {_label_key(m.group(1)) for line in task.splitlines()
              if (m := _ITEM_START.match(line, _MARKERS.match(line).end())) and _label_key(m.group(1)) not in requested}
    sections: dict[str, list[int]] = {}
    item = ""
    for i, (key, line, _) in enumerate(lines):
        start = _ITEM_START.match(line, _LEAD.match(line).end())
        if key in requested:
            item = ""
        elif start and _label_key(start.group(1)) in listed:
            item = _label_key(start.group(1))
        elif ends and ends(line):
            item = ""
        if item:
            sections.setdefault(item, []).append(i)
    return sections


def _fields(task: str, lines: list[tuple[str, str, int]],
            sections: dict[str, list[int]]) -> dict[str, list[tuple[int, str, str]]]:
    """Item key -> (line index, label key, verdict) of each line label in its sections: _FIELD, no requested label,
    a value that _verdict cuts to words."""
    requested = {_label_key(m.group(1)) for m in _REQUESTED.finditer(task)}
    fields: dict[str, list[tuple[int, str, str]]] = {}
    for item, found in sections.items():
        fields[item] = []
        for i in found:
            field = _FIELD.match(lines[i][1])
            label = _label_key(field.group(1)) if field else ""
            if field and label not in requested and _value_words(value := _verdict(field.group(2))):
                fields[item].append((i, label, value))
    return fields


def _verdict_label(fields: dict[str, list[tuple[int, str, str]]]) -> str:
    """The line label found exactly once in each of the most sections, at least two; "" when none or on a tie."""
    counts = [Counter(label for _, label, _ in found) for found in fields.values()]
    ranked = Counter(label for c in counts for label, n in c.items() if n == 1).most_common(2)
    if not ranked or ranked[0][1] < 2 or (len(ranked) == 2 and ranked[1][1] == ranked[0][1]):
        return ""
    return ranked[0][0]


def sync_lines(task: str, answer: str) -> tuple[str, list[str]]:
    """The answer with its lines for the labels of item_lines(task) set to the verdict of their item's own section,
    and the changes (`A4 неверно → верно (item У1)`). A section starts at a line that begins, after _LEAD, with an
    item the task lists (_ITEM_START) and ends at the next such line or at a requested line. The verdict label is the
    line label (_FIELD, no requested label, a value) found exactly once in each of the most sections, at least two
    (a tie: nothing changes); a section's verdict is that line's value cut by _verdict. A line changes only when its
    value, cut the same way, is the verdict of some section and has other words than its item's verdict."""
    table = item_lines(task)
    if not table:
        return answer, []
    lines = _lines(answer)
    fields = _fields(task, lines, _sections(task, lines))
    if not (verdict := _verdict_label(fields)):
        return answer, []
    verdicts = {item: values[0] for item, found in fields.items()
                if len(values := [v for _, k, v in found if k == verdict]) == 1}
    known = [_value_words(v) for v in verdicts.values()]
    wanted = {_label_key(label): (label, item) for label, item in table.items()}
    out, changes = [], []
    for key, line, at in lines:
        if key in wanted:
            label, item = wanted[key]
            old, new = line[at:].strip(), verdicts.get(_label_key(item), "")
            if new and _value_words(_verdict(old)) in known and _value_words(_verdict(old)) != _value_words(new):
                line = f"{line[:at].rstrip()} {new}"
                changes.append(f"{label} {old} → {new} (item {item})")
        out.append(line)
    return ("\n".join(out), changes) if changes else (answer, [])


_STATUSES = ("unverified", "negative", "positive")  # verdict classes of StatusWords, in the order they are tested
_CLAUSE_END = re.compile(r"[.!?;](?=\s|$)|\)|\n")  # where a concession's clause ends: its sentence or parenthesis


def check_status(task: str, answer: str, words: StatusWords) -> tuple[str, list[str]]:
    """The answer with the verdict word of its lines for the labels of item_lines(task) changed where the line's own
    reason supports another status, and the changes (`A4 неверно → не подтверждено (item У4, R1)`). The reason is
    the item's section (as in sync_lines, also ended by a `---` line, a heading or a bold `summary` block that is not
    the verdict label's line) and the line's text after its verdict word; the claim is the item's text in the task's
    list. The verdict (the line's value cut by _verdict) has the first class of _STATUSES whose words start it. Rules,
    the first that applies:
    R1 negative, a `short` reason, a claim without `certainty` -> unverified;
    R2 unverified, a `short` reason without `missing`, a claim with `certainty` and without `document` -> negative;
    R3 positive, a reason with `calculation` and a `concession` whose clause has a `premise` word (_conceded) ->
    negative.
    The new word is a word of that class the task names (outside its list): the one the answer itself uses most on
    its item lines and its sections' verdict label lines, else of the task's, the one of fewest words, then the one
    nearest a word of another class (as in `Correct / Incorrect / Unverified`); without one nothing changes. It
    replaces the verdict word on the line and on the section's verdict label line (sync_lines' label, once in the
    section, a verdict of the same class). A line changes only when its label is on one line and its item has a
    section and a claim."""
    table = item_lines(task)
    if not table:
        return answer, []
    alt = {name: "|".join(f"(?:{f})" for f in fragments) for name, fragments in words.model_dump().items()}
    find = {name: re.compile(pattern, re.I) for name, pattern in alt.items()}
    heads = {name: re.compile(rf"\W*({alt[name]})(?!\w)", re.I) for name in _STATUSES}
    requested = {_label_key(m.group(1)) for m in _REQUESTED.finditer(task)}
    claims: dict[str, str] = {}
    said = []  # the task without its list of items, whose claims may use verdict words
    for line in task.splitlines():
        m = _ITEM_START.match(line, _MARKERS.match(line).end())
        if m and _label_key(m.group(1)) not in requested:
            claims.setdefault(_label_key(m.group(1)), line[m.end():].strip())
        else:
            said.append(line)
    text, spans = "\n".join(said), []
    for name in _STATUSES:  # each word found is blanked for the next class: `not supported` holds no `supported`
        anywhere = re.compile(rf"(?<!\w)(?:{alt[name]})(?!\w)", re.I)
        spans += [(m.start(), m.end(), name, m.group(0)) for m in anywhere.finditer(text)]
        text = anywhere.sub(lambda m: " " * len(m.group(0)), text)
    named = {}  # class -> the task's word for it: fewest words, then nearest a word of another class, then first
    for name in _STATUSES:
        ranked = [(len(a[3].split()), min((max(b[0] - a[1], a[0] - b[1]) for b in spans if b[2] != name),
                                          default=len(task)), a[0], a[3]) for a in spans if a[2] == name]
        if ranked:
            named[name] = min(ranked)[3]
    lines = _lines(answer)
    label = _verdict_label(_fields(task, lines, _sections(task, lines)))
    boundary = re.compile(rf"\s*(?:-{{3,}}\s*$|#{{1,6}}\s|(?:[-*+]\s+)?\*\*\s*(?:{alt['summary']}))", re.I)
    sections = _sections(task, lines, lambda line: bool(boundary.match(line)) and not (
        label and (f := _FIELD.match(line)) and _label_key(f.group(1)) == label))
    wanted, inside = {_label_key(name) for name in table}, {j for found in sections.values() for j in found}
    used: Counter[tuple[str, str]] = Counter()  # (class, word) on the item lines and the sections' verdict lines
    for j, (key, line, at) in enumerate(lines):
        f = _FIELD.match(line) if label and j in inside else None
        value = line[at:] if key in wanted else f.group(2) if f and _label_key(f.group(1)) == label else None
        if value is not None and (status := _status(_verdict(value), heads))[0]:
            used[status] += 1
    for name in _STATUSES:  # the answer's own word first, when the task names it
        given = {a[3].casefold() for a in spans if a[2] == name}
        if own := [w for (c, w), _ in used.most_common() if c == name and w.casefold() in given]:
            named[name] = own[0]
    counts = Counter(key for key, _, _ in lines)
    out, changes = [line for _, line, _ in lines], []
    for name, item in table.items():
        key, claim, found = _label_key(name), claims.get(_label_key(item), ""), sections.get(_label_key(item))
        if counts[key] != 1 or not claim or not found:
            continue
        i = next(i for i, (k, _, _) in enumerate(lines) if k == key)
        line, at = out[i], lines[i][2]
        old, word = _status(_verdict(line[at:]), heads)
        start = line.find(word, at)
        reason = "\n".join(lines[j][1] for j in found) + "\n" + line[start + len(word):]
        short, certain = find["short"].search(reason), find["certainty"].search(claim)
        if old == "negative" and short and not certain:
            rule, new = "R1", named.get("unverified")
        elif (old == "unverified" and short and certain and not find["missing"].search(reason)
              and not find["document"].search(claim)):
            rule, new = "R2", named.get("negative")
        elif old == "positive" and find["calculation"].search(reason) and _conceded(reason, find):
            rule, new = "R3", named.get("negative")
        else:
            continue
        if not new:
            continue
        out[i] = line[:start] + new + line[start + len(word):]
        verdict_lines = [j for j in found if label and (f := _FIELD.match(out[j])) and _label_key(f.group(1)) == label]
        if len(verdict_lines) == 1:
            j = verdict_lines[0]
            field = _FIELD.match(out[j])
            was, given = _status(_verdict(field.group(2)), heads)
            if was == old:
                s = out[j].find(given, field.start(2))
                out[j] = out[j][:s] + new + out[j][s + len(given):]
        changes.append(f"{name} {word} → {new} (item {item}, {rule})")
    return ("\n".join(out), changes) if changes else (answer, [])


def _conceded(reason: str, find: dict[str, re.Pattern[str]]) -> bool:
    """The reason has a `concession` whose clause, from it to the end of its sentence or parenthesis (_CLAUSE_END),
    has a `premise` word (`correct despite the wrong input`, `accurate (though it uses …)`)."""
    for m in find["concession"].finditer(reason):
        end = _CLAUSE_END.search(reason, m.end())
        if find["premise"].search(reason, m.start(), end.start() if end else len(reason)):
            return True
    return False


def _status(verdict: str, heads: dict[str, re.Pattern[str]]) -> tuple[str, str]:
    """The first class of heads whose pattern starts the verdict, and the verdict word it matched; ("", "") if none."""
    for name, head in heads.items():
        if m := head.match(verdict):
            return name, m.group(1)
    return "", ""


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
    norm = _decimal_points(blank, bool(re.search("[А-Яа-яЁё]", text))).translate(_NORM)
    numbers = read_numbers(norm)
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
