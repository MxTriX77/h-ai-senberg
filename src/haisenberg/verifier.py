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
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

import httpx

from .config import Config, VerdictWords
from .calc import Arithmetic, computations, converted, find_arithmetic, prefixed
from .checkers import CHECKERS
from .l1.dense import non_latin
from .l1.store import STOPWORDS, Value
from .l1.text import parse_formula, to_hill
from .l1.units import parse_unit, read_unit, same_unit
from .llm import LLMClient, UpstreamError, estimate_tokens
from .stages import CHECK, MODEL, stage
from .tools import SEARCH, Knowledge, Source, Sources, term_share

log = logging.getLogger(__name__)

SUPPORTED, UNVERIFIED, CORRECTED, REMOVED, NO_CLAIM = "supported", "unverified", "corrected", "removed", "no_claim"
OUTCOMES = (SUPPORTED, UNVERIFIED, CORRECTED, REMOVED)
# What the evidence gave for a claim (verifier.support_quotes), next to its outcome: a source states it (code found
# the quote or the record's value), a quoted passage states a part of it, a source states another value, a passage is
# about its subject and none states it, no passage is about its subject (or its re-search found nothing), nothing
# compared it with a source.
PARTLY, CONTRADICTED, NOT_STATED = "partly", "contradicted", "not_stated"
NOT_FOUND, NOT_CHECKED = "not_found", "not_checked"
STATUSES = (SUPPORTED, PARTLY, CONTRADICTED, NOT_STATED, NOT_FOUND, NOT_CHECKED)
_OPEN = (PARTLY, NOT_STATED, NOT_FOUND, NOT_CHECKED)  # the statuses of an unsettled claim; an earlier one says more
NO_VERDICT = "judge: no verdict"  # the note of a claim that a judge reply has no line for

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
# The verdicts of verifier.support_prompt (verifier.support_quotes).
_QUOTED_VERDICT = re.compile(r"^\W*C(\d+)\W+(supports|partly|contradicts|not[_ ]stated|not[_ ]found|no[_ ]claim)\b(.*)$",
                             re.I)
# The same verdicts with the evidence first, `C1: S2 "<words of S2>" -> supports` (also `=>`, `→`): the passage
# ids, the words copied, then the verdict; after a `contradicts`, `|` and the correction.
_QUOTE_FIRST = re.compile(r"^\W*C(\d+)\W+(.*?)\s*(?:->|=>|→)\W*"
                          r"(supports|partly|contradicts|not[_ ]stated|not[_ ]found|no[_ ]claim)\b(.*)$", re.I)
_BOLD_LINE = re.compile(r"(\*\*|__)((?:(?!\1).)+)\1:?")  # a line that is one bold run: a heading written as text
# A label (an id such as A1, C or (b), maybe after one word: "Claim C", "Claim 3") and a colon starts the line.
_LABELLED = re.compile(r"[*_`]*(?:(?:[^\W\d_]+\s+)?\(?[^\W\d_]\d{0,2}\)?|[^\W\d_]+\s+\d{1,2})[*_`\s]*[:：][*_`]*(?:\s|$)")
# An equation, comparison or operation with numbers: a computation, not a sourceable fact.
_COMPUTED = re.compile(r"[=≈<>≤≥]|\d\s*(?:[×*/÷·⋅]|\\times|\\cdot)\s*[-−(]?\s*\d|\\frac")
# Written arithmetic (verifier.computation_needs_arithmetic): two numbers joined by an operator. A unit or a couple
# of words may stand between a number and its operator (`0.046 mol × 130.19 g/mol`). After a unit, a slash, a middle
# dot or a plus is an operator only between spaces (`54.0 g / 20.0 cm³`; the slash of `g/mol` is not one), and so is
# a hyphen or dash anywhere (`70–80 °C` and `1-2 mL` are ranges). The `}{` of a LaTeX fraction divides. Powers and
# bold marks are taken out first (_bare). _RESULT_SIGN: a result sign with its number; _FREE: a number that is not part of
# a name (`R1`, `x2`).
_GAP = r"[^\d=≈;:!?\n]{0,24}?"
_JOINED = re.compile(r"\d" + _GAP + r"(?:[×*÷⋅]|\\times|\\cdot|\}\s*\{|(?<=\s)[·/+](?=\s)|(?<= )[-−–](?= ))" + _GAP + r"\d"
                     r"|\d\s*[/·+]\s*[-−(]?\s*\d")
_RESULT_SIGN = re.compile(r"(?:=|≈|\\approx)[^\d=≈.;!?\n]{0,16}?\d")
_FREE = re.compile(r"(?<![^\W\d])\d[\d.,]*")
_POWER = re.compile(r"(?:[×x*·⋅]|\\times|\\cdot)\s*10\s*(?:(?:\^|\*\*)\s*\{?\(?[-−+]?\d+\)?\}?|[⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+|-\d+)"
                    r"|(?<=[\w)\]}])(?:\^|\*\*)\s*\{?\(?[-−+]?\d+\)?\}?|[⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+|\*{2,}")
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
    status: str = ""  # one of STATUSES with verifier.support_quotes, else ''
    context: str = ""  # what the judge is shown with the claim (see split_claims)


@dataclass
class Verification:
    answer: str  # the draft rendered by verifier.answer_style (see render), after the consistency review
    claims: list[Claim]
    consistency: list[dict[str, str]] = field(default_factory=list)  # applied review fixes: old, new, why, conflict
    arithmetic: list[dict[str, str]] = field(default_factory=list)  # fixed results: expression, stated, fixed
    checks: list[dict[str, Any]] = field(default_factory=list)  # domain checker findings: checker, text, fixed, note
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)  # every source issued: label, kind, passage
    judge_calls: list[dict[str, Any]] = field(default_factory=list)  # what each judge call saw, and its reply
    statuses: bool = False  # the claims have an evidence status (verifier.support_quotes)
    computations: list[dict[str, Any]] = field(default_factory=list)  # per computation: n, its numbers and origins

    def report(self) -> dict[str, Any]:
        claims = [c for c in self.claims if c.outcome != NO_CLAIM]
        status = self.statuses
        return {
            "claims": [{"n": c.n, "text": c.text, "outcome": c.outcome, "ids": c.ids, "note": c.note,
                        **({"status": c.status} if status else {})} for c in claims],
            "counts": {o: sum(c.outcome == o for c in claims) for o in OUTCOMES},
            **({"statuses": {s: sum(c.status == s for c in claims) for s in STATUSES}} if status else {}),
            "consistency": self.consistency,
            "arithmetic": self.arithmetic,
            "checks": self.checks,
            "sources": self.sources,
            "judge_calls": self.judge_calls,
            "computations": self.computations,
        }


# --- claims ---------------------------------------------------------------------------------------


def split_claims(draft: str, skip_labelled: bool = False, context: int = 0, any_script: bool = False) -> list[Claim]:
    """Sentences of paragraphs and list items; not headings, code, display math, tables or lead-ins, and with
    `skip_labelled` not labelled answer or verdict lines (`A1: …`, `Claim C: Incorrect`). A sentence needs a Latin
    letter, or with `any_script` a letter of any script (verifier.claims_any_script). With `context` (characters:
    verifier.context_chars) every claim also gets what the judge is shown with it, each part cut to that length: the
    nearest heading above it (a `#` line, or a line that is one bold run) and the sentence before it under that
    heading, as `under: Procedure; after: Heat the flask.`"""
    claims: list[Claim] = []
    pos, fence = 0, False
    under = before = ""
    for line in draft.splitlines(keepends=True):
        start, pos = pos, pos + len(line)
        stripped = line.strip()
        if stripped.startswith("```"):
            fence = not fence
            continue
        if not fence and stripped.startswith("#"):
            under, before = plain(stripped.lstrip("#")), ""
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
            if _is_claim(text, any_script):
                at = body_at + a + lead
                shown = [f"{name}: {part[:context]}" for name, part in (("under", under), ("after", before)) if part]
                claims.append(Claim(len(claims) + 1, at, at + len(text), text, cites(text),
                                    context="; ".join(shown) if context else ""))
            before = plain(text)
        if heading := _BOLD_LINE.fullmatch(body.strip()):
            under, before = plain(heading.group(2)), ""
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


def _is_claim(text: str, any_script: bool = False) -> bool:
    p = plain(text)
    letter = any(ch.isalpha() for ch in p) if any_script else re.search(r"[A-Za-z]", p)  # isalpha: not ² or ×
    if not letter or p.endswith((":", "?")) or _BARE_LABEL.fullmatch(p):
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
                   max_change: float = 0.0, read: dict[int, bool] | None = None) -> tuple[str, list[dict[str, str]], int]:
    """(the draft with every wrong `expression = result` fixed in place, the fixes, how many were checked).
    `read` gets, by where its result starts in the draft as given, whether each checked expression gives the result
    written there.
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
        agrees = any(abs(v * s - lhs) <= max(half * s, rel_tol * abs(lhs))
                     for lhs in (a.fraction, a.plain) for v, half in stated for s in scales)
        if read is not None:
            read[a.result_at] = agrees
        if agrees:
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


_GAP = "[ \xa0  ]"  # a space that may group digits in threes (20 000)


def _number_spans(text: str, number: str) -> list[tuple[int, int]]:
    """Spans where the text writes `number` exactly so, not as part of another number (1350, 350.5, 1 350, 3,65) or
    of a clock time (17:00, 9:17)."""
    return [m.span() for m in re.finditer(
        rf"(?<![\w.,])(?<!\d{_GAP})(?<!\\,)(?<!\d[:：]){re.escape(number)}(?!\w|[.,]\d|{_GAP}\d{{3}}(?!\d)|\\,\d|[:：]\d)",
        text)]


def _unit_at(text: str, end: int) -> str:
    """The % sign or the unit word right after a number that ends at `end`; '' for none (a stopword is none)."""
    m = _UNIT.match(text.translate(_NORM), end)
    return m[1] if m and m[1].lower() not in STOPWORDS else ""


def carry_forward(draft: str, fixes: list[dict[str, str]], request: str, task: str = "",
                  strict: bool = True) -> tuple[str, list[str]]:
    """(the draft with the old value of each applied arithmetic fix replaced by the fixed one wherever the draft
    writes it again after the fixed computation, the trace lines). An occurrence is replaced when it is the same
    written number (see _number_spans), a % sign is after both or neither, it has the fixed result's unit word (a
    unit there, the same word here) or is a bare input of a written computation (`=` after it on its line: later
    computations write their inputs bare), it is not the stated result of a computation (that follows from its own
    left side) and, with the `task`, it is not quoted from it (see quoted). A bare number in prose or one with
    another unit (`7 days`, `Table 4`) may be another quantity: never carried. Nothing is carried for a fix whose
    old value the request writes (it may be given data), an old value fixed to two values, or one that another
    computation of the draft also gives (within rounding). Sets `carried` (the occurrences replaced) on every fix."""
    comma = bool(re.search("[А-Яа-яЁё]", draft))
    found = list(find_arithmetic(draft, comma, strict))
    results, mask = {a.result_at for a in found}, quoted(draft, task) if task else []
    ways: dict[str, set[str]] = {}
    for f in fixes:
        ways.setdefault(f["stated"], set()).add(f["fixed"])
    spans: dict[tuple[int, int], str] = {}
    trace = []
    for f in fixes:
        old, new, f["carried"] = f["stated"], f["fixed"], 0
        after = f"(after the fix of {f['expression']} = {old})"
        if _number_spans(request, old):
            trace.append(f"> not carried: {old} is in the request, maybe given data {after}")
            continue
        if len(ways[old]) > 1:
            trace.append(f"> not carried: {old} was fixed to {len(ways[old])} different values {after}")
            continue
        value = read_numbers(_decimal_points(old, comma))
        also = next((a for a in found if a.expression != f["expression"] and value
                     and any(abs(x - value[0].value) <= value[0].half for x in (a.plain, a.fraction))), None)
        if also:  # another computation gives the same number: a later use may mean that one
            trace.append(f"> not carried: {old} is also the value of {also.expression} {after}")
            continue
        at = next((a.result_at + len(new) for a in found
                   if a.expression == f["expression"] and draft.startswith(new, a.result_at)), None)
        if at is None:
            continue
        unit = _unit_at(draft, at)
        for s, e in _number_spans(draft, old):
            other = _unit_at(draft, e)
            if (s >= at and s not in results and (s, e) not in spans and not (mask and all(mask[s:e]))
                    and (unit == "%") == (other == "%")
                    and ((unit and unit == other) or (not other and "=" in draft[e : _line_end(draft, e)]))):
                spans[s, e] = new
                f["carried"] += 1
                trace.append(f"> carried: {old} → {new} {after}")
    for (s, e), new in sorted(spans.items(), reverse=True):
        draft = draft[:s] + new + draft[e:]
    return draft, trace


def _line_end(text: str, at: int) -> int:
    end = text.find("\n", at)
    return end if end >= 0 else len(text)


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


def _about(norm: str, n: Number, s: Source, sources: Sources) -> bool:
    """Whether a number may contradict record s: for a compound record, the claim names a substance before the
    number and the nearest such mention (a name or synonym of any source, or a formula) is s's own, and the number
    is no computed result (`5 × 18.015 = 90.08`); `Ethanol (46.07 g/mol) … water`, `Heavy water D2O … 20.03`,
    `CuSO4·5H2O … 249.68` citing H2SO4 are about another substance. Other records: always."""
    hills = {to_hill(f) for f in s.record.formulas()}
    head = norm[: n.start]
    if not hills:
        return True
    if re.search(r"\d\s*[×*/÷·⋅+−-]\s*[\d(][^=\n]*=\s*$", head):
        return False
    low, mentions = head.lower(), []
    for x in sources.items:
        for name in x.record.names():
            if len(name) >= 3:
                mentions += [(m.end(), x.id) for m in re.finditer(rf"(?<!\w){re.escape(name.lower())}(?!\w)", low)]
        for f in x.record.formulas():  # as written, also inside a bracket: M(H2O)
            mentions += [(m.end(), x.id) for m in re.finditer(rf"(?<![A-Za-z0-9]){re.escape(f)}(?![a-z0-9])", head)]
    for hill, _, end in read_formulas(head):
        owner = next((x.id for x in sources.items if hill in {to_hill(f) for f in x.record.formulas()}), None)
        mentions.append((end, owner))
    return bool(mentions) and max(mentions)[1] == s.id


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
        if units and any(_matches(n, v.number, rel_tol) and _fits(n, v, norm, True, named[s.id], units)[0]
                         for n in nums for v in s.record.values()):
            nums_s = []  # the claim states the record's value: its other numbers are about something else
        else:
            nums_s = nums
        for n in nums_s:
            if any(_matches(n, v.number, rel_tol) and _fits(n, v, norm, True, named[c.id], units)[0]
                   for c in cited for v in c.record.values()):
                continue  # the value of a cited record
            if units and not _about(norm, n, s, sources):
                continue  # a number of another substance, or a computed one
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


def _bare(text: str) -> str:
    """The text without what looks like an operation and is none: the power of ten of scientific notation, an
    exponent, bold marks."""
    return _POWER.sub("", text)


def _worked(left: str) -> bool:
    """`left`, the text before a `=` or `≈` (through _bare), is written arithmetic that the sign gives the result
    of: its last clause (after a `;` or `:`) joins two numbers by an operator (_JOINED), and both from there to its
    last number and from that number to the sign stand at most two words (a unit, a remark), no comma between
    words and, after the number, no bracket left open (a remark in brackets is passed over). A value that has a name
    of its own before the sign is stated, not computed: `a 3 × 5 cm strip (m = 0.1 g)`, `1.25 × (1 + R1/R2) with
    R1 = 240 Ω`, `a 3 × 5 cm strip and note that 1 inch = 2.54 cm`."""
    clause = re.split(r"[;:]\s", left)[-1]
    numbers, joins = list(_FREE.finditer(clause)), list(_JOINED.finditer(clause))
    if not numbers or not joins:
        return False
    last = numbers[-1]
    gap, tail = (re.sub(r"\\[A-Za-z]+|\\.", " ", t) for t in (clause[joins[-1].end() : last.start()], clause[last.end() :]))
    tail = re.sub(r"\([^()]*\)", " ", tail)
    return not re.search(r"[(\[]", tail) and all(
        not re.search(r",\s", t) and len(re.findall(r"[^\W\d_]\S*", t)) <= 2 for t in (gap, tail))


def writes_arithmetic(text: str) -> bool:
    """The text writes arithmetic with its result: `a op b = c`, two numbers joined by an operator and then `=` or
    `≈` and a number (see _JOINED and _worked). `R = 0.082 L·atm/(mol·K)`, `5.29 × 10⁻¹¹ m`, `70–80 °C`, a formula
    in letters, a size such as `10 × 20 cm` and an expression without its result do not."""
    bare = _bare(text)
    return any(_worked(bare[: m.start()]) for m in _RESULT_SIGN.finditer(bare))


def results(draft: str, arithmetic: bool = False) -> list[float]:
    """Every number written right after `=` or `≈` in the draft and not followed by an operator: values the
    answer computed. With `arithmetic` (verifier.computation_needs_arithmetic) only where the text before the
    sign, back to the start of its line or sentence, is written arithmetic (_worked; `\\approx` is a sign too): the
    value of `R = 0.082` is stated, not computed."""
    out = []
    for m in re.finditer(r"(?:[=≈]|\\approx)[ \t$]*" if arithmetic else r"[=≈][ \t$]*", draft):
        text = draft[m.end() : m.end() + 40]
        nums = read_numbers(text)
        if nums and nums[0].start == 0 and not re.match(r"\s*(?:[×*/·⋅÷+^−-]|\\times|\\cdot)", text[nums[0].end :]):
            start = max(draft.rfind("\n", 0, m.start()), draft.rfind(". ", 0, m.start())) + 1
            if not arithmetic or _worked(_bare(draft[start : m.start()])):
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


ARITHMETIC, RESTATED, GIVEN = "arithmetic", "restated", "given"


def computation(claim: Claim, given: list[float], computed: list[float], rel_tol: float) -> str:
    """What kind of computation the claim is with verifier.computation_needs_arithmetic, or '': `arithmetic` (it
    writes arithmetic with its result, see writes_arithmetic), `restated` (every number is the task's or a
    result of the answer, at least one a result) or `given` (every number is the task's and the claim writes them
    with `=` or `≈`: the task's own data repeated). Any other claim with a number is a stated value: a value
    written with `=`, `≈` or a comparison sign, a number in scientific notation, a formula in letters."""
    nums = read_numbers(claim.text)
    if not nums:
        return ""
    if writes_arithmetic(claim.text):
        return ARITHMETIC
    is_given = [any(_matches(n, v, rel_tol) for v in given) for n in nums]
    is_result = [any(_matches(n, v, rel_tol) for v in computed) for n in nums]
    if all(g or r for g, r in zip(is_given, is_result)):
        return RESTATED if any(is_result) else GIVEN if re.search("[=≈]", claim.text) else ""
    return ""


def calculator_results(text: str) -> list[float]:
    """The numbers a calculator call returned: of every line `… = result` of its result text whose result is one
    number, that number."""
    out = []
    for line in text.splitlines():
        result = line.rpartition(" = ")[2].strip() if " = " in line else ""
        nums = read_numbers(result)
        if len(nums) == 1 and (nums[0].start, nums[0].end) == (0, len(result)):
            out.append(nums[0].value)
    return out


def origins(claim: Claim, given: list[float], computed: list[float], held: list[tuple[str, list[float]]],
            rel_tol: float) -> list[dict[str, str]]:
    """Where each number of a claim comes from, the first of these that holds: `given` (the task writes it),
    `source` with its id (`held`: the numbers of each of the answer's sources, title and passage), `result` (it
    stands behind written arithmetic, or a calculator call of the answer returned it), `factor` (a whole number up
    to 12 or a power of ten, written without a unit), else `assumed`. Each with the number as written and, when one
    follows it, its unit."""
    units = {n.start: unit for n, unit in _quantities(claim.text)}
    out = []
    for n in read_numbers(claim.text):
        source = next((i for i, values in held if any(_matches(n, v, rel_tol) for v in values)), "")
        factor = n.start not in units and ((n.value.is_integer() and abs(n.value) <= 12)
                                           or (n.value > 0 and math.log10(n.value).is_integer()))
        origin = ("given" if any(_matches(n, v, rel_tol) for v in given) else "source" if source
                  else "result" if any(_matches(n, v, rel_tol) for v in computed) else "factor" if factor else "assumed")
        out.append({"number": claim.text[n.start : n.end], **({"unit": units[n.start]} if n.start in units else {}),
                    "origin": origin, **({"id": source} if origin == "source" else {})})
    return out


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


def verdict_tokens(text: str, words: VerdictWords) -> list[str]:
    """The verdict words of text in order, case ignored: each negator (a whole word) and each word that starts
    with a verdict stem, as its longest such stem (`is not verified` -> not, verif; `неверно` -> неверн)."""
    stems = sorted((s.casefold() for s in words.stems), key=len, reverse=True)
    negators = {n.casefold() for n in words.negators}
    found = []
    for w in re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)*", text.replace("’", "'").casefold()):
        if w in negators:
            found.append(w)
        elif stem := next((s for s in stems if w.startswith(s)), None):
            found.append(stem)
    return found


def _answer_line(text: str, start: int, end: int) -> bool:
    """A line that text[start:end] touches is a labelled answer or verdict line (`A1: …`, as in split_claims)."""
    stop = text.find("\n", end)
    lines = text[text.rfind("\n", 0, start) + 1 : stop if stop >= 0 else len(text)].splitlines()
    return any(_LABELLED.match(line[_MARKERS.match(line).end() :]) for line in lines)


def _follows_own_numbers(answer: str, fix: dict[str, str]) -> bool:
    """A review fix that changes a verdict may stand only when it brings a number the answer itself states or
    computes elsewhere (`85 % (consistent)` → `98.6 % (implausibly high)` next to its own 98.6 %): the verdict
    then follows the answer's own numbers. A bare flip (`incorrect` → `correct`) never does."""
    comma = bool(re.search("[А-Яа-яЁё]", answer))
    had = {n.value for n in read_numbers(_decimal_points(fix["old"], comma))}
    adds = any(n.value not in had for n in read_numbers(_decimal_points(fix["new"], comma)))
    return adds and not unconfirmed(answer, fix)


def _unit_key(text: str, end: int) -> tuple[str, str]:
    """(kind, unit) written right after a number that ends at `end`: `%` or a currency sign as both, a unit
    read_unit knows as (its SI powers, those and its scale), or ("", "") for none or an unknown word."""
    if m := re.match(r"[ \t]*([%€$£¥₹₽])", text[end:]):
        return m[1], m[1]
    unit, _ = read_unit(text[end:])
    kind = repr(sorted((k, p) for k, p in unit[1].items() if p)) if unit else ""
    return (kind, f"{unit[0]:g}{kind}") if kind != "[]" and kind else ("", "")


def unconfirmed(answer: str, fix: dict[str, str]) -> str:
    """Why a review fix whose OLD text occurs once in the answer does not resolve a conflict inside the answer,
    or "" when it does: its CONFLICT quote is elsewhere in the answer, NEW adds something, every number NEW adds
    (a new value, or an old one with another unit) measures what OLD's numbers measure and is stated in the quote
    with the same unit or none (within the rounding of either) or is the value of a computation written there, and
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
    o, w, q = (_decimal_points(t, comma) for t in (old, new, quote))
    stated, computed = read_numbers(q), computations(quote, comma)
    had = {(n.value, _unit_key(o, n.end)[1]) for n in read_numbers(o)}
    kinds = {k for n in read_numbers(o) if (k := _unit_key(o, n.end)[0])}
    numbers = [n for n in read_numbers(w) if (n.value, _unit_key(w, n.end)[1]) not in had]
    for n in numbers:
        kind, unit = _unit_key(w, n.end)
        if kind and kinds and kind not in kinds:  # "the 12 % loss" → "the 12 h loss": another quantity
            return f"{new[n.start : n.end]} measures something else than {old!r}"
        if not (any(abs(n.value - s.value) <= max(n.half, s.half) and unit in ("", _unit_key(q, s.end)[1])
                    or abs(n.value - s.value) <= max(n.half, s.half) and not _unit_key(q, s.end)[1] for s in stated)
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
    an edit puts in is in the passages, a number within half a unit of the last written digit of either, and no
    number or content word it takes out is (a passage that states both does not contradict the claim). Only the
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
        for key, a, b, n in old[i1:i2]:  # the source states what the edit takes out too: no contradiction
            if (any(abs(n.value - s.value) <= max(n.half, s.half) for s in stated) if n
                    else key in words and len(key) >= 3 and key not in STOPWORDS):
                return None, f"not a contradiction: the source also states {text[a:b]}"
    out = text
    for _, i1, i2, j1, j2 in reversed(edits):
        out = out[: old[i1][1]] + correction[new[j1][1] : new[j2 - 1][2]] + out[old[i2 - 1][2] :]
    return out, ""


def parse_verdicts(text: str, given: set[str], quotes: bool = False) -> dict[int, tuple[str, list[str], str | None]]:
    """Claim number -> (verdict, decisive ids among `given`, what follows `|`: a correction) from the judge's reply
    lines. With `quotes` the lines are those of verifier.support_prompt: `supports` and `partly` carry a quote. Such
    a line is read evidence first (`C1: S2 "words" -> supports`: the ids are those before the quoted words, and a
    correction follows `contradicts` after `|`) or verdict first (`C1: supports S2 | "words"`). A verdict that needs
    a passage and names none of the given ones is not_stated (without `quotes`: not_said), unless it carries a
    quote: the quote then says which passage (see quoted_support)."""
    pattern, unsettled = (_QUOTED_VERDICT, NOT_STATED) if quotes else (_VERDICT, "not_said")
    out: dict[int, tuple[str, list[str], str | None]] = {}
    for line in text.splitlines():
        m = pattern.match(line.strip())
        first = _QUOTE_FIRST.match(line.strip()) if quotes and not m else None
        if first:
            number, evidence, verdict, tail = first.groups()
            mark = re.search(r"[\"“”«»„]", evidence)  # the words start at the quote mark, or after the ids
            cut = mark.start() if mark else re.match(r"(?:\W*\bS\d+\b)*\W*", evidence).end()
            ids_part, correction = evidence[:cut], evidence[cut:]
            if verdict.lower() == "contradicts":
                correction = tail.partition("|")[2]
        elif m:
            number, verdict = m.group(1), m.group(2)
            ids_part, _, correction = m.group(3).partition("|")
        else:
            continue
        verdict = verdict.lower().replace(" ", "_")
        ids = [i for i in dict.fromkeys(f"S{x}" for x in re.findall(r"S(\d+)", ids_part)) if i in given]
        quoted = quotes and verdict in ("supports", "partly") and correction.strip()
        if verdict in ("supports", "partly", "contradicts") and not ids and not quoted:
            verdict = unsettled
        out.setdefault(int(number), (verdict, ids, correction.strip() or None))
    return out


def _readings(text: str, fractions: bool = False) -> list[tuple[Number | None, list[tuple[float, float, str]]]]:
    """Per number of a sentence (commas between digits read as in unconfirmed: decimal commas in Russian text), what
    it may stand for, each as (value, half a unit of its last digit, kind): the number as written (kind ''); a
    percentage also as the share it is (50 % = 0.5); a number with a unit this code reads also in SI base units, its
    kind the unit's base powers (so km/s compares with m/s and never with kg; a degree Celsius next to another unit
    is a step of temperature and counts as a kelvin there, so J/(g·°C) compares with J/(kg·K)). With `fractions`,
    every a/b written in the text is one more value (1/2 = 0.5)."""
    points = _decimal_points(text, bool(re.search("[А-Яа-яЁё]", text)))
    norm = points.translate(_NORM)
    out: list[tuple[Number | None, list[tuple[float, float, str]]]] = []
    for n in read_numbers(points):
        values = [(n.value, n.half, "")]
        if re.match(r"\s*\\?%", norm[n.end :]):
            values.append((n.value / 100, n.half / 100, ""))
        unit, _ = read_unit(norm[n.end :])
        if unit and any(unit[1].values()):
            powers = {k: p for k, p in unit[1].items() if p}
            if "°C" in powers and len(powers) > 1:  # with another unit a degree is a step of temperature: a kelvin
                powers["K"] = powers.get("K", 0) + powers.pop("°C")
            kind = repr(sorted((k, p) for k, p in powers.items() if p))
            values.append((n.value * unit[0], n.half * unit[0], kind))
        out.append((n, values))
    if fractions:
        out += [(None, [(int(a) / int(b), 0.0, "")])
                for a, b in re.findall(r"(?<![\w./])(\d{1,3})\s*/\s*(\d{1,3})(?![\w./])", norm) if int(b)]
    return out


_UNSEEN = re.compile("[​-‏⁠﻿­]")  # zero-width spaces, joiners and marks, soft hyphens
# What a quote is compared by: runs of letters and digits, and every other character alone. The markup of a formula
# comes first and is none of them: a `\displaystyle` (any `\…style`), the delimiters `$`, `\(`, `\)`, `\[`, `\]`,
# and braces.
_SIGN = re.compile(r"\\[a-z]*style\b|\$|\\[()\[\]]|[{}]|\w+|[^\w\s]")


def _signs(text: str) -> list[tuple[str, int]]:
    """The words and signs a quote is compared by (_SIGN), in order, each with the number of the typeset formula it
    stands in, 0 for none. The text is read as _quote_text reads it, case and spacing aside, and without the markup
    of a formula, so one written plainly is the same signs as its marked-up form. A typeset formula is a brace group
    that opens with a style command, as an encyclopedia passage writes its math."""
    out: list[tuple[str, int]] = []
    depth = inside = formula = 0  # open braces; the brace depth of the open typeset formula; its number
    last = ""
    for m in _SIGN.finditer(_quote_text(text)[0]):
        sign = m.group()
        if sign == "{":
            depth += 1
        elif sign == "}":
            depth = max(depth - 1, 0)
            inside = inside if depth >= inside else 0
        elif sign.endswith("style") and sign.startswith("\\"):
            if last == "{" and not inside:
                inside, formula = depth, formula + 1
        elif sign not in ("$", "\\(", "\\)", "\\[", "\\]"):
            out.append((sign.casefold(), formula if inside else 0))
        last = sign
    return out


def _row(want: list[str], have: list[tuple[str, int]], start: int) -> int | None:
    """Where the signs `want` end in `have` (see _signs) when they stand there in a row at or after `start`: the
    index after the last of them, or None. Between two of them a typeset formula of `have` may be left out whole:
    a reader who copies the words of a sentence skips markup."""
    for first in range(start, len(have)):
        at, k = first, 0
        while k < len(want) and at < len(have):
            if have[at][0] == want[k]:
                at, k = at + 1, k + 1
            elif k and have[at][1] and have[at][1] != have[at - 1][1]:  # a formula starts here
                number = have[at][1]
                while at < len(have) and have[at][1] == number:
                    at += 1
            else:
                break
        if k == len(want):
            return at
    return None


def _beside(text: str, n: Number) -> set[tuple[float, str]]:
    """A number of the text with the word before it and with the word after it (each word by its first four letters,
    as term_share matches words): what tells the number of a name, `Apollo 11`, from the same number elsewhere."""
    words = (re.search(r"([^\W\d_]+)\W*$", text[: n.start]), re.match(r"\W*([^\W\d_]+)", text[n.end :]))
    return {(n.value, m[1].casefold()[:4]) for m in words if m}


def quoted_support(claim: str, quote: str, passage: str, min_words: int = 1, min_share: float = 0.0,
                   title: str = "") -> tuple[str, str]:
    """(status, why it is not supported) of a claim the judge says `passage` supports with `quote`
    (verifier.support_quotes). supported, when all of these hold:
    - the quote is in the passage as whole words and signs in a row (see _signs: case, spacing, the marks around the
      quote, characters no reader sees and the markup of a formula aside; a formula the passage typesets may be left
      out whole; dots in the quote stand for words left out: every part is there, in their order);
    - it has at least `min_words` words, or holds a number of the claim;
    - it bears on the claim: it holds a number of the claim, or shares with it a content word that is not a word
      of `title` (the passage's title: the name of its subject shows what a passage is about, not what it states)
      and at least `min_share` of its content words (tools.term_share; not asked when claim and quote are in
      different scripts);
    - it holds every number of the claim, within the rounding of either (see _readings: 50 % is 1/2, km/s is m/s,
      per °C is per K), but for a number of the subject's name: one that stands in `title` next to the same word as
      in the claim.
    not_stated when one of the first three fails; partly when only a number of the claim is not in the quote."""
    quote, passage = _UNSEEN.sub("", quote), _UNSEEN.sub("", passage)
    parts = []  # per stretch between dots, its words and signs without the signs around it
    for part in re.split(r"\.{3,}|…", _quote_text(quote)[0]):
        signs = [sign for sign, _ in _signs(part)]
        words = [i for i, sign in enumerate(signs) if re.match(r"\w", sign)]
        if words:
            parts.append(signs[words[0] : words[-1] + 1])
    if not parts:
        return NOT_STATED, "no quote"
    said = " ".join(sign for part in parts for sign in part)
    have = [r for _, group in _readings(quote, fractions=True) for r in group]
    want = _readings(claim)
    missing = [n for n, group in want
               if not any(k == j and abs(v - w) <= max(h, g) for v, h, k in group for w, g, j in have)]
    held = len(missing) < len(want)
    if len(_words(said)) < min_words and not held:
        return NOT_STATED, "quote too short"
    shown, at = _signs(passage), 0
    for part in parts:
        at = _row(part, shown, at)
        if at is None:
            return NOT_STATED, "quote not found"
    if not held and non_latin(claim) == non_latin(quote):
        share = term_share(claim, said)
        if not share:
            return NOT_STATED, "the quote shares no word with the claim"
        if not term_share(claim, said, skip=title):
            return NOT_STATED, "the quote shares only its passage's title with the claim"
        if share < min_share:
            return NOT_STATED, f"the quote holds under {min_share:g} of the claim's words"
    if missing:  # the passage states a number of its subject's name by its title
        named = {pair for n in read_numbers(title) for pair in _beside(title, n)}
        missing = [n for n in missing if not _beside(claim, n) & named]
    if missing:
        return PARTLY, f"{' and '.join(claim[n.start : n.end] for n in missing)} not in the quote"
    return SUPPORTED, ""


def _open_status(c: Claim, status: str) -> None:
    """Give an unsettled claim an evidence status, unless the one it has from an earlier call says more (_OPEN)."""
    if c.status not in _OPEN or _OPEN.index(status) < _OPEN.index(c.status):
        c.status = status


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

    async def verify(self, question: str, draft: str, sources: Sources,
                     calculated: Sequence[float] = ()) -> AsyncIterator[dict | Verification]:
        """Yield reasoning deltas while checking, then the Verification. `calculated`: the numbers the answer's
        calculator calls returned (results of the answer, like the ones its written arithmetic states)."""
        arithmetic, checked, lines, carried = [], 0, [], []
        task = question if self.cfg.keep_quotes else ""
        args = (self.cfg.arithmetic_rel_tol, self.cfg.arithmetic_prefixes, task, self.cfg.arithmetic_strict_parse,
                self.cfg.arithmetic_max_change)
        if self.cfg.arithmetic_check:
            with stage("arithmetic", CHECK):
                draft, arithmetic, checked = fix_arithmetic(draft, *args)
        kept = [a for a in arithmetic if "note" in a]  # quoted from the task: only in the trace
        arithmetic = [a for a in arithmetic if "note" not in a]
        first = len(arithmetic)
        if self.cfg.carry_fixes_forward and arithmetic:
            with stage("carry_forward", CHECK):
                draft, carried = carry_forward(draft, arithmetic, question, task, self.cfg.arithmetic_strict_parse)
                if any(a["carried"] for a in arithmetic):  # once more: a later computation may have used an old value
                    draft, again, _ = fix_arithmetic(draft, *args)
                    arithmetic += [a for a in again if "note" not in a]
                    kept += [a for a in again if "note" in a and a not in kept]
        if self.cfg.arithmetic_answer_lines and arithmetic:
            draft, lines = fix_answer_lines(draft, arithmetic)
        if checked:
            yield _say(f"\n\nArithmetic: {checked} calculation{'s' if checked != 1 else ''} recomputed,"
                       f" {len(arithmetic)} fixed\n"
                       + "".join(f"- {a['expression']} = {a['stated']} → {a['fixed']}\n" for a in arithmetic[:first])
                       + "".join(f"{line}\n" for line in carried)
                       + "".join(f"- {a['expression']} = {a['stated']} → {a['fixed']}\n" for a in arithmetic[first:])
                       + "".join(f"- {a['expression']} = {a['stated']}: {a['note']} (would be {a['fixed']})\n"
                                 for a in kept)
                       + "".join(f"- answer line: {a['stated']} → {a['fixed']} (as {a['expression']})\n"
                                 for a in lines))
        arithmetic += lines
        checks: list[dict[str, Any]] = []
        if self.cfg.checkers:
            with stage("checkers", CHECK):
                draft, checks = run_checkers(draft, self.cfg.checkers, question, self.cfg.keep_quotes,
                                             self.cfg.checkers_clear_only)
        for name in self.cfg.checkers:
            found = [c for c in checks if c["checker"] == name]
            if found:
                yield _say(f"\n\nCheck {name}: {len(found)} finding{'s' if len(found) != 1 else ''}\n" + "".join(
                    f"- {c['text']!r} → {c['fixed']!r} ({c['note']})\n" if c["fixed"] is not None
                    else f"- {c['text']!r}: {c['note']}\n" for c in found))
        claims: list[Claim] = []
        calls: list[dict[str, Any]] = []  # the judge calls, for the report
        computations: list[dict[str, Any]] = []  # the numbers of each computation and where they come from
        answer, fixes = draft, []
        if self.cfg.record_check or self.cfg.claim_judge:  # else no claim stage: the draft goes on as it is
            checking = self._check_claims(question, draft, sources, claims, calls, list(calculated), computations)
            try:
                async for line in checking:
                    yield line
            finally:
                await checking.aclose()
            answer = render(draft, claims, self.cfg.answer_style == "annotated", self.cfg.strip_source_ids)
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
        shown = {s.id: {"label": s.record.label(), "kind": s.kind, "literature": s.literature, "passage": s.passage}
                 for s in sources.items}  # the model's lookups and the re-searches, exactly as shown
        yield Verification(answer, claims, consistency=fixes, arithmetic=arithmetic, checks=checks, sources=shown,
                           judge_calls=calls, statuses=self.cfg.support_quotes, computations=computations)

    def _computation(self, c: Claim, given: list[float], computed: list[float], read: dict[int, bool]) -> str:
        """'' or the note of a claim that is a computation, which no evidence check reads: by is_computation, or
        with verifier.computation_needs_arithmetic by computation, where the note says what was done (`read`: the
        results the arithmetic check read in the draft, by position, and whether its own result is the one written)."""
        if not self.cfg.computation_needs_arithmetic:
            done = is_computation(c, given, computed, self.cfg.rel_tol)
            return "computation: checked in code, not against sources" if done else ""
        kind = computation(c, given, computed, self.cfg.rel_tol)
        if kind == ARITHMETIC:
            same = [ok for at, ok in read.items() if c.start <= at < c.end]
            if not same:
                return "computation: not recomputed (the check could not read it)"
            return "computation: recomputed in code" + ("" if all(same) else " with another result, left as written")
        return {RESTATED: "computation: restates the task's or the answer's numbers",
                GIVEN: "computation: the task's own numbers repeated"}.get(kind, "")

    async def _check_claims(self, question: str, draft: str, sources: Sources, claims: list[Claim],
                            calls: list[dict[str, Any]], calculated: list[float],
                            computations: list[dict[str, Any]]):
        """The claim stage: split the draft into `claims` (filled in place), check them against the records in
        code (verifier.record_check), judge the rest and re-search what is not supported (verifier.claim_judge).
        Sets every claim's outcome and adds every judge call to `calls`. `calculated`: the numbers the answer's
        calculator calls returned; `computations` gets, per claim filed as a computation, its numbers and where
        each comes from (origins). Yields reasoning lines."""
        derived = self.cfg.skip_derived_claims
        pending = []
        with stage("claims", CHECK):
            claims += self.split(draft)
            yield _say(f"\n\nVerification: {len(claims)} claims\n")
            given = [n.value for n in read_numbers(question)]
            worked = [*results(draft, arithmetic=True), *calculated]  # behind written arithmetic, or calculated
            strict = self.cfg.computation_needs_arithmetic
            computed = worked if strict else results(draft)
            read: dict[int, bool] = {}
            if strict:  # the expressions the arithmetic check reads in this draft, recomputed
                fix_arithmetic(draft, self.cfg.arithmetic_rel_tol, self.cfg.arithmetic_prefixes,
                               question if self.cfg.keep_quotes else "", self.cfg.arithmetic_strict_parse,
                               self.cfg.arithmetic_max_change, read)
            held: list[tuple[str, list[float]]] | None = None  # the numbers of each source, read when first needed
            mask = quoted(draft, question) if self.cfg.keep_quotes else []
            for c in claims:
                if is_quotation(draft, mask, c.start, c.end):  # before the code checks: they would rewrite it too
                    c.outcome, c.note = UNVERIFIED, "quoted from the task: not checked"
                    yield _say(f"- C{c.n} {c.outcome} ({c.note})\n")
                elif self.cfg.record_check and (
                        found := code_check(c, sources, self.cfg.rel_tol, self.cfg.unit_checks)):
                    c.outcome, c.ids, c.fixed, c.note = found
                    c.note = f"code: {c.note}"
                    yield _say(f"- C{c.n} {c.outcome} ({c.note})\n")
                elif derived and (note := self._computation(c, given, computed, read)):
                    c.outcome, c.note = UNVERIFIED, note
                    if held is None:
                        held = [(s.id, [m.value for m in read_numbers(f"{s.record.label()}\n{s.passage}")])
                                for s in sources.items]
                    numbers = origins(c, given, worked, held, self.cfg.rel_tol)
                    computations.append({"n": c.n, "numbers": numbers})
                    assumed = [f"{x['number']} {x['unit']}" for x in numbers
                               if strict and x["origin"] == "assumed" and "unit" in x]
                    yield _say(f"- C{c.n} {c.outcome} ({c.note})"
                               + (f"; from no task, source or result: {', '.join(assumed)}" if assumed else "") + "\n")
                else:
                    pending.append(c)

        if not self.cfg.claim_judge:  # no judge call, so no re-search either: nothing could rate its results
            for c in pending:
                c.outcome, c.note = UNVERIFIED, "not judged: the claim judge is off"
            if pending:
                yield _say(f"- {len(pending)} claim{'s' if len(pending) != 1 else ''} not judged"
                           " (the claim judge is off)\n")
            pending = []
        own = list(sources.items)  # what the answer's own lookups returned: the re-searches below add theirs
        every = self.cfg.judge_all_sources  # each claim meets those sources, cited or not (see _shown)
        met: dict[int, set[str]] = {}  # claim number -> the sources a judge call has shown it
        first = own if every else [s for s in own if any(s.id in c.cites for c in pending)]
        if pending and first:
            async for line in self._judge(question, pending, first, calls, pool=own if every else None,
                                          must={c.n: c.cites for c in pending}, met=met):
                yield line
        research = [c for c in pending if not c.outcome]
        if self.cfg.research_values_first:  # a number with a unit first; draft order within each group
            research.sort(key=lambda c: not _quantities(plain(c.text)))
        for c in research[self.cfg.max_research :]:
            c.outcome, c.note = UNVERIFIED, "not supported; no re-search left"
        research = research[: self.cfg.max_research]

        found_for: dict[int, list[str]] = {}
        for c in research:
            query = plain(c.text)
            yield _say(f"\n> {SEARCH}(query={json.dumps(query, ensure_ascii=False)})  [re-search C{c.n}]\n")
            _, lines, ids = self.knowledge.run(SEARCH, json.dumps({"query": query}), sources, self.cfg.research_k,
                                               internal=True)
            yield _say("".join(f"- {line}\n" for line in lines))
            found_for[c.n] = ids
        # A claim is judged again when its re-search found something or it cites a source. With judge_all_sources
        # only when the re-search found a source it has not met, and it is shown those alone: a second reading of
        # the same passages is a second draw of the judge, not more evidence.
        new = {c.n: [i for i in found_for[c.n] if not every or i not in met.get(c.n, ())] for c in research}
        again = [c for c in research if new[c.n] or (c.cites and not every)]
        if again:
            wanted = {i for c in again for i in [*new[c.n], *c.cites]}
            async for line in self._judge(question, again, [s for s in sources.items if s.id in wanted], calls,
                                          "research_judge", pool=[] if every else None, must=new):
                yield line
        for c in research:
            if not c.outcome:
                c.outcome, c.note = UNVERIFIED, c.note or "not supported after re-search"
        if self.cfg.reference_over_literature:
            for c in claims:
                if c.outcome == SUPPORTED and (why := reference_conflict(c, sources, self.cfg.rel_tol)):
                    c.outcome, c.ids, c.note = UNVERIFIED, [], f"{c.note}; only literature supports {why}"
                    if self.cfg.support_quotes:
                        c.status = CONTRADICTED  # a reference record states another value
                    yield _say(f"- C{c.n} {c.outcome} (only literature supports {why})\n")
        if self.cfg.support_quotes:  # the status of every claim no judge call gave one
            for c in claims:
                if c.outcome in (SUPPORTED, CORRECTED, REMOVED):  # settled in code, from a record
                    c.status = c.status or (SUPPORTED if c.outcome == SUPPORTED else CONTRADICTED)
                elif c.n in found_for and not (found_for[c.n] or (c.cites and not every)):
                    _open_status(c, NOT_FOUND)  # its re-search returned nothing (a status it has that says more stays)
                elif c.outcome != NO_CLAIM:
                    c.status = c.status or NOT_CHECKED  # a quotation, a computation, no re-search slot, no verdict
        for c in claims:
            if c.outcome == CORRECTED:
                yield _say(f"- C{c.n} corrected from {', '.join(c.ids)}: {plain(c.fixed or '')}\n")
            elif c.outcome == REMOVED:
                yield _say(f"- C{c.n} removed (contradicted by {', '.join(c.ids)})\n")

    async def _consistency(self, question: str, answer: str):
        """One review call over the whole answer and the task. Yields reasoning lines, then (answer, fixes)."""
        messages = [{"role": "system", "content": self.cfg.consistency_prompt},
                    {"role": "user", "content": f"Task:\n{question}\n\nAnswer:\n{answer}"}]
        if estimate_tokens(messages, self.chars_per_token) > self.budget:
            yield _say("\n\nConsistency review skipped: the answer is too long for the context\n")
            yield answer, []
            return
        try:
            reply = await self._complete(messages, "review")
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
            elif self.cfg.consistency_keep_verdicts and (verdict_tokens(f["old"], self.cfg.verdict_words)
                                                         != verdict_tokens(f["new"], self.cfg.verdict_words)) \
                    and not _follows_own_numbers(answer, f):
                why = f"verdict kept; conflict: {f.get('conflict') or f['why'] or 'not named'!r}"
            else:
                why = unconfirmed(answer, f) if self.cfg.consistency_quotes else ""
            if why:
                yield _say(f"- skipped {f['old']!r} ({why})\n")
                continue
            answer = answer.replace(f["old"], f["new"])
            applied.append(f)
            yield _say(f"- {f['old']!r} → {f['new']!r}" + (f" ({f['why']})" if f["why"] else "") + "\n")
        yield answer, applied

    def split(self, draft: str) -> list[Claim]:
        """The claims of a draft, as this verifier's settings split them."""
        return split_claims(draft, skip_labelled=self.cfg.skip_derived_claims and self.cfg.skip_labelled_claims,
                            context=self.cfg.context_chars if self.cfg.support_quotes else 0,
                            any_script=self.cfg.claims_any_script)

    async def _judge(self, question: str, claims: list[Claim], passages: list[Source], calls: list[dict[str, Any]],
                     stage_name: str = "judge", pool: list[Source] | None = None,
                     must: dict[int, list[str]] | None = None, met: dict[int, set[str]] | None = None):
        """The judge calls of one stage for these claims: one call, or with verifier.judge_batch calls of at most
        that many claims, in their order, each with these passages (see _judge_call). With `pool`
        (verifier.judge_all_sources) each call shows only the passages chosen for its own claims (see _shown), and
        a call with none is not made: its claims wait for the re-search. `met` gets, per claim number, the sources
        a call has shown it. Yields reasoning lines."""
        size = self.cfg.judge_batch or len(claims) or 1
        for i in range(0, len(claims), size):
            batch = claims[i : i + size]
            shown = passages if pool is None else self._shown(batch, passages, pool, must or {})
            if shown or pool is None:
                async for line in self._judge_call(question, batch, shown, calls, stage_name):
                    yield line
                if met is not None and calls[-1]["reply"]:  # a failed call has rated nothing
                    for c in batch:
                        if c.note != NO_VERDICT:  # a claim the reply has no line for has not been judged on them
                            met.setdefault(c.n, set()).update(calls[-1]["passages"])

    def _shown(self, claims: list[Claim], passages: list[Source], pool: list[Source],
               must: dict[int, list[str]]) -> list[Source]:
        """The passages of one judge call with verifier.judge_all_sources, the most wanted first (the last ones go
        when the call does not fit the context): of `passages`, the sources `must` names for these claims (the ones
        they cite; after a re-search, what it found that they have not met); then the verifier.judge_passages
        sources of `pool`, the answer's own lookups, that hold the largest share of the claims' words
        (tools.term_share; digits aside: a digit in common says nothing). A source that holds none of those words
        is left out, unless it is in another script than the claims, where words cannot be compared: those follow
        in the order of the lookups."""
        named = {i for c in claims for i in must.get(c.n, [])}
        words = " ".join(w for w in re.findall(r"\w+", " ".join(plain(c.text) for c in claims)) if not w.isdigit())
        ranked = []
        for at, s in enumerate(pool):
            text = f"{s.record.label()}\n{s.passage}"
            share = term_share(words, text)
            if s.id not in named and (share or non_latin(words) != non_latin(text)):
                ranked.append((-share, at, s))
        return [s for s in passages if s.id in named] + [s for *_, s in sorted(ranked)[: self.cfg.judge_passages]]

    async def _judge_call(self, question: str, claims: list[Claim], passages: list[Source],
                          calls: list[dict[str, Any]], stage_name: str):
        """One judge call for these claims (named `stage_name` in an open stage log); sets their outcome and adds
        what the call saw and its reply to `calls` (a failed call: an empty reply). With verifier.support_quotes the
        judge follows support_prompt, a `supports` counts only as far as code finds its quote in a passage it names
        (quoted_support), a claim without a line in the reply is not rated, and every rated claim gets its evidence
        status. Yields reasoning lines."""
        quotes = self.cfg.support_quotes
        prompt = self.cfg.support_prompt if quotes else self.cfg.judge_prompt
        passages = list(passages)
        while True:  # leave passages out from the end until the call fits the context
            user = (
                f"Question: {question}\n\nPassages:\n"
                + "\n\n".join(f"[{s.id}] {s.record.label()}\n{s.passage}" for s in passages)
                + "\n\nClaims:\n"
                + "\n".join(f"C{c.n}{f' [{c.context}]' if c.context else ''}: {plain(c.text)}" for c in claims)
            )
            messages = [{"role": "system", "content": prompt}, {"role": "user", "content": user}]
            if not passages or estimate_tokens(messages, self.chars_per_token) <= self.budget:
                break
            yield _say(f"- {passages.pop().id} left out of the judge call (context budget)\n")
        given = {s.id for s in passages}
        call = {"stage": stage_name, "claims": [{"n": c.n, "text": plain(c.text),
                                                 **({"context": c.context} if c.context else {})} for c in claims],
                "passages": [s.id for s in passages], "reply": ""}
        calls.append(call)
        try:
            reply = await self._complete(messages, stage_name)
        except (UpstreamError, httpx.HTTPError) as e:
            log.error("judge call failed: %s", e)
            yield _say(f"- judge failed: {e}\n")
            return
        call["reply"] = reply
        verdicts = parse_verdicts(reply, given, quotes)
        for c in claims:
            if quotes and c.n not in verdicts:  # no line for it (a reply cut off, a claim left out): not rated
                c.note = NO_VERDICT
                yield _say(f"- C{c.n} {c.note}\n")
                continue
            verdict, ids, correction = verdicts.get(c.n, ("not_said", [], None))
            c.note = f"judge: {verdict}" + (f" {' '.join(ids)}" if ids else "")
            if quotes and verdict in ("supports", "partly"):  # as far as code finds the quote in a passage shown
                # The quote says which passage: the ones the line names first, then the others of the call.
                order = [s for s in passages if s.id in ids] + [s for s in passages if s.id not in ids]
                (status, why), sid = min(
                    ((quoted_support(plain(c.text), correction or "", f"{s.record.label()}\n{s.passage}",
                                     self.cfg.quote_min_words, self.cfg.quote_min_share, s.record.label()), s.id)
                     for s in order),
                    key=lambda found: STATUSES.index(found[0][0]), default=((NOT_STATED, "no passage shown"), ""))
                if verdict == "partly" and status == SUPPORTED:
                    status = PARTLY  # the judge itself says a part of the claim is not there
                c.note += f" | {correction}" if correction else ""
                if why.endswith("not found"):
                    why += f" in {' '.join(ids)}" if ids else " in a passage shown"
                elif sid and sid not in ids:
                    c.note += f", the quote is in {sid}"
                if why:
                    c.note += f", {why}: {status}"
                if status == SUPPORTED:
                    c.outcome, c.ids, c.status = SUPPORTED, [sid], SUPPORTED
                else:
                    _open_status(c, status)
                yield _say(f"- C{c.n} {c.note}\n")
                continue
            if (verdict == "contradicts" and correction and self.cfg.ignore_prompt_echo
                    and echoes(correction, prompt)):
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
            if quotes and c.outcome in (CORRECTED, REMOVED):
                c.status = CONTRADICTED
            elif quotes and verdict != "no_claim":  # a contradiction that changed nothing counts as not stated
                _open_status(c, verdict if verdict in (NOT_STATED, NOT_FOUND) else NOT_STATED)
            yield _say(f"- C{c.n} {c.note}\n")

    async def _complete(self, messages: list[dict[str, Any]], stage_name: str) -> str:
        """The judge's answer text; its reasoning tokens, if any, are ignored."""
        return await complete(self.llm, messages, self.cfg.request, stage_name)


async def complete(llm: LLMClient, messages: list[dict[str, Any]], request: dict[str, Any],
                   stage_name: str = MODEL) -> str:
    """The answer text of one call with these request fields (named `stage_name` in an open stage log); its
    reasoning tokens, if any, are ignored."""
    parts: list[str] = []
    stream = llm.stream_chat(messages, stage=stage_name, **request)
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


def render(draft: str, claims: list[Claim], annotated: bool = True, strip_ids: bool = True) -> str:
    """The draft with every claim rewritten by its outcome. Annotated: labels, ids and the summary line.
    Plain: fixes applied and contradicted claims without a fix dropped, silently; no summary, and no ids unless
    `strip_ids` is off (verifier.strip_source_ids)."""
    out = draft
    for c in sorted(claims, key=lambda c: c.start, reverse=True):
        end = c.end + (1 if c.outcome == REMOVED and out[c.end : c.end + 1] == " " else 0)
        out = out[: c.start] + (_render(c) if annotated else _plain(c)) + out[end:]
    out = re.sub(r"(?m)^[ \t]*(?:[-*+]|\d+[.)])[ \t]*(?:\r?\n|$)", "", out)  # list items left empty by removals
    if not annotated:
        return (strip_cites(out) if strip_ids else out).rstrip()
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
