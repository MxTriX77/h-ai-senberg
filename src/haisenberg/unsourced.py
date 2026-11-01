"""Numbers the final answer states without any source (D-043, F10 invented specifics): found and flagged, never changed.

A number of the answer is sourced when its value is stated in the conversation (the request and earlier turns), in a
tool result, or in the answer's own written computations (the operands and the stated result of every `a op b = c`,
and the value of every arithmetic expression written), within half a unit of the last written digit of either, as in
the review's quote rules (verifier.unconfirmed); signs are ignored. A restatement in another unit or as a percent is
sourced only by a written conversion. Not counted, in the answer or as a source: list numbering, item ids (C3, У2:
digits after a letter are never read as a number), years and dates, and integers 0-10 without a unit; in the answer
also numbers inside text it quotes from the conversation (verifier.quoted).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .calc import computations, find_arithmetic
from .l1.units import read_unit
from .verifier import Number, _decimal_points, quoted, read_numbers

CONTEXT = 60  # characters of context on each side of a flagged number

_CYRILLIC = re.compile("[А-Яа-яЁё]")
# A range: a hyphen between digits (3-5; with spaces, 1 - 0.20, it is a minus), a dash (60–70, 60 – 70), `to`, `до`.
_DASH = re.compile(r"(?<=\d)-(?=\d)|(?<=\d)([ \t]*)[–—](?=[ \t]*\d)")
_GAP = re.compile(r"-|[ \t]*[–—][ \t]*|[ \t]+(?:to|до)[ \t]+")
_UNIT = re.compile(r"[ \t\xa0]*(%|°[ \t]?[CF]|℃|[^\W\d_]+(?:\^?-?\d)?(?:[ \t]*[/·⋅∙*][ \t]*\(?[^\W\d_]+(?:\^?-?\d)?\)?)*)")
_LATEX_UNIT = re.compile(r"\\(?:text|mathrm|rm)\s*\{[^{}]*\}")
_EXPONENT = re.compile(r"\^[{(]?$")  # right before a number: its base's exponent (10^{12}, e^(-2))
_EQUALS = re.compile(r"[ \t*_$]*(?:[^\W\d_][^\s\d=≈]*[ \t*_$]*)?[=≈][ \t*_$]*")  # `=` after a number, maybe a unit
_LIST_START = re.compile(r"[ \t>#*+_-]*")  # Markdown markers before a list number at a line start
_LIST_END = re.compile(r"[.)](?:[ \t*_]|$)")
_MONTHS = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?"
           r"|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?(?![a-z])"
           r"|января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря")
_DATE = re.compile(rf"(?<!\d)(?:\d{{4}}-\d{{1,2}}-\d{{1,2}}|\d{{1,2}}[./]\d{{1,2}}[./]\d{{2,4}}"
                   rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})|(?:{_MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?)(?!\d)", re.I)


@dataclass
class _Token:
    start: int
    end: int  # after the unit
    numbers: list[Number]  # one, or the two ends of a range
    unit: str


def _integer(n: Number) -> bool:
    return n.decimals == 0 and not n.scientific


def _tokens(text: str) -> list[_Token]:
    """The numbers of the text that count (see the module doc), a range (`60–70 %`, `3 to 5`) as one token, each with
    the unit right after it: `%` or a unit the unit reader knows (g/mol, °C, mL). An exponent (`10^{12}`) is part of
    its number."""
    prepared = _DASH.sub(lambda m: f"{m[1] or ''};", _decimal_points(text, bool(_CYRILLIC.search(text))))  # same length
    nums, dates = read_numbers(prepared), [m.span() for m in _DATE.finditer(text)]
    out, i = [], 0
    while i < len(nums):
        group = nums[i : i + 2] if i + 1 < len(nums) and _GAP.fullmatch(text[nums[i].end : nums[i + 1].start]) \
            else nums[i : i + 1]
        i += len(group)
        start, end = group[0].start, group[-1].end
        if any(a < end and start < b for a, b in dates) or _EXPONENT.search(text, max(0, start - 2), start):
            continue
        m = _UNIT.match(text, end)
        unit = m[1] if m and read_unit(m[1])[0] is not None else ""
        end = m.end() if unit else end
        n, line = group[0], text.rfind("\n", 0, start) + 1
        if len(group) == 1 and (lead := _LIST_START.fullmatch(text, line, start)) and (
                "#" in lead.group() or _integer(n) and _LIST_END.match(text, n.end)):
            continue  # list or section numbering: "12. …", "3) …", "### 3.2 …"
        if not unit and all(re.fullmatch(r"\d{4}", text[x.start : x.end]) and 1000 <= x.value <= 2100 for x in group):
            continue  # a year, or years (1957 to 1947)
        if not unit and all(_integer(x) and abs(x.value) <= 10 for x in group):
            continue
        out.append(_Token(start, end, group, unit))
    return out


def _within(n: Number, s: Number) -> bool:
    return abs(abs(n.value) - abs(s.value)) <= max(n.half, s.half)


def _restates(a: Number, b: Number) -> bool:
    """b restates a in another unit or as a percent: a times a power of ten, within the rounding of either."""
    return a.value != 0 and any(abs(abs(b.value) - abs(a.value) * 10.0**k) <= max(b.half, a.half * 10.0**k)
                                for k in range(-12, 13))


def unsourced(answer: str, conversation: list[str], tool_results: list[str]) -> list[dict[str, Any]]:
    """The numbers of the answer that no source states (see the module doc), in answer order: the text as written
    (with its unit), the value (a range: [low, high]), the unit and up to CONTEXT characters of context on each side.
    A written restatement (`118,3 Вт·ч = 0,1183 кВт·ч`, `0,7432 = 74,32 %`: an `=` or `≈` between two numbers, the
    second the first times a power of ten) is sourced when the number it restates is, and the other way round; `≈`
    before a result counts as `=`."""
    task = "\n".join(conversation)
    stated = [n for text in [*conversation, *tool_results] for t in _tokens(text) for n in t.numbers]
    comma = bool(_CYRILLIC.search(answer))
    # `≈` before a result as `=`, LaTeX units (\text{g/mol}) as blanks: the same length, so positions stay valid
    equations = _LATEX_UNIT.sub(lambda m: " " * len(m[0]), answer).replace("≈", "=")
    for strict in (True, False):  # operands and the stated result of every `a op b = c`, read both ways
        for a in find_arithmetic(equations, comma, strict):
            stated += read_numbers(a.expression)
            result = read_numbers(_decimal_points(answer[a.result_at : a.result_at + 80], comma))
            stated += result[:1] if result and result[0].start == 0 else []
    computed = computations(equations, comma)
    mask = quoted(answer, task) if task else []

    def sourced(n: Number, unit: str) -> bool:
        return any(_within(n, s) for s in stated) or any(  # a computed value, maybe written as a percent
            abs(abs(n.value) - abs(v)) <= n.half or unit == "%" and abs(abs(n.value) - 100 * abs(v)) <= n.half
            for v in computed)

    tokens = _tokens(answer)
    pairs = [(i, i + 1) for i, (a, b) in enumerate(zip(tokens, tokens[1:])) if len(a.numbers) == len(b.numbers) == 1
             and _EQUALS.fullmatch(answer, a.end, b.start) and _restates(a.numbers[0], b.numbers[0])]
    while True:  # a restatement of a sourced number is a source too, along chains (a = b = c)
        ok = [bool(mask) and all(mask[i] for n in t.numbers for i in range(n.start, n.end))  # quoted from the task
              or all(sourced(n, t.unit) for n in t.numbers) for t in tokens]
        if not (restated := [tokens[j if ok[i] else i].numbers[0] for i, j in pairs if ok[i] != ok[j]]):
            break
        stated += restated
    found = []
    for t, good in zip(tokens, ok):
        if not good:
            values = [n.value for n in t.numbers]
            found.append({"text": answer[t.start : t.end].strip(), "value": values[0] if len(values) == 1 else values,
                          "unit": t.unit,
                          "context": re.sub(r"\s+", " ", answer[max(0, t.start - CONTEXT) : t.end + CONTEXT])})
    return found


def trace_line(found: list[dict[str, Any]]) -> str:
    """The reasoning line: `> unsourced numbers: 2 (60–70 %, 0.5–0.7)`."""
    return f"> unsourced numbers: {len(found)}" + (f" ({', '.join(f['text'] for f in found)})" if found else "")
