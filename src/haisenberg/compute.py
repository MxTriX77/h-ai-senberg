"""Exact procedures without executing code: the fixed operations of the `compute` tool on dates, clock times and
durations, lists and sets, integer ranges, task schedules and small tables (agent-tools spec, Compute tool).

An entry is `name(arguments)`, read by a small parser (lists in brackets, items in quotes, bare text up to the next
comma or bracket); `schedule` reads its own argument text. Numbers are exact fractions, times whole seconds.
Every entry gives one line: `<entry> = <result>` or `<entry>: error: <what is wrong>`.
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Callable
from datetime import date, timedelta
from fractions import Fraction

MAX_CHARS = 4000  # longest entry read
MAX_LISTED = 50  # longest integer range written out
MAX_PATHS = 10  # critical paths listed

DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December")
UNITS = {"s": 1, "sec": 1, "second": 1, "min": 60, "minute": 60, "h": 3600, "hr": 3600, "hour": 3600,
         "d": 86400, "day": 86400, "wk": 604800, "week": 604800}  # time units in seconds

Value = str | list  # an argument: an item's text, or a list of values


class ComputeError(ValueError):
    """The entry cannot be read or computed; the message says what is wrong."""


def run(entry: object) -> str:
    """One result line for one entry of the `compute` tool."""
    if not isinstance(entry, str) or not entry.strip():
        return "error: an operation must be a non-empty string, e.g. weekday(2026-10-29)"
    entry = entry.strip()
    try:
        return f"{entry} = {evaluate(entry)}"
    except ComputeError as e:
        return f"{entry}: error: {e}"
    except (ArithmeticError, ValueError) as e:  # a number too large for a date, a float or a list
        return f"{entry}: error: cannot compute: {e}"


_ENTRY = re.compile(r"([A-Za-z_]\w*)\s*\((.*)\)", re.S)


def evaluate(entry: str) -> str:
    """The result of one operation; raises ComputeError."""
    if len(entry) > MAX_CHARS:
        raise ComputeError(f"operation longer than {MAX_CHARS} characters")
    m = _ENTRY.fullmatch(entry.strip())
    if not m:
        raise ComputeError("write an operation as name(arguments), e.g. weekday(2026-10-29)")
    op = _OPS.get(m[1].lower())
    if op is None:
        raise ComputeError(f"unknown operation {m[1]!r}; available: {', '.join(_OPS)}")
    return op(m[2]) if op is _schedule else op(parse_args(m[2]))


# --- arguments ------------------------------------------------------------------------------------------------

def parse_args(text: str) -> list[Value]:
    """The comma-separated arguments of an operation."""
    if not text.strip():
        return []
    return _values(text, 0, "", 0)[0]


def _skip(s: str, i: int) -> int:
    while i < len(s) and s[i].isspace():
        i += 1
    return i


def _values(s: str, i: int, close: str, depth: int) -> tuple[list[Value], int]:
    """Values separated by commas from s[i] up to `close` (']', or '' for the end) inside `depth` lists; returns
    them and the index after the close."""
    out: list[Value] = []
    i = _skip(s, i)
    if close and s[i:i + 1] == close:
        return out, i + 1
    while True:
        value, i = _value(s, i, depth)
        out.append(value)
        i = _skip(s, i)
        if i == len(s):
            if close:
                raise ComputeError("a list is not closed: missing ]")
            return out, i
        if s[i] == ",":
            i += 1
        elif close and s[i] == close:
            return out, i + 1
        else:
            raise ComputeError(f"unexpected {s[i]!r} (a missing comma or bracket?)")


def _value(s: str, i: int, depth: int) -> tuple[Value, int]:
    i = _skip(s, i)
    if s[i:i + 1] == "[":
        if depth >= 2:
            raise ComputeError("a list may hold lists (rows), but no deeper")
        return _values(s, i + 1, "]", depth + 1)
    if s[i:i + 1] in ('"', "'"):
        end = s.find(s[i], i + 1)
        if end < 0:
            raise ComputeError(f"a quote is not closed: missing {s[i]}")
        return s[i + 1:end], end + 1
    j = i
    while j < len(s) and s[j] not in ",[]":
        j += 1
    if s[j:j + 1] == "[":
        raise ComputeError("unexpected '[' inside an item (quote the item if it holds brackets)")
    if not s[i:j].strip():
        raise ComputeError("an empty item (two commas in a row, or a comma at the end?)")
    return s[i:j].strip(), j


def _arity(args: list[Value], usage: str, low: int, high: int | None = None) -> None:
    if not low <= len(args) <= (low if high is None else high):
        raise ComputeError(f"wrong number of arguments; use {usage}")


def _text(value: Value, what: str) -> str:
    if isinstance(value, list):
        raise ComputeError(f"{what} must be one value, not a list")
    return value


_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d{1,3})?")  # exponents kept small: exact ints


def number(text: str) -> Fraction | None:
    """The exact value of a plain number (`-3`, `2.5`, `1e3`), else None."""
    t = text.strip().replace("−", "-")
    return Fraction(t) if _NUMBER.fullmatch(t) else None


def _integer(value: Value, what: str) -> int:
    n = number(_text(value, what))
    if n is None or n.denominator != 1:
        raise ComputeError(f"{what} must be a whole number: {value}")
    return n.numerator


def show(q: Fraction, unit: str = "") -> str:
    """Exact: an integer, a finite decimal, or a fraction with its value to 10 significant digits."""
    u = f" {unit}" if unit else ""
    if q.denominator == 1:
        return f"{q.numerator}{u}"
    d, twos, fives = q.denominator, 0, 0
    while d % 2 == 0:
        d, twos = d // 2, twos + 1
    while d % 5 == 0:
        d, fives = d // 5, fives + 1
    if d != 1:
        return f"{q.numerator}/{q.denominator}{u} (≈ {float(q):.10g}{u})"
    k = max(twos, fives)
    digits = str(abs(q.numerator) * 10**k // q.denominator).rjust(k + 1, "0")
    return f"{'-' if q < 0 else ''}{digits[:-k]}.{digits[-k:]}{u}"


def _plural(n: int | Fraction, word: str, plural: str = "") -> str:
    return f"{show(Fraction(n))} {word if n in (1, -1) else plural or word + 's'}"


# --- dates ----------------------------------------------------------------------------------------------------

_DATE = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")


def _date(value: Value) -> date:
    t = _text(value, "a date")
    m = _DATE.fullmatch(t.strip())
    if not m:
        raise ComputeError(f"not a date: {t!r} (write YYYY-MM-DD)")
    y, mo, d = (int(g) for g in m.groups())
    if y < 1:
        raise ComputeError(f"unknown date {t} (years start at 1)")
    if not 1 <= mo <= 12:
        raise ComputeError(f"unknown date {t} (there is no month {mo})")
    last = calendar.monthrange(y, mo)[1]
    if not 1 <= d <= last:
        raise ComputeError(f"unknown date {t} ({MONTHS[mo - 1]} {y} has {last} days)")
    return date(y, mo, d)


def _dated(d: date) -> str:
    return f"{d.isoformat()} ({DAYS[d.weekday()]})"


def _shift(d: date, **delta: int) -> date:
    try:
        return d + timedelta(**delta)
    except OverflowError:
        raise ComputeError("the result is outside the years 1 to 9999") from None


def _weekday(args: list[Value]) -> str:
    _arity(args, "weekday(date)", 1)
    return DAYS[_date(args[0]).weekday()]


def _add(unit: str) -> Callable[[list[Value]], str]:
    def add(args: list[Value]) -> str:
        _arity(args, f"add_{unit}(date, n)", 2)
        d, n = _date(args[0]), _integer(args[1], "n")
        if unit == "months":
            return _add_months(d, n)
        return _dated(_shift(d, **{unit: n}))
    return add


def _add_months(d: date, n: int) -> str:
    y, m0 = divmod(d.month - 1 + n, 12)
    y += d.year
    if not 1 <= y <= 9999:
        raise ComputeError("the result is outside the years 1 to 9999")
    last = calendar.monthrange(y, m0 + 1)[1]
    result = _dated(date(y, m0 + 1, min(d.day, last)))
    if d.day <= last:
        return result
    return f"{result[:-1]}; clamped to the month end, {y:04d}-{m0 + 1:02d}-{d.day:02d} does not exist)"


def _days_between(args: list[Value]) -> str:
    _arity(args, "days_between(date1, date2)", 2)
    n = (_date(args[1]) - _date(args[0])).days
    earlier = "the second date is earlier; " if n < 0 else ""
    return f"{_plural(n, 'day')} ({earlier}exclusive; {abs(n) + 1} counting both dates)"


def _day_name(value: Value) -> int:
    t = _text(value, "a weekday").strip()
    w = t.lower()
    for i, name in enumerate(DAYS):
        if w in (name.lower(), name.lower() + "s", name[:3].lower()):
            return i
    raise ComputeError(f"not a weekday: {t!r} (write Monday or Mon, ...)")


def _count_weekdays(args: list[Value]) -> str:
    usage = "count_weekdays(date1, date2, weekday, ...)"
    if len(args) < 3:
        raise ComputeError(f"wrong number of arguments; use {usage}")
    start, end = _date(args[0]), _date(args[1])
    if start > end:
        raise ComputeError("the first date is after the second")
    names = [v for a in args[2:] for v in (a if isinstance(a, list) else [a])]
    days = list(dict.fromkeys(_day_name(v) for v in names))
    span = (end - start).days + 1
    counts = [span // 7 + ((w - start.weekday()) % 7 < span % 7) for w in days]
    per_day = ", ".join(f"{DAYS[w]} {c}" for w, c in zip(days, counts))
    return f"{sum(counts)} ({per_day}; both dates included)"


# --- clock times and durations --------------------------------------------------------------------------------

_HMS = re.compile(r"(\d+):([0-5]\d)(?::([0-5]\d))?")
_AMOUNT = re.compile(r"([-+]?(?:\d+(?:\.\d+)?|\.\d+))\s*([A-Za-z]+)")


def _unit(text: str) -> int:
    w = text.strip().lower()
    if w not in UNITS and w.endswith("s") and w[:-1] in UNITS:
        w = w[:-1]
    if w not in UNITS:
        raise ComputeError(f"unknown time unit {text.strip()!r} (use s, min, h, d or week)")
    return UNITS[w]


def _seconds(term: str) -> int:
    """A duration in whole seconds: H:MM, H:MM:SS or a number with a time unit (90 min, 1.5 h)."""
    t = term.strip()
    if m := _HMS.fullmatch(t):
        return int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3] or 0)
    if m := _AMOUNT.fullmatch(t):
        value = Fraction(m[1]) * _unit(m[2])
        if value.denominator != 1:
            raise ComputeError(f"{t} is not a whole number of seconds")
        return value.numerator
    raise ComputeError(f"not a time or duration: {t!r} (write H:MM, H:MM:SS or a number with a unit, e.g. 90 min)")


def _clock_time(term: str) -> int:
    t = term.strip()
    m = _HMS.fullmatch(t)
    if not m or int(m[1]) > 23:
        raise ComputeError(f"not a clock time: {t!r} (00:00 to 23:59)")
    return _seconds(t)


def _hms(seconds: int, pad: bool = False) -> str:
    h, rest = divmod(abs(seconds), 3600)
    m, s = divmod(rest, 60)
    text = f"{h:02d}:{m:02d}" if pad else f"{h}:{m:02d}"
    return f"{'-' if seconds < 0 else ''}{text}{f':{s:02d}' if s else ''}"


def _terms(value: Value, usage: str) -> list[tuple[int, str]]:
    """(sign, term) of `t ± t ± ...`; the first term may carry a sign."""
    if not isinstance(value, str):
        raise ComputeError(f"use {usage}")
    terms, sign = [], 1
    for i, token in enumerate(re.split(r"([+\-−])", value)):
        if i % 2:
            sign = 1 if token == "+" else -1
        elif token.strip():
            terms.append((sign, token))
            sign = 1
        elif i:
            raise ComputeError(f"a sign without a time or duration after it; use {usage}")
    if not terms:
        raise ComputeError(f"use {usage}")
    return terms


def _clock(args: list[Value]) -> str:
    usage = "clock(09:45 + 2:30 - 0:15)"
    _arity(args, usage, 1)
    (sign, first), *rest = _terms(args[0], usage)
    if sign < 0:
        raise ComputeError(f"start with a clock time; use {usage}")
    total = _clock_time(first) + sum(s * _seconds(t) for s, t in rest)
    days, seconds = divmod(total, 86400)
    note = {0: "", 1: " (next day)", -1: " (previous day)"}.get(days)
    if note is None:
        note = f" ({days} days later)" if days > 0 else f" ({-days} days earlier)"
    return _hms(seconds, pad=True) + note


def _duration(args: list[Value]) -> str:
    usage = "duration(1:30 + 2:45 - 0:50)"
    _arity(args, usage, 1)
    total = sum(s * _seconds(t) for s, t in _terms(args[0], usage))
    size = _plural(total // 60, "minute") if total % 60 == 0 else _plural(total, "second")
    if abs(total) >= 86400:
        days, rest = divmod(abs(total), 86400)
        size += f"; {'-' if total < 0 else ''}{_plural(days, 'day')} {_hms(rest)}"
    return f"{_hms(total)} ({size})"


def _time_between(args: list[Value]) -> str:
    _arity(args, "time_between(time1, time2)", 2)
    start, end = (_clock_time(_text(a, "a clock time")) for a in args)
    total = (end - start) % 86400
    size = _plural(total // 60, "minute") if total % 60 == 0 else _plural(total, "second")
    return f"{_hms(total)} ({size}{'; past midnight' if end < start else ''})"


def _convert(args: list[Value]) -> str:
    usage = "convert(value, unit, unit), e.g. convert(90, min, h)"
    _arity(args, usage, 2, 3)
    texts = [_text(a, "each argument") for a in args]
    if len(texts) == 2:  # convert(90 min, h)
        m = _AMOUNT.fullmatch(texts[0].strip())
        if not m:
            raise ComputeError(f"use {usage}")
        texts = [m[1], m[2], texts[1]]
    value = number(texts[0])
    if value is None:
        raise ComputeError(f"not a number: {texts[0]!r}")
    return show(value * _unit(texts[1]) / _unit(texts[2]), texts[2].strip())


# --- lists and sets -------------------------------------------------------------------------------------------

def _items(args: list[Value], usage: str) -> list[str]:
    if len(args) == 1 and isinstance(args[0], list):
        args = args[0]
    if any(isinstance(a, list) for a in args):
        raise ComputeError(f"give the items, or one list of them; use {usage}")
    return args


def _key(item: str) -> tuple:
    n = number(item)
    return ("n", n) if n is not None else ("t", item)


def _order(items: list[str]) -> tuple[Callable[[str], object], str]:
    """How items compare: as numbers, as times, else as text ignoring case."""
    if items and all(number(x) is not None for x in items):
        return number, "as numbers"
    if items and all(_HMS.fullmatch(x.strip()) for x in items):
        return _seconds, "as times"
    return str.casefold, "as text, ignoring case"


def _quote(item: str) -> str:
    return f'"{item}"' if any(c in item for c in ",[]") else item


def _join(items: list[str]) -> str:
    return ", ".join(_quote(x) for x in items) if items else "(empty list)"


def _first_seen(items: list[str]) -> dict[tuple, str]:
    """Each distinct item (by _key) with its first occurrence, in order."""
    seen: dict[tuple, str] = {}
    for x in items:
        seen.setdefault(_key(x), x)
    return seen


def _sorter(descending: bool) -> Callable[[list[Value]], str]:
    def sort(args: list[Value]) -> str:
        items = _items(args, "sort(a, b, c) or sort([a, b, c])")
        if not items:
            return "(empty list)"
        key, how = _order(items)
        return f"{_join(sorted(items, key=key, reverse=descending))} ({how}{', descending' if descending else ''})"
    return sort


def _unique(args: list[Value]) -> str:
    items = _items(args, "unique(a, b, a) or unique([a, b, a])")
    distinct = list(_first_seen(items).values())
    return f"{_join(distinct)} ({len(distinct)} distinct of {_plural(len(items), 'item')})" if items else "(empty list)"


def _count(args: list[Value]) -> str:
    items = _items(args, "count(a, b, c) or count([a, b, c])")
    return f"{_plural(len(items), 'item')} ({len(_first_seen(items))} distinct)"


def _position(args: list[Value]) -> str:
    usage = "position(item, [list])"
    _arity(args, usage, 2)
    if not isinstance(args[1], list) or any(isinstance(x, list) for x in args[1]):
        raise ComputeError(f"the second argument must be a list; use {usage}")
    key = _key(_text(args[0], "the item"))
    found = [i for i, x in enumerate(args[1], 1) if _key(x) == key]
    size = _plural(len(args[1]), "item")
    if not found:
        return f"not found (0 matches in {size})"
    return f"{', '.join(map(str, found))} (1-based; {_plural(len(found), 'match', 'matches')} in {size})"


def _lists(args: list[Value], usage: str) -> list[list[str]]:
    if len(args) < 2 or not all(isinstance(a, list) and not any(isinstance(x, list) for x in a) for a in args):
        raise ComputeError(f"give two or more lists; use {usage}")
    return args


def _union(args: list[Value]) -> str:
    lists = _lists(args, "union([a, b], [b, c])")
    items = list(_first_seen([x for lst in lists for x in lst]).values())
    return _sized(items)


def _intersection(args: list[Value]) -> str:
    first, *others = _lists(args, "intersection([a, b], [b, c])")
    keys = [{_key(x) for x in lst} for lst in others]
    return _sized([x for k, x in _first_seen(first).items() if all(k in ks for ks in keys)])


def _difference(args: list[Value]) -> str:
    first, *others = _lists(args, "difference([a, b, c], [b])")
    keys = {_key(x) for lst in others for x in lst}
    return _sized([x for k, x in _first_seen(first).items() if k not in keys])


def _sized(items: list[str]) -> str:
    return f"{_join(items)} ({_plural(len(items), 'item')})" if items else "(empty list)"


def _range(args: list[Value]) -> str:
    _arity(args, "range(a, b) or range(a, b, step)", 2, 3)
    a, b = _integer(args[0], "a"), _integer(args[1], "b")
    step = _integer(args[2], "the step") if len(args) == 3 else (1 if b >= a else -1)
    if step == 0:
        raise ComputeError("the step must not be 0")
    if (b - a) * step < 0:
        raise ComputeError(f"a step of {step} never reaches {b} from {a}")
    n = (b - a) // step + 1
    listed = f": {', '.join(str(a + i * step) for i in range(n))}" if n <= MAX_LISTED else ""
    ends = (f"{a} to {b}, both included" if abs(step) == 1
            else f"{a} to {b} in steps of {step}; first {a}, last {a + (n - 1) * step}")
    hidden = f"; list not shown, more than {MAX_LISTED}" if not listed else ""
    return f"{_plural(n, 'integer')}{listed} ({ends}{hidden})"


# --- schedules ------------------------------------------------------------------------------------------------

_TASK_REST = re.compile(r"\s*(\S+)\s*(?:after\s+(.+?))?\s*", re.S | re.I)
_SCHEDULE_USE = "NAME: DURATION after NAME, NAME; ..., e.g. schedule(A: 3; B: 2 after A)"


def _schedule(text: str) -> str:
    """Earliest and latest start and finish, slack, critical paths and total duration of tasks with dependencies."""
    durations: dict[str, Fraction] = {}
    deps: dict[str, list[str]] = {}
    for part in re.split(r"[;\n]", text):
        if not part.strip():
            continue
        name, colon, rest = (p.strip() for p in part.partition(":"))
        m = _TASK_REST.fullmatch(rest)
        if not colon or not name or "," in name or not m:
            raise ComputeError(f"cannot read task {part.strip()!r}; write {_SCHEDULE_USE}")
        d = number(m[1])
        if d is None or d < 0:
            raise ComputeError(f"the duration of {name} is not a number of at least 0: {m[1]}")
        if name in durations:
            raise ComputeError(f"task {name} is defined twice")
        durations[name] = d
        after = [x.strip() for x in m[2].split(",")] if m[2] else []
        if "" in after:
            raise ComputeError(f"an empty task name after {name}'s 'after'")
        deps[name] = list(dict.fromkeys(after))
    if not durations:
        raise ComputeError(f"no tasks; write {_SCHEDULE_USE}")
    for name, after in deps.items():
        for d in after:
            if d not in durations:
                raise ComputeError(f"unknown task {d!r} ({name} is after it)")
    succ: dict[str, list[str]] = {n: [] for n in durations}
    for n in durations:
        for d in deps[n]:
            succ[d].append(n)
    order = _topological(deps, succ)
    es: dict[str, Fraction] = {}
    for n in order:  # forward pass: a task starts when its last predecessor finishes
        es[n] = max((es[d] + durations[d] for d in deps[n]), default=Fraction(0))
    ef = {n: es[n] + durations[n] for n in durations}
    total = max(ef.values())
    ls: dict[str, Fraction] = {}
    for n in reversed(order):  # backward pass: a task must finish by its successors' latest start
        ls[n] = min((ls[s] for s in succ[n]), default=total) - durations[n]
    slack = {n: ls[n] - es[n] for n in durations}
    paths = _critical_paths(deps, succ, es, ef, slack)
    shown = paths[:MAX_PATHS]
    label = "critical path" if len(paths) == 1 else (
        "critical paths" if len(paths) <= MAX_PATHS else f"critical paths (first {MAX_PATHS} shown)")
    rows = [f"{n}: earliest start {show(es[n])}, earliest finish {show(ef[n])}, latest start {show(ls[n])}, "
            f"latest finish {show(ls[n] + durations[n])}, slack {show(slack[n])}" for n in durations]
    return "; ".join([f"total duration {show(total)}",
                      f"{label} {', '.join(' → '.join(p) for p in shown)}", *rows])


def _topological(deps: dict[str, list[str]], succ: dict[str, list[str]]) -> list[str]:
    """The tasks with every task after its predecessors; raises ComputeError naming a dependency cycle."""
    waiting = {n: len(after) for n, after in deps.items()}
    ready = [n for n in deps if not waiting[n]]
    order: list[str] = []
    while ready:
        n = ready.pop(0)
        order.append(n)
        for s in succ[n]:
            waiting[s] -= 1
            if not waiting[s]:
                ready.append(s)
    if len(order) == len(deps):
        return order
    left = [n for n in deps if n not in order]  # each still waits for a task of `left`: follow those back to a repeat
    path, at = [left[0]], {left[0]: 0}
    while True:
        nxt = next(d for d in deps[path[-1]] if d in left)
        if nxt in at:
            cycle = [*path[at[nxt]:], nxt][::-1]  # written in the order the tasks would run
            raise ComputeError(f"dependency cycle: {' → '.join(cycle)}")
        at[nxt] = len(path)
        path.append(nxt)


def _critical_paths(deps, succ, es, ef, slack) -> list[list[str]]:
    """Chains of zero-slack tasks from a task without predecessors, each next task starting when the one before
    finishes (at most MAX_PATHS + 1)."""
    paths: list[list[str]] = []
    stack = [[n] for n in reversed(list(deps)) if not deps[n] and slack[n] == 0]
    while stack and len(paths) <= MAX_PATHS:
        path = stack.pop()
        nxt = [s for s in succ[path[-1]] if slack[s] == 0 and es[s] == ef[path[-1]]]
        if nxt:
            stack.extend([*path, s] for s in reversed(nxt))
        else:
            paths.append(path)
    return paths


# --- tables ---------------------------------------------------------------------------------------------------

def _rows(args: list[Value], usage: str) -> tuple[int, list[list[str]]]:
    if not args:
        raise ComputeError(f"wrong number of arguments; use {usage}")
    col, rows = _integer(args[0], "the column"), args[1:]
    if col < 1:
        raise ComputeError("columns are numbered from 1")
    if len(rows) == 1 and isinstance(rows[0], list) and rows[0] and all(isinstance(r, list) for r in rows[0]):
        rows = rows[0]  # one list of rows
    if not rows:
        raise ComputeError(f"no rows; use {usage}")
    for i, row in enumerate(rows, 1):
        if not isinstance(row, list) or any(isinstance(c, list) for c in row):
            raise ComputeError(f"row {i} must be a list of cells, e.g. [Ann, 31]; use {usage}")
        if len(row) < col:
            raise ComputeError(f"row {i} has no column {col}")
    return col, rows


def _row_sorter(descending: bool) -> Callable[[list[Value]], str]:
    def sort_rows(args: list[Value]) -> str:
        col, rows = _rows(args, "sort_rows(column, [row], [row], ...)")
        key, how = _order([r[col - 1] for r in rows])
        ordered = sorted(rows, key=lambda r: key(r[col - 1]), reverse=descending)
        return (f"{', '.join(f'[{_join(r)}]' for r in ordered)}"
                f" (by column {col}, {how}{', descending' if descending else ''})")
    return sort_rows


def _column_stats(args: list[Value]) -> str:
    col, rows = _rows(args, "column_stats(column, [row], [row], ...)")
    values = []
    for i, row in enumerate(rows, 1):
        if (v := number(row[col - 1])) is None:
            raise ComputeError(f"row {i}, column {col} is not a number: {row[col - 1]!r}")
        values.append(v)

    def at(v: Fraction) -> str:
        found = [str(i) for i, x in enumerate(values, 1) if x == v]
        return f"{show(v)} ({'row' if len(found) == 1 else 'rows'} {', '.join(found)})"

    return f"sum {show(sum(values))}, min {at(min(values))}, max {at(max(values))}; {_plural(len(rows), 'row')}"


_OPS: dict[str, Callable] = {
    "weekday": _weekday, "add_days": _add("days"), "add_weeks": _add("weeks"), "add_months": _add("months"),
    "days_between": _days_between, "count_weekdays": _count_weekdays,
    "clock": _clock, "duration": _duration, "time_between": _time_between, "convert": _convert,
    "sort": _sorter(False), "sort_desc": _sorter(True), "unique": _unique, "count": _count, "position": _position,
    "union": _union, "intersection": _intersection, "difference": _difference, "range": _range,
    "schedule": _schedule,
    "sort_rows": _row_sorter(False), "sort_rows_desc": _row_sorter(True), "column_stats": _column_stats,
}
