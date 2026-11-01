"""Arithmetic without executing code: a whitelisted evaluator for the `calculate` tool, and a finder for
`expression = result` in a draft so the verifier can recompute it (claim-verifier, Arithmetic check)."""

from __future__ import annotations

import ast
import json
import math
import operator
import re
from collections.abc import Iterator
from dataclasses import dataclass

MAX_CHARS = 500  # longest expression evaluated
MAX_EXPONENT = 1000

_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_FUNCS = {
    "sqrt": math.sqrt, "exp": math.exp, "ln": math.log, "log": math.log10, "log10": math.log10, "log2": math.log2,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "abs": abs, "min": min, "max": max,
}
_CONSTS = {"pi": math.pi, "e": math.e}


class CalcError(ValueError):
    """The expression is not plain arithmetic, or it cannot be computed."""


def _pow(base: float, exponent: float) -> float:
    if abs(exponent) > MAX_EXPONENT:
        raise CalcError("power too large")
    result = base**exponent
    if isinstance(result, complex):
        raise CalcError("complex result")
    return result


_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.Pow: _pow, ast.Mod: operator.mod}


def normalize(expression: str, decimal_comma: bool = False) -> str:
    """Typographic and LaTeX arithmetic as Python syntax (units are not handled here). Digits grouped in
    threes with spaces (20 000, 2\\,450) are one number. A comma between digits is a decimal comma, except
    before exactly three digits (1,900) unless `decimal_comma`."""
    s = expression.strip().replace("{,}", ",")  # LaTeX decimal comma: 1{,}20
    s = re.sub(r"\\(?:left|right)\b", "", s)
    for _ in range(3):  # nested \frac and \sqrt
        s = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", s)
        s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\(?:times|cdot)", "*", s).replace("\\div", "/").replace("\\%", "%").replace("\\pi", "pi")
    s = re.sub(r"\\[,;:! ]", " ", s)
    s = re.sub(r"(?<![\d.,])\d{1,3}(?:[ \xa0  ]\d{3})+(?!\d)", lambda m: re.sub(r"\D", "", m[0]), s)  # 20 000
    s = re.sub(r"([\d)\]])\s*([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)", lambda m: f"{m[1]}^({m[2].translate(_SUP)})", s)
    s = re.sub(r"√\s*\(", "sqrt(", s)
    s = re.sub(r"√\s*(\d+(?:[.,]\d+)?)", r"sqrt(\1)", s)
    s = s.translate(str.maketrans({"×": "*", "·": "*", "⋅": "*", "∙": "*", "÷": "/", "−": "-", "–": "-", "{": "(",
                                   "}": ")", "[": "(", "]": ")"}))
    s = re.sub(r"(?<=[\d)])\s*[xX]\s*(?=[\d(])", "*", s)
    if not re.search(r"\b(?:min|max)\s*\(", s):  # there a comma separates arguments
        if not decimal_comma:
            s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)  # every group: 1,234,567
        s = re.sub(r"(\d),(\d)", r"\1.\2", s)
    return s.replace("^", "**")


def evaluate(expression: str, decimal_comma: bool = False) -> float:
    """The value of a plain arithmetic expression; raises CalcError for anything else."""
    if len(expression) > MAX_CHARS:
        raise CalcError(f"expression longer than {MAX_CHARS} characters")
    try:
        tree = ast.parse(normalize(expression, decimal_comma), mode="eval")
    except SyntaxError:
        raise CalcError("not an arithmetic expression") from None
    try:
        value = float(_eval(tree.body))
    except (ArithmeticError, ValueError, TypeError) as e:  # ZeroDivisionError, OverflowError, math domain error
        raise CalcError(str(e) or type(e).__name__) from None
    if not math.isfinite(value):
        raise CalcError("result is not a finite number")
    return value


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _BINARY[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*(_eval(a) for a in node.args))
    raise CalcError(f"not allowed in a calculation: {ast.unparse(node)[:40]!r}")


def show(value: float) -> str:
    return f"{value:.10g}"


def run_tool(arguments: str) -> tuple[str, list[str]]:
    """The `calculate` tool: (result text for the model, one line for the reasoning)."""
    try:
        args = json.loads(arguments or "{}")
        expression = args.get("expression") if isinstance(args, dict) else None
        if not isinstance(expression, str) or not expression.strip():
            raise CalcError("'expression' must be a non-empty string")
        text = f"{expression.strip()} = {show(evaluate(expression))}"
    except json.JSONDecodeError:
        text = "Error: arguments are not valid JSON"
    except CalcError as e:
        text = f"Error: {e}"
    return text, [text]


# --- `expression = result` in a draft ---------------------------------------------------------------

_NUM = r"\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?"
_UNIT_CHARS = r"A-Za-zА-Яа-яЁёµμΩÅ°⁰¹²³⁴⁵⁶⁷⁸⁹⁻"
_UNIT = rf"(?:\s?(?:%|(?![eE][-+]?\d)[{_UNIT_CHARS}]+(?:[·*/][{_UNIT_CHARS}]+|\*\*\(?-?\d+\)?|-?\d)*))"  # not "e7"
_OPEN = r"(?:(?:sqrt|ln|log|exp)?\s*\(\s*)"
_OPERAND = rf"[-+]?{_OPEN}*[-+]?{_NUM}(?:{_UNIT})?(?:\s*\))*(?:{_UNIT})?"
_OP = r"\s*(?:\*\*|[-+*/])\s*"
_EXPR = re.compile(rf"(?<![\w.,]){_OPERAND}(?:{_OP}{_OPERAND})+\s*$")
_EQUALS = re.compile(r"(?<![<>=!≤≥])=(?![=>])")
_RESULT_LEAD = re.compile(r"[ \t]*(?:\\[,;! ]|[$])?[ \t]*\{?[ \t]*")
_NEXT_OP = re.compile(  # after a stated result, maybe its % or unit, an operator: "= 0.0500 L × 1.50", "= 2.0 \times"
    rf"[ \t]*(?:%|(?:\\[,;! ]|[ \t])*(?:\\(?:text|mathrm)\s*\{{[^{{}}\n]*\}}|[{_UNIT_CHARS}][{_UNIT_CHARS}·⋅/^\d.-]*+))?"
    r"[ \t]*(?:[×/·÷+^−-]|\*(?!\*)|\\(?:times|cdot|div|frac|ln|log|sqrt)\b)(?!\s*$)")
_SCI_AFTER = re.compile(r"[ \t]*(?:[×x·*]|\\times|\\cdot)[ \t]*10(?:\^|\*\*|[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺])")  # "= 1.25 × 10²³": one number
_DATE = re.compile(  # a day and a month ("3 марта + 98 дней = 9 июня"): a date, not a number with a unit
    r"\d[ \t]*(?:(?i:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?"
    r"|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?![A-Za-z])|(?i:январ|феврал|март|апрел|июн|июл|август|сентябр"
    r"|октябр|ноябр|декабр)\w*|(?i:ма[йяю])(?![А-Яа-яЁё]))")
_RANGE = re.compile(r"(?<![\d.,])(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)(?![\d.,])")  # "10-12", "3–7" (normalized): a range
_OPERAND_UNIT = re.compile(rf"\d\s?([{_UNIT_CHARS}][{_UNIT_CHARS}/·*²³]*)")
_SI_PREFIX = "kMGTPmµμnc"


def _mixed_units(expression: str) -> bool:
    """Operands in different units that the check would add unconverted (`2 h + 30 min`, `1 m + 50 cm`), or in
    one unit with different prefixes (`1 TB / 1 GB`): without conversions in code the value is not the result."""
    units = {u.rstrip("*") for u in _OPERAND_UNIT.findall(re.sub(r"(?<=\d)[eE][-+]?\d+", "", expression))}
    if len(units) < 2:
        return False
    core = re.sub(r"(?<=\d)[eE][-+]?\d+", "", expression).lstrip("+-( ")
    if re.search(r"[+\-]\s*[\d(]", core):
        return True
    bases = [u[1:] if len(u) > 1 and u[0] in _SI_PREFIX else u for u in units]
    return len(set(bases)) < len(bases)


_NUMBER_IN_UNITS = re.compile(rf"({_NUM}|\))(?:{_UNIT})")
_LIST_MARKER = re.compile(r"[ \t]*(?:>[ \t]*)*(?:(?:[-*+]|\d+[.)])[ \t]+)?")  # "- ", "  * ", "> 1. ": not a sign

# Strict reading (verifier.arithmetic_strict_parse): only a whole left side, `a × 10ⁿ` as one number.
_CURRENCY = re.compile(r"\\?[$€£¥₹₽]\s?(?=\d)")  # "$1,200": the sign is not part of the arithmetic
_POW10 = re.compile(r"(?<![\w.*])(?<!\*\*\s)(?:(\d+(?:\.\d+)?)\s*\*\s*)?10\s*\*\*\s*(?:\(\s*([-+]?\d+)\s*\)|([-+]?\d+)(?![\d.]))"
                    r"(?!\s*\*\*)")  # "6.4 * 10**(7)", "10**9": one number, unless part of a power
_SCALE_WORDS = (r"hundreds?|thousands?|millions?|billions?|trillions?|dozens?|lakhs?|crores?|bn|mn|mln|bln"
                r"|сот[ен]\w*|тыс\w*|млн|миллион\w*|млрд|миллиард\w*|трлн|триллион\w*|дюжин\w*")
_SCALED = re.compile(rf"\d(?:[kKMBT]|\s?(?i:{_SCALE_WORDS}))(?![{_UNIT_CHARS}])")  # "1.3B", "20k", "2 million"
_EMPHASIS = re.compile(r"(?:(?<=\S)[*_]\s+|(?<!\S)[*_])$")  # Markdown "*" or "_" (italics) around the equation
_MULTIPLIERS = (rf"half|third|quarter|twice|thrice|double|triple|{_SCALE_WORDS}"
                r"|половин\w*|трет\w*|четверт\w*|дважды|трижды|вдвое|втрое")
_PART_BEFORE = re.compile(  # the text before the expression belongs to the math: an operator, "(", "½", "π", "half of",
    # "per" (22 g per 100 g × 0.40), a dash written as a minus (1 — 0.95^12)
    rf"(?:[-+*/(—€£¥₹₽¼½¾⅐-⅞π]|\bpi|(?<![A-Za-zА-Яа-яЁё])(?i:{_MULTIPLIERS}|per)(?:\s+(?:of|от))?)\s*$")


def _powers_of_ten(s: str) -> str:
    return _POW10.sub(lambda m: f"{m[1] or 1}e{m[2] or m[3]}", s)


@dataclass
class Arithmetic:
    """One `<expression> = <result>`. A % sign can be read two ways, so the left side has two values:
    `fraction` reads `a%` as a/100 and drops a final `× 100%` (which only says "as a percent");
    `plain` drops every % sign. They are equal when the left side has no % sign."""

    expression: str  # as written in the draft (normalized)
    fraction: float
    plain: float
    result_at: int  # where the stated result starts in the draft (read with verifier.read_numbers)
    as_percent: bool  # the left side ends in `× 100%`: the result is the fraction written as a percent
    has_percent: bool  # some operand carries a % sign (other than that final `× 100%`)
    times_100: bool  # the left side multiplies by 100 somewhere (its value may already be a percent)


def find_arithmetic(text: str, decimal_comma: bool = False, strict: bool = True) -> Iterator[Arithmetic]:
    """Every `<arithmetic> = <number>` in the text, with the computed value of the left side.
    Expressions with names other than units and functions, or that cannot be computed, are skipped, and so are
    clock times and dates, and a step of a chain (a "result" followed by an operator, after its unit or %).
    With `strict`, a currency sign before a number is dropped, `a × 10ⁿ` and `10ⁿ` are one number (a left side
    that is one number is not arithmetic), and an expression is skipped when the text before it belongs to the
    math (an operator, "(", "½", "π", "half of", "million"; not Markdown emphasis) or an operand has a scale
    suffix (1.3B, 20k, 2 million): only a whole left side is checked."""
    for eq in _EQUALS.finditer(text):
        line_start = max(text.rfind("\n", 0, eq.start()), text.rfind("=", 0, eq.start())) + 1
        if line_start == text.rfind("\n", 0, eq.start()) + 1:  # the left side starts its line
            line_start = _LIST_MARKER.match(text, line_start).end()
        lhs = text[max(line_start, eq.start() - 200) : eq.start()]
        if strict:  # "**" in the draft is Markdown bold (a power is "^" or a superscript); "$1,200" is 1200
            lhs = _CURRENCY.sub("", lhs.replace("**", "").replace("__", ""))
        lhs = normalize(lhs, decimal_comma)
        m = _EXPR.search(lhs)
        if not m or re.search(r"\d\s+$|[\d)]$", lhs[: m.start()]):  # "6.626 070 15 × 2", "2 (3 + 4)", "2(1)(1) + 1":
            continue  # not one reading
        if re.search(r"\d[:：]$", lhs[: m.start()]) and lhs[m.start()].isdigit():  # "10:50 - 1 hour": minutes of a
            continue  # clock time, not a number ("A2: (1000 − 800) / 800" is still checked)
        expression = m.group().strip()
        if _DATE.search(expression) or _mixed_units(expression) or \
                any(float(a) < float(b) for a, b in _RANGE.findall(expression)):  # a range "10-12" is no subtraction
            continue
        if strict:
            before = lhs[: m.start()]
            if not m.group()[:1].isspace():  # italics hug the text; "P_1 * (1 - 0.20)" has a times sign before it
                before = _EMPHASIS.sub("", before)
            if _PART_BEFORE.search(before) or _SCALED.search(expression):
                continue
            expression = _powers_of_ten(expression)
            if not _EXPR.search(expression) and not re.search(r"\b(?:sqrt|ln|log|exp)\s*\(", expression):
                continue  # one number (125 × 10⁶ bytes/s): a restatement or a unit conversion, not arithmetic
        lead = _RESULT_LEAD.match(text, eq.end())
        at = lead.end()
        if not re.match(r"[-−]?\d", text[at : at + 2]):
            continue
        number_end = re.match(r"[-−]?[\d.,\s]*\d", text[at:]).end()
        if re.match(r"[:：]\d|\{,\}\d", text[at + number_end :]) or _DATE.match(text, at):  # "= 01:00", "= 9 июня",
            continue  # "= 377{,}1" (LaTeX decimal comma: not read as one number here)
        if _NEXT_OP.match(text, at + number_end) and not _SCI_AFTER.match(text, at + number_end):
            continue  # "= 2 × 3", "= 0.0500 L × 1.50": the right side is an expression, not a result
        as_percent = bool(_AS_PERCENT.search(expression))
        core = _AS_PERCENT.sub("", expression) if as_percent else expression
        try:
            fraction = evaluate(_python(core, percent_is_fraction=True))
            plain = evaluate(_python(expression, percent_is_fraction=False))
        except CalcError:
            continue
        yield Arithmetic(expression, fraction, plain, at, as_percent, "%" in core,
                         bool(re.search(r"\*\s*100(?:\.0+)?(?![\d.])", expression)))


_ANY_EXPR = re.compile(rf"(?<![\w.,]){_OPERAND}(?:{_OP}{_OPERAND})+")


def computations(text: str, decimal_comma: bool = False) -> list[float]:
    """The value of every arithmetic expression written in the text, with or without a result after it
    (`96 °C − (−18 °C)` gives 114), under both readings of its % signs; what cannot be computed is skipped."""
    values = []
    for m in _ANY_EXPR.finditer(normalize(text, decimal_comma)):
        for percent_is_fraction in (True, False):
            try:
                values.append(evaluate(_python(m.group(), percent_is_fraction)))
            except CalcError:
                pass
    return values


_AS_PERCENT = re.compile(r"\s*\*\s*100(?:\.0+)?\s*%\s*$")  # a final "× 100%"

_SI_PREFIXES = ("G", "M", "k", "m", "µ", "μ", "u", "n", "Г", "М", "к", "мк", "м", "н")  # 10^±3, ±6, ±9
_BASE_UNITS = {
    "W", "Wh", "J", "V", "A", "Ah", "g", "s", "L", "l", "m", "Pa", "N", "Hz", "Ω", "C", "F", "eV", "mol", "bar", "cal",
    "Вт", "Втч", "Дж", "В", "А", "Ач", "г", "с", "л", "м", "Па", "Н", "Гц", "Ом", "Кл", "Ф", "эВ", "моль", "бар", "кал",
}


def prefixed(text: str) -> bool:
    """Some word of the text is a unit with an SI prefix (kW, mA, мг; not W, m, mol or min)."""
    return any(w not in _BASE_UNITS and any(w.startswith(p) and w[len(p) :] in _BASE_UNITS for p in _SI_PREFIXES)
               for w in re.findall(r"[A-Za-zА-Яа-яЁёµμΩ]+", text))


_SCALE_LETTERS = ("k", "K", "M", "G", "T", "P", "m", "µ", "μ", "u", "n", "c", "к", "М", "Г", "Т", "мк", "м", "н")


def _scaled_base(word: str) -> str | None:
    """The unit under a scale prefix that the prefix readings do not know (MB → B, Gbit → bit, cm → m), else None."""
    if word in _BASE_UNITS or prefixed(word):
        return None
    return next((word[len(p) :] for p in _SCALE_LETTERS if word.startswith(p) and 1 <= len(word) - len(p) <= 3), None)


def _same_unit(a: str, b: str) -> bool:
    """Equal without case and plural s, or a one-letter symbol that starts the other (B ~ bytes, B ~ bit)."""
    a, b = (w.lower()[:-1] if len(w) > 2 and w.lower().endswith("s") else w.lower() for w in (a, b))
    return a == b or (min(len(a), len(b)) == 1 and a[0] == b[0])


def converted(expression: str, units: str) -> bool:
    """A unit of the result (`units`) that the left side does not write, where one side has a unit with a scale prefix
    the prefix readings do not know and the other the same unit without it or with another prefix (bytes/s → MB/s,
    bits → Gbit, Gbit → GB, m → cm): a unit conversion, not plain arithmetic."""
    words = r"[A-Za-zА-Яа-яЁёµμΩ]+"
    left = set(re.findall(words, expression))
    for r in set(re.findall(words, units)) - left:
        for w in left:
            rb, wb = _scaled_base(r), _scaled_base(w)
            if (rb and _same_unit(rb, wb or w)) or (wb and _same_unit(wb, r)):
                return True
    return False


def _python(expression: str, percent_is_fraction: bool) -> str:
    """Units dropped; `a%` read as (a/100), or as plain a when not percent_is_fraction."""

    def unit(m: re.Match[str]) -> str:
        if percent_is_fraction and m.group().rstrip().endswith("%"):
            return f"({m.group(1)}/100)"
        return m.group(1)

    return _NUMBER_IN_UNITS.sub(unit, expression)
