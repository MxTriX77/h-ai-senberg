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
    s = expression.strip()
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
_UNIT = rf"(?:\s?(?:%|[{_UNIT_CHARS}]+(?:[·*/][{_UNIT_CHARS}]+|\*\*\(?-?\d+\)?|-?\d)*))"
_OPEN = r"(?:(?:sqrt|ln|log|exp)?\s*\(\s*)"
_OPERAND = rf"[-+]?{_OPEN}*[-+]?{_NUM}(?:{_UNIT})?(?:\s*\))*(?:{_UNIT})?"
_OP = r"\s*(?:\*\*|[-+*/])\s*"
_EXPR = re.compile(rf"(?<![\w.,]){_OPERAND}(?:{_OP}{_OPERAND})+\s*$")
_EQUALS = re.compile(r"(?<![<>=!≤≥])=(?![=>])")
_RESULT_LEAD = re.compile(r"[ \t]*(?:\\[,;! ]|[$])?[ \t]*\{?[ \t]*")
_NEXT_OP = re.compile(r"[ \t]*[×*/·÷+^−-](?!\s*$)")
_NUMBER_IN_UNITS = re.compile(rf"({_NUM}|\))(?:{_UNIT})")
_LIST_MARKER = re.compile(r"[ \t]*(?:>[ \t]*)*(?:(?:[-*+]|\d+[.)])[ \t]+)?")  # "- ", "  * ", "> 1. ": not a sign


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


def find_arithmetic(text: str, decimal_comma: bool = False) -> Iterator[Arithmetic]:
    """Every `<arithmetic> = <number>` in the text, with the computed value of the left side.
    Expressions with names other than units and functions, or that cannot be computed, are skipped."""
    for eq in _EQUALS.finditer(text):
        line_start = max(text.rfind("\n", 0, eq.start()), text.rfind("=", 0, eq.start())) + 1
        if line_start == text.rfind("\n", 0, eq.start()) + 1:  # the left side starts its line
            line_start = _LIST_MARKER.match(text, line_start).end()
        lhs = normalize(text[max(line_start, eq.start() - 200) : eq.start()], decimal_comma)
        m = _EXPR.search(lhs)
        if not m or re.search(r"\d\s+$", lhs[: m.start()]):  # "6.626 070 15 × 2", "2 (3 + 4)": not one reading
            continue
        lead = _RESULT_LEAD.match(text, eq.end())
        at = lead.end()
        if not re.match(r"[-−]?\d", text[at : at + 2]):
            continue
        number_end = re.match(r"[-−]?[\d.,\s]*\d", text[at:]).end()
        if _NEXT_OP.match(text, at + number_end) and not re.match(r"\s*[×x·]\s*10", text[at + number_end :]):
            continue  # "= 2 × 3": the right side is an expression, not a result
        expression = m.group().strip()
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


def _python(expression: str, percent_is_fraction: bool) -> str:
    """Units dropped; `a%` read as (a/100), or as plain a when not percent_is_fraction."""

    def unit(m: re.Match[str]) -> str:
        if percent_is_fraction and m.group().rstrip().endswith("%"):
            return f"({m.group(1)}/100)"
        return m.group(1)

    return _NUMBER_IN_UNITS.sub(unit, expression)
