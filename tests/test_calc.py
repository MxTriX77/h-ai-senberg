"""The calculator's evaluator and the arithmetic finder (agent-tools Calculator tool; claim-verifier Arithmetic check)."""

import json
import math

import pytest

from haisenberg.calc import CalcError, evaluate, find_arithmetic, run_tool


@pytest.mark.parametrize("expression, value", [
    ("2 + 3 * 4", 14),
    ("2.10 / (2.00 / 138.12 * 180.16) × 100", 80.50005),
    ("6.022 × 10^23 × 0.0555", 3.342e22),
    ("6.022e23 / 18.015", 3.3428e22),
    ("1.6 × 10⁻¹⁹ × 2", 3.2e-19),
    ("sqrt(2 * 9.81 * 5)", 9.9045),
    ("√(2 × 9.81 × 10)", 14.007),
    ("-log(1.34e-3)", 2.873),
    ("ln(2) / 8.02", 0.086427),
    ("0.5 ^ (30 / 8.02)", 0.074809),
    ("max(1, 2.5) - min(3, 4)", -0.5),
    ("2 x 3", 6),
    ("16,775 / 2", 8387.5),  # a comma before exactly three digits groups thousands
    ("0,0500 × 56,08", 2.804),
    ("\\frac{2.10}{2.609} \\times 100", 80.49),
    ("2 * pi", 2 * math.pi),
    ("10^{-3} \\cdot 5", 0.005),
])
def test_evaluate(expression, value):
    assert evaluate(expression) == pytest.approx(value, rel=1e-3)


def test_decimal_comma_switch():
    assert evaluate("16,775 / 2", decimal_comma=True) == pytest.approx(8.3875)


@pytest.mark.parametrize("expression", [
    "__import__('os').system('x')", "open('f')", "x + 1", "2 ** 5000", "1 / 0", "sqrt(-1)", "(-8) ** (1/3)",
    "1e308 * 10", "a.b", "[1, 2]", "1 if 2 else 3", "2" * 600,
])
def test_not_arithmetic(expression):
    with pytest.raises(CalcError):
        evaluate(expression)


def test_tool():
    text, lines = run_tool(json.dumps({"expression": "2.10 / (2.00 / 138.12 * 180.16) * 100"}))
    assert text.startswith("2.10 / (2.00 / 138.12 * 180.16) * 100 = 80.498") and lines == [text]
    assert run_tool(json.dumps({"expression": "__import__('os')"}))[0].startswith("Error: not allowed")
    assert run_tool("not json")[0] == "Error: arguments are not valid JSON"
    assert run_tool(json.dumps({"expr": "1"}))[0].startswith("Error: 'expression'")


def found(text, decimal_comma=False):
    return [(a.expression, round(a.value, 6), text[a.result_at:a.result_at + 6]) for a in find_arithmetic(text, decimal_comma)]


def test_finds_and_computes():
    text = "The mass of CaO is 0.0500 × 56.08 = 3.80 g, and 5.00 − 3.80 = 2.20 g is lost."
    assert found(text) == [("0.0500 * 56.08", 2.804, "3.80 g"), ("5.00 - 3.80", 1.2, "2.20 g")]


@pytest.mark.parametrize("text, value", [
    ("n = 2.00 g / 138.12 g/mol = 0.01448 mol", 0.01448),
    ("$$\\frac{2.10}{2.609} \\times 100\\% = 80.5\\%$$", 80.49),
    ("yield = 2.10 g / 2.609 g × 100 % = 80.5 %", 80.49),
    ("15 % × 200 = 30", 30),
    ("Kc = (0.20)^2 / (0.50 × 1.50^3) = 0.0237", 0.0237),
    ("v = √(2 × 9.81 × 10) = 14.0 m/s", 14.007),
    ("pH = -log(1.34 × 10^-3) = 2.87", 2.873),
    ("N = 0.0555 mol × 6.022 × 10²³ mol⁻¹ = 3.34 × 10²² molecules", 3.342e22),
    ("c = 0.250 mol / 0.250 L = 1.00 mol/L", 1.0),
    ("m = 0,0500 моль × 56,08 г/моль = 2,80 г", 2.804),
])
def test_units_latex_and_percent(text, value):
    (a,) = list(find_arithmetic(text, decimal_comma="моль" in text))
    assert a.value == pytest.approx(value, rel=2e-3)


@pytest.mark.parametrize("text", [
    "1 atm = 101.325 kPa",
    "E = mc^2 = 9.0 × 10^16 J",
    "Step 2 = done",
    "x = 2 × n = 6",
    "a ≤ 3 = b",
    "2 + 3 == 5",
    "2 × 3 ≈ 6.1",
    "1 L = 1000 mL",
    "0.5 × 4 = 2 × 1",  # the right side is an expression, not a result
])
def test_not_found(text):
    assert found(text) == []


def test_chain_checks_each_step():
    text = "n = 2.00/138.12 = 0.01448 mol × 180.16 g/mol = 2.609 g"
    assert [round(a.value, 4) for a in find_arithmetic(text)] == [0.0145, 2.6087]
