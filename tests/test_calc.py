"""The calculator's evaluator and the arithmetic finder (agent-tools Calculator tool; claim-verifier Arithmetic check)."""

import json
import math

import pytest

from haisenberg.calc import CalcError, computations, converted, evaluate, find_arithmetic, prefixed, run_tool


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
    return [(a.expression, round(a.plain, 6), text[a.result_at:a.result_at + 6]) for a in find_arithmetic(text, decimal_comma)]


def test_finds_and_computes():
    text = "The mass of CaO is 0.0500 × 56.08 = 3.80 g, and 5.00 − 3.80 = 2.20 g is lost."
    assert found(text) == [("0.0500 * 56.08", 2.804, "3.80 g"), ("5.00 - 3.80", 1.2, "2.20 g")]


@pytest.mark.parametrize("text, fraction, plain", [
    ("n = 2.00 g / 138.12 g/mol = 0.01448 mol", 0.01448, 0.01448),
    ("$$\\frac{2.10}{2.609} \\times 100\\% = 80.5\\%$$", 0.8049, 80.49),
    ("yield = 2.10 g / 2.609 g × 100 % = 80.5 %", 0.8049, 80.49),
    ("15 % × 200 = 30", 30, 3000),
    ("$$\\frac{80.0\\%}{87.4\\%} = 91.5\\%$$", 0.9153, 0.9153),
    ("Kc = (0.20)^2 / (0.50 × 1.50^3) = 0.0237", 0.0237, 0.0237),
    ("v = √(2 × 9.81 × 10) = 14.0 m/s", 14.007, 14.007),
    ("pH = -log(1.34 × 10^-3) = 2.87", 2.873, 2.873),
    ("N = 0.0555 mol × 6.022 × 10²³ mol⁻¹ = 3.34 × 10²² molecules", 3.342e22, 3.342e22),
    ("c = 0.250 mol / 0.250 L = 1.00 mol/L", 1.0, 1.0),
    ("m = 0,0500 моль × 56,08 г/моль = 2,80 г", 2.804, 2.804),
])
def test_units_latex_and_percent(text, fraction, plain):
    (a,) = list(find_arithmetic(text, decimal_comma="моль" in text))
    assert (a.fraction, a.plain) == (pytest.approx(fraction, rel=2e-3), pytest.approx(plain, rel=2e-3))


def test_percent_flags():
    (a,) = find_arithmetic("x = 0.0986/0.1236 × 100% = 79.8%")
    assert a.as_percent and a.times_100 and not a.has_percent
    (b,) = find_arithmetic("80% / 90% = 88.9%")
    assert b.has_percent and not b.as_percent and not b.times_100


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
    "h = 6.626 070 15 × 2 = 13.25",  # digits before the expression that it cannot join: not one reading
    "y = 2 (3 + 4) = 14",
])
def test_not_found(text):
    assert found(text) == []


@pytest.mark.parametrize("text, expression, value", [
    ("Q = 12 500 mA·h × 3.7 V = 46 250 mWh", "12500 mA*h * 3.7 V", 46250),
    (r"m = 1\,250 \times 4 = 5\,000", "1250 * 4", 5000),
    ("n = 2 400 × 3 = 7 200", "2400 * 3", 7200),
    ("s = 1,234,567 × 2 = 2,469,134", "1234567 * 2", 2469134),
    ("x = 1,234.5 × 2 = 2,469", "1234.5 * 2", 2469),
    ("x = 1 234,5 × 2 = 2 469 г", "1234.5 * 2", 2469),
])
def test_grouped_numbers_are_one_number(text, expression, value):
    (a,) = find_arithmetic(text, decimal_comma="г" in text)
    assert (a.expression, a.plain) == (expression, pytest.approx(value))


@pytest.mark.parametrize("text, has", [
    ("2.45 kW", True), ("12500 mA*h", True), ("18500 мА*ч", True), ("250 mg", True), ("2000 ms", True),
    ("3 кВт", True), ("5 µs", True), ("2452 W", False), ("5 m", False), ("0.5 mol", False), ("3 min", False),
    ("101 Pa", False), ("sqrt(2) * pi", False), ("so the motor", False),
])
def test_prefixed_units(text, has):
    assert prefixed(text) is has


@pytest.mark.parametrize("text, value", [
    ("- (800 - 2) / 800 × 100 = 99.75%", 99.75),
    ("  * (6 - 1) × 2 = 10", 10),
    ("+ (6 - 1) × 2 = 10", 10),
    ("> 1. (9 - 3) / 2 = 3", 3),
    ("2) (9 - 3) / 2 = 3", 3),
    ("Notes:\n- sqrt(16) - 1 = 3", 3),
    ("-5 × 2 = -10", -10),  # a sign that touches its number is a sign
    ("- -5 × 2 = -10", -10),
    ("-(6 - 1) × 2 = -10", -10),
])
def test_list_markers_are_not_signs(text, value):
    (a,) = find_arithmetic(text)
    assert a.plain == pytest.approx(value)


@pytest.mark.parametrize("text, values", [
    ("the gap is 96 °C − (−18 °C), about 120 °C", {114}),
    ("margin (800 - 2) / 800 × 100 %", {0.9975, 99.75}),  # both readings of the % sign
    ("x = 2 × 3 = 6, and 4 + 1 more", {6, 5}),
    ("ethanol ~78 °C, dimethyl ether −24 °C", set()),  # no operator between them: not a computation
    ("5 / 0 and 2 × n", set()),
    ("0,5 × 3", {1.5}),
])
def test_computations(text, values):
    assert {round(v, 6) for v in computations(text, decimal_comma="0,5" in text)} == values


def test_chain_checks_its_last_step():  # "0.01448 mol × …" is a step of the chain, like "= 0.0600 L × 2.00 mol/L"
    text = "n = 2.00/138.12 = 0.01448 mol × 180.16 g/mol = 2.609 g"
    assert [round(a.plain, 4) for a in find_arithmetic(text)] == [2.6087]


# c4: only a whole left side is read; a × 10ⁿ is one number.
@pytest.mark.parametrize("text, expression, value", [
    ("ratio = 6.4 × 10⁷ ÷ 2.0 × 10⁶ = 32", "6.4e7 / 2.0e6", 32),
    ("4.5 × 10⁹ / 125 × 10⁶ = 36", "4.5e9 / 125e6", 36),
    ("x = 6.4·10^7 / 2.0·10^6 = 32", "6.4e7 / 2.0e6", 32),
    (r"$3.0 \times 10^{4} / 1.5 \times 10^{2} = 200$", "3.0e4 / 1.5e2", 200),
    ("pH = -log(1.34 × 10^-3) = 2.87", "-log(1.34e-3)", 2.873),  # a function of one number is arithmetic
    ("1.6e-19 × 2 = 3.2e-19", "1.6e-19 * 2", 3.2e-19),  # e-notation is one number, "e-19" is no unit
])
def test_powers_of_ten_are_one_number(text, expression, value):
    (a,) = find_arithmetic(text)
    assert (a.expression, a.plain) == (expression, pytest.approx(value, rel=1e-3))


def test_powers_of_ten_left_to_right_when_not_strict():
    (a,) = find_arithmetic("ratio = 6.4 × 10⁷ ÷ 2.0 × 10⁶ = 32", strict=False)
    assert a.plain == pytest.approx(3.2e13)


@pytest.mark.parametrize("text", [
    "250 × 10⁶ bytes/s = 250 MB/s",  # one number: a unit conversion
    "10⁹ bits/s = 1 Gbit/s",
    r"$(5.00 \times 10^{-3}) = 5.00 \times 10^{-5}$",
    "μ0 = 2π × 10⁻³ = 6.283 × 10⁻³",
    "KE = ½ · 800 kg · (20.0 m/s)² = 160,000 J",  # a factor before the expression
    "growth = (v₂ − v₁) / 48.00 × 100 = 8.5 %",  # cut before "× 100"
    "half of 12 × 4 = 24",
    "половина от 12 × 4 = 24",
    "Area × 2 × 3 = 6",  # a variable and an operator
    "x · 2 × 3 = 6",
    "€ (2 × 3) = 6",
    "**Step 1** 2.5 × 4 = 12",  # Markdown bold is not a power
    "rate = 2.6B / (30×24×60×60) = 1003",  # scale suffixes
    "20k × 3 = 60000",
    "2 million × 3 = 6 000 000",
    "1,5 млн × 3 = 4 500 000",
])
def test_cut_off_left_sides_skipped(text):
    assert found(text, decimal_comma="млн" in text) == []


@pytest.mark.parametrize("text, expression, value", [
    ("margin = ($2,500,000 − $1,750,000) / $2,500,000 × 100 = 30.0 %", "(2500000 - 1750000) / 2500000 * 100", 30),
    ("$1,200 × 3 = $3,600", "1200 * 3", 3600),
    ("€5 × 4 = 20 €", "5 * 4", 20),
    ("**2.5 × 4 = 12 kg**", "2.5 * 4", 10),  # emphasis around the equation
    ("*2.5 × 4 = 12 kg*", "2.5 * 4", 10),
    ("**Cost:** 2 × 3 = 6", "2 * 3", 6),
    ("The mass of CaO is 0.0500 × 56.08 = 3.80 g", "0.0500 * 56.08", 2.804),
])
def test_whole_left_sides_read(text, expression, value):
    (a,) = find_arithmetic(text)
    assert (a.expression, a.plain) == (expression, pytest.approx(value))


def test_cut_off_left_side_read_when_not_strict():
    (a,) = find_arithmetic("KE = ½ · 800 kg · (20.0 m/s)² = 160,000 J", strict=False)
    assert a.plain == pytest.approx(320000)


@pytest.mark.parametrize("expression, units, is_conversion", [
    ("10 Gbit/s * 60 s", "GB", True),  # bits → bytes
    ("125e6 bytes/s * 2", "MB/s", True),
    ("2 MB * 3", "bytes", True),
    ("2 m * 3", "cm", True),
    ("1000 * 9.81 * 0.04 * 6 W", "kW", False),  # an SI prefix on a known unit: the prefix readings decide
    ("2 h * 60", "min", False),
    ("0.0555 mol * 6.022e23 mol⁻¹", "molecules", False),
    ("4.5e9 / 8", "MB", False),  # no unit on the left to convert from
    ("2 GB * 3", "GB", False),
    ("2 m * 3", "more", False),
])
def test_unit_conversions(expression, units, is_conversion):
    assert converted(expression, units) is is_conversion
