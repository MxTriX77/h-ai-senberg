"""Chemistry checker plug-in: element balance (claim-verifier, Domain checkers)."""

import pytest

from haisenberg.checkers import CHECKERS
from haisenberg.checkers.element_balance import check, composition, find_equations, hill


def fixed(draft):
    out = draft
    for f in sorted(check(draft), key=lambda f: f.start, reverse=True):
        if f.fixed is not None:
            out = out[: f.start] + f.fixed + out[f.end :]
    return out


@pytest.mark.parametrize("formula, counts", [
    ("H2O", {"H": 2, "O": 1}),
    ("Ca(OH)2", {"Ca": 1, "O": 2, "H": 2}),
    ("CuSO4·5H2O", {"Cu": 1, "S": 1, "O": 9, "H": 10}),
    ("K4[Fe(CN)6]", {"K": 4, "Fe": 1, "C": 6, "N": 6}),
    ("C₁₀H₁₀O₄", {"C": 10, "H": 10, "O": 4}),
    ("Xy2", None),
    ("Ca(OH", None),
])
def test_composition(formula, counts):
    assert composition(formula) == counts


def test_hill():
    assert hill({"O": 1, "H": 2, "C": 2}) == "C2H2O" and hill({"Cl": 1, "Na": 1}) == "ClNa"


@pytest.mark.parametrize("draft, expected", [
    ("H2 + O2 → H2O", "2H2 + O2 → 2H2O"),
    ("C3H8 + O2 → CO2 + H2O", "C3H8 + 5O2 → 3CO2 + 4H2O"),
    ("Ca(OH)2 + HCl -> CaCl2 + H2O (neutralisation)", "Ca(OH)2 + 2HCl -> CaCl2 + 2H2O (neutralisation)"),
    ("Сгорание: C2H5OH + O2 = CO2 + H2O", "Сгорание: C2H5OH + 3O2 = 2CO2 + 3H2O"),
    ("H₂ + O₂ → H₂O", "2H₂ + O₂ → 2H₂O"),
    ("2 Fe + O2 → Fe2O3", "4 Fe + 3 O2 → 2 Fe2O3"),
])
def test_unbalanced_equations_rebalanced(draft, expected):
    assert fixed(draft) == expected


@pytest.mark.parametrize("draft", [
    "2H2 + O2 → 2H2O",
    "CH4 + 2O2 → CO2 + 2H2O",
    "CuSO4·5H2O → CuSO4 + 5H2O",
    "C2H5OH + 3O2 = 2CO2 + 3H2O",
    "4 H2 + 2 O2 → 4 H2O",  # not the smallest coefficients, but balanced
    "Fe3+ + 3OH- → Fe(OH)3",  # ions: not handled
    "ferulic acid → vanillin + acetic acid",  # names: not handled
    "2 + 3 = 5", "C = 12, H = 1, O = 16", "H2O = 18 g/mol", "pH = 7",
])
def test_left_alone(draft):
    assert check(draft) == []


def test_unbalanceable_equation_reported_not_changed():
    (f,) = check("NaCl → Na2O")
    assert f.fixed is None and f.note.startswith("not balanced (")


def test_wrong_loss_fixed():
    draft = "Ferulic acid (C10H10O4) → vanillin (C8H8O3)? The molecule C10H10O4 → C8H8O3 loses CO2."
    (f,) = check(draft)
    assert (f.text, f.fixed) == ("CO2", "C2H2O") and fixed(draft).endswith("loses C2H2O.")


@pytest.mark.parametrize("draft", [
    "C2H6 → C2H4: the molecule loses H2.",
    "C2H4 → C2H6 by addition of H2",
    "C2H5OH → C2H4 (отщепляет H2O)",
])
def test_right_loss_or_gain_left_alone(draft):
    assert check(draft) == []


def test_mixed_change_reported():
    (f,) = check("CH4 → CO2 loses H4")
    assert f.fixed is None and "does not match" in f.note


def test_equations_found_with_spans():
    (eq,) = find_equations("Equation: 2H2 + O2 → 2H2O.")
    assert "Equation: 2H2 + O2 → 2H2O."[eq.start:eq.end] == "2H2 + O2 → 2H2O"


def test_registered():
    assert CHECKERS["element_balance"] is check
