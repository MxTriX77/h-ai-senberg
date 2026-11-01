"""Chemistry checker: element balance. Reaction equations are checked by counting atoms and rebalanced when
exactly one positive whole-number balance exists, otherwise reported; "A → B loses/gains X" statements are
checked by subtracting formulas and X is replaced when the difference is a plain loss (or gain). Only plain
formulas count (H2O, Ca(OH)2, CuSO4·5H2O, subscript digits); names, ions, electrons and LaTeX are skipped.
A fix is doubtful (only reported, verifier.checkers_clear_only) when the equation may not be chemistry: a
formula with an explicit count of 1 (P₁V₁, S1), a written coefficient over 20 (1000 IU) or a label (S5.)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fractions import Fraction

from . import Finding

NAME = "element_balance"

ELEMENTS = frozenset(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr "
    "Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W "
    "Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt "
    "Ds Rg Cn Nh Fl Mc Lv Ts Og".split()
)
_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")  # same length: spans stay valid

_GROUP = r"(?:[A-Z][a-z]?\d*|\((?:[A-Z][a-z]?\d*)+\)\d*|\[(?:[A-Z][a-z]?\d*)+\]\d*)"
_FORMULA = rf"{_GROUP}+(?:[·•∙*]\d*{_GROUP}+)*"
_STATE = r"(?:\s?\((?:s|l|g|aq)\))?"
_VULGAR = {"½": "1/2", "⅓": "1/3", "⅔": "2/3", "¼": "1/4", "¾": "3/4"}  # ½ Cl₂ → Cl
_COEF = rf"(?:\d+(?:\.\d+)?|\d+/\d+|[{''.join(_VULGAR)}])"
_SPECIES_RE = re.compile(rf"(?:(?P<coef>{_COEF})(?P<gap>\s?))?(?P<formula>{_FORMULA})(?P<state>{_STATE})")
_SPECIES = rf"(?:{_COEF}\s?)?{_FORMULA}{_STATE}"
_SIDE = rf"{_SPECIES}(?:\s+\+\s+{_SPECIES})*"
_ARROWS = r"→|⟶|->|=>|⇌|⇄|↔|⇒"
_ARROW = rf"\s*(?:{_ARROWS}|=)\s*"
_EQUATION = re.compile(rf"(?<![\w(+\-−^=])({_SIDE})({_ARROW})({_SIDE})(?![\w(^⁺⁻=\-])")  # not "C=O → CH-OH"
_CHARGE = re.compile(r"[⁺⁻^]|[\w)][+−-](?=\s|$|[,.;)])|\be-")  # ions and electrons are not handled
_LOSS = (r"loses|lose|losing|loss of|eliminat(?:es|ing|ion of)|releas(?:es|ing|e of)|splits? off|gives? off"
         r"|теря\w*|потер\w*|отщепл\w*|отщеп\w*")
_GAIN = r"gains|gaining|gain of|takes? up|adds|addition of|присоедин\w*|приобрет\w*"
_CHANGE = re.compile(rf"(?i:({_LOSS})|({_GAIN}))\s+(?:(?i:an?|one|a molecule of|one molecule of|молекулу)\s+)?"
                     rf"({_FORMULA})(?![\w(])")
MAX_COEFFICIENT = 20  # a larger written number in front is an amount (1000 IU), not a coefficient
# a label (S5. / **S5.** / - S5: / 1. (S5)): one capital letter and digits starting a line or list item
_LABEL = re.compile(r"^[ \t#>]*(?:(?:[-*•+]|\d+[.)])[ \t]+)?[*_(\[]*([A-Z]\d+)[*_]*[.:)\]]", re.M)


def composition(formula: str) -> dict[str, int] | None:
    """Atom counts of a plain formula (parentheses, brackets, hydrate dots), None for anything else."""
    counts: dict[str, int] = {}
    for part in re.split(r"[·•∙*]", formula.translate(_SUBSCRIPTS)):
        m = re.match(r"(\d*)(.+)$", part)
        parsed = _parse(m.group(2)) if m else None
        if parsed is None:
            return None
        for element, n in parsed.items():
            counts[element] = counts.get(element, 0) + n * (int(m.group(1)) if m.group(1) else 1)
    return counts or None


def _parse(formula: str) -> dict[str, int] | None:
    stack: list[dict[str, int]] = [{}]
    for token in re.finditer(r"([A-Z][a-z]?)(\d*)|([(\[])|([)\]])(\d*)|(.)", formula):
        element, n, opening, closing, times, other = token.groups()
        if other is not None or (element and element not in ELEMENTS):
            return None
        if element:
            stack[-1][element] = stack[-1].get(element, 0) + (int(n) if n else 1)
        elif opening:
            stack.append({})
        else:
            if len(stack) == 1:
                return None
            group = stack.pop()
            for e, k in group.items():
                stack[-1][e] = stack[-1].get(e, 0) + k * (int(times) if times else 1)
    return stack[0] if len(stack) == 1 else None


def hill(counts: dict[str, int]) -> str:
    """C, H, then alphabetical (all alphabetical without carbon), as PubChem writes formulas."""
    order = sorted(e for e in counts if counts[e])
    if "C" in order:
        order = ["C"] + (["H"] if "H" in order else []) + [s for s in order if s not in ("C", "H")]
    return "".join(s + (str(counts[s]) if counts[s] > 1 else "") for s in order)


@dataclass
class Species:
    coef: Fraction
    formula: str  # as written in the draft
    counts: dict[str, int]
    state: str
    gap: str | None  # between a written coefficient and the formula


@dataclass
class Equation:
    start: int
    end: int
    left: list[Species]
    arrow: str
    right: list[Species]
    doubt: str | None = None  # why this may not be chemistry

    def balanced(self) -> bool:
        return _totals(self.left) == _totals(self.right)

    def balance(self) -> list[int] | None:
        """The smallest positive whole-number coefficients (left, then right) when exactly one balance exists."""
        species = [*self.left, *self.right]
        elements = sorted({e for s in species for e in s.counts})
        rows = [[Fraction(s.counts.get(e, 0) * (1 if i < len(self.left) else -1)) for i, s in enumerate(species)]
                for e in elements]
        vector = _null_vector(rows, len(species))
        if vector is None:
            return None
        if all(v < 0 for v in vector):
            vector = [-v for v in vector]
        if not all(v > 0 for v in vector):
            return None
        scale = math.lcm(*(v.denominator for v in vector))
        ints = [int(v * scale) for v in vector]
        g = math.gcd(*ints)
        return [i // g for i in ints]

    def written(self, coefs: list[int]) -> str:
        gap = next((s.gap for s in [*self.left, *self.right] if s.gap is not None), "")
        sides: list[list[str]] = [[], []]
        for i, (s, c) in enumerate(zip([*self.left, *self.right], coefs)):
            sides[i >= len(self.left)].append((f"{c}{gap}" if c != 1 else "") + s.formula + s.state)
        return " + ".join(sides[0]) + self.arrow + " + ".join(sides[1])

    def imbalance(self) -> str:
        left, right = _totals(self.left), _totals(self.right)
        return ", ".join(f"{e} {_num(left.get(e, Fraction(0)))}/{_num(right.get(e, Fraction(0)))}"
                         for e in sorted({*left, *right}) if left.get(e, 0) != right.get(e, 0))


def _num(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{float(x):g}"


def _totals(species: list[Species]) -> dict[str, Fraction]:
    out: dict[str, Fraction] = {}
    for s in species:
        for e, n in s.counts.items():
            out[e] = out.get(e, Fraction(0)) + s.coef * n
    return {e: v for e, v in out.items() if v}


def _null_vector(rows: list[list[Fraction]], n: int) -> list[Fraction] | None:
    """The one vector spanning the null space of the matrix (None when the null space is not one-dimensional)."""
    m = [r[:] for r in rows]
    pivots: list[int] = []
    r = 0
    for c in range(n):
        p = next((i for i in range(r, len(m)) if m[i][c] != 0), None)
        if p is None:
            continue
        m[r], m[p] = m[p], m[r]
        m[r] = [x / m[r][c] for x in m[r]]
        for i in range(len(m)):
            if i != r and m[i][c] != 0:
                m[i] = [a - m[i][c] * b for a, b in zip(m[i], m[r])]
        pivots.append(c)
        r += 1
    free = [c for c in range(n) if c not in pivots]
    if len(free) != 1:
        return None
    vector = [Fraction(0)] * n
    vector[free[0]] = Fraction(1)
    for i, c in enumerate(pivots):
        vector[c] = -m[i][free[0]]
    return vector


def find_equations(text: str, labels: frozenset[str] = frozenset()) -> list[Equation]:
    """Reaction equations written with plain formulas; lines with ions or electrons are skipped. `labels`: tokens
    used as labels (`find_labels`), which make an equation doubtful."""
    norm, out = text.translate(_SUBSCRIPTS), []
    for m in _EQUATION.finditer(norm):
        start, end = norm.rfind("\n", 0, m.start()) + 1, norm.find("\n", m.end())
        if _CHARGE.search(norm[start : end if end >= 0 else len(norm)]):
            continue
        if m.group(2) == "=":  # "(CH₃)₂C=CHCH₃", "O=C=O": a double bond in a structural formula
            continue
        if re.search(rf"(?:{_ARROWS})\s*$", norm[start : m.start()]) or re.match(rf"\s*(?:{_ARROWS})", norm[m.end() :]):
            continue  # one step of a chain "S1 → S2 → S4" (a path, a sequence): not one reaction
        if re.search(r"\+\s*$", norm[start : m.start()]):
            continue  # "(N₂ + 3H₂ ⇌ 2NH₃", "He*(2³S₁) + Ne → …": the left side starts before what was read
        sides = [_species(text, norm, m.start(1), m.end(1)), _species(text, norm, m.start(3), m.end(3))]
        if None in sides or (m.group(2).strip() == "=" and not _chemical(sides)):
            continue
        out.append(Equation(m.start(), m.end(), sides[0], text[m.start(2) : m.end(2)], sides[1],
                            _doubt([*sides[0], *sides[1]], labels)))
    return out


def find_labels(*texts: str) -> frozenset[str]:
    """Tokens of one capital letter and digits that start a line or list item (S5. / **S5.** / - S5: / (S5)),
    when the letter starts two or more of them (S4, S5): a lone "O2: 32 g/mol" line stays a formula."""
    tokens = {m.group(1) for t in texts for m in _LABEL.finditer(t.translate(_SUBSCRIPTS))}
    return frozenset(t for t in tokens if sum(u[0] == t[0] for u in tokens) > 1)


def _doubt(species: list[Species], labels: frozenset[str]) -> str | None:
    """Why an equation may not be chemistry: a label, an indexed symbol (explicit count of 1) or an amount."""
    for s in species:
        formula = s.formula.translate(_SUBSCRIPTS)
        if formula in labels:
            return f"{s.formula} is a label"
        if re.search(r"(?<!\d)1(?!\d)", formula):
            return f"{s.formula} has an explicit count of 1 (an index or a label)"
        if s.coef > MAX_COEFFICIENT:
            return f"{_num(s.coef)} {s.formula} is an amount, not a coefficient"
    return None


def _species(text: str, norm: str, start: int, end: int) -> list[Species] | None:
    out, pos = [], start
    for part in re.split(r"(\s+\+\s+)", norm[start:end]):
        if re.fullmatch(r"\s+\+\s+", part):
            pos += len(part)
            continue
        m = _SPECIES_RE.fullmatch(part)
        counts = composition(m.group("formula")) if m else None
        if counts is None:
            return None
        formula = text[pos + m.start("formula") : pos + m.end("formula")]  # as written (subscripts kept)
        coef = Fraction(_VULGAR.get(m.group("coef"), m.group("coef"))) if m.group("coef") else Fraction(1)
        out.append(Species(coef, formula, counts, m.group("state"), m.group("gap") if m.group("coef") else None))
        pos += len(part)
    return out


def _chemical(sides: list[list[Species]]) -> bool:
    """With `=` as the arrow, both sides must look like chemistry: some formula with two elements or a digit."""
    return all(any(len(s.counts) > 1 or re.search(r"\d", s.formula.translate(_SUBSCRIPTS)) for s in side)
               for side in sides)


def check(draft: str, task: str = "") -> list[Finding]:
    """Unbalanced equations (rebalanced when one balance exists) and wrong "loses/gains X" statements; the fixes
    of an equation that may not be chemistry carry a doubt (labels are read from the draft and the task)."""
    findings: list[Finding] = []
    transformations: set[int] = set()  # "A → B" lines that a loses/gains statement explains: not equations
    norm, labelled = draft.translate(_SUBSCRIPTS), find_labels(task, draft)
    for m in _CHANGE.finditer(norm):
        line_start, line_end = norm.rfind("\n", 0, m.start()) + 1, norm.find("\n", m.end())
        pairs = [e for e in find_equations(draft[line_start : line_end if line_end >= 0 else len(draft)], labelled)
                 if len(e.left) == 1 and len(e.right) == 1]
        stated = composition(m.group(3))
        if not pairs or stated is None:
            continue
        transformations.add(line_start + pairs[0].start)
        a, b = pairs[0].left[0].counts, pairs[0].right[0].counts
        first, second = (a, b) if m.group(1) else (b, a)
        actual = {e: first.get(e, 0) - second.get(e, 0) for e in {*a, *b}}
        actual = {e: n for e, n in actual.items() if n}
        if actual == stated:
            continue
        verb = "loses" if m.group(1) else "gains"
        text = draft[m.start(3) : m.end(3)]
        if actual and all(n > 0 for n in actual.values()):
            findings.append(Finding(m.start(3), m.end(3), text, hill(actual), f"{verb} {hill(actual)}, not {text}",
                                    pairs[0].doubt))
        else:
            findings.append(Finding(m.start(3), m.end(3), text, None, f"{verb} {text} does not match the formulas"))
    for eq in find_equations(draft, labelled):
        if eq.balanced() or eq.start in transformations:
            continue
        text, coefs = draft[eq.start : eq.end], eq.balance()
        if coefs:
            findings.append(Finding(eq.start, eq.end, text, eq.written(coefs), "rebalanced", eq.doubt))
        else:
            findings.append(Finding(eq.start, eq.end, text, None, f"not balanced ({eq.imbalance()})"))
    return sorted(findings, key=lambda f: f.start)
