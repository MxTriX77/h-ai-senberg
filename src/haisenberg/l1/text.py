"""Text helpers shared by the Layer 1 build and queries, so both sides normalize identically."""

from __future__ import annotations

import re
import unicodedata


def normalize(s: str) -> str:
    """Lookup key for names and titles: NFKC, underscores as spaces, case-folded, whitespace collapsed."""
    return " ".join(unicodedata.normalize("NFKC", s).replace("_", " ").casefold().split())


_FORMULA = re.compile(r"^((?:[A-Z][a-z]?\d*)+)([+-]\d*)?$")
_ELEMENT = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(formula: str) -> tuple[dict[str, int], str] | None:
    """`C9H18NO4+` -> ({'C': 9, 'H': 18, 'N': 1, 'O': 4}, '+'); None if it is not a plain formula."""
    m = _FORMULA.match(formula.strip())
    if not m:
        return None
    counts: dict[str, int] = {}
    for symbol, n in _ELEMENT.findall(m.group(1)):
        counts[symbol] = counts.get(symbol, 0) + (int(n) if n else 1)
    return counts, m.group(2) or ""


def to_hill(formula: str) -> str | None:
    """Hill order as PubChem writes it: C, H, then alphabetical; all alphabetical without carbon."""
    parsed = parse_formula(formula)
    if parsed is None:
        return None
    counts, charge = parsed
    order = sorted(counts)
    if "C" in counts:
        order = ["C"] + (["H"] if "H" in counts else []) + [s for s in order if s not in ("C", "H")]
    return "".join(s + (str(counts[s]) if counts[s] > 1 else "") for s in order) + charge
