"""Units as records and claims write them, so a claim's unit can be compared with a record's in code. Two
spellings are the same unit when they come to the same SI base units and scale: `J Hz^-1` = `J s` = `J·s` =
`kg m^2 s^-1`, `m s^-1` = `m/s` = `metres per second`; `km/h` and `eV s` are other units."""

from __future__ import annotations

import math
import re

Unit = tuple[float, dict[str, int]]  # scale, powers of SI base units (°C, °F and sr kept as their own)

_UNITS = {  # symbol: (scale, SI base units)
    "m": (1, "m"), "g": (1e-3, "kg"), "s": (1, "s"), "A": (1, "A"), "K": (1, "K"), "mol": (1, "mol"),
    "sr": (1, "sr"), "Hz": (1, "s-1"), "N": (1, "kg m s-2"), "Pa": (1, "kg m-1 s-2"), "J": (1, "kg m2 s-2"),
    "W": (1, "kg m2 s-3"), "C": (1, "A s"), "V": (1, "kg m2 s-3 A-1"), "F": (1, "kg-1 m-2 s4 A2"),
    "Ω": (1, "kg m2 s-3 A-2"), "S": (1, "kg-1 m-2 s3 A2"), "Wb": (1, "kg m2 s-2 A-1"), "T": (1, "kg s-2 A-1"),
    "H": (1, "kg m2 s-2 A-2"), "L": (1e-3, "m3"), "Å": (1e-10, "m"), "min": (60, "s"), "h": (3600, "s"),
    "eV": (1.602176634e-19, "kg m2 s-2"), "Da": (1.66053906892e-27, "kg"), "u": (1.66053906892e-27, "kg"),
    "atm": (101325, "kg m-1 s-2"), "bar": (1e5, "kg m-1 s-2"), "cal": (4.184, "kg m2 s-2"),
    "°C": (1, "°C"), "°F": (1, "°F"), "%": (0.01, ""),
}
_NAMES = {
    "metre": "m", "meter": "m", "gram": "g", "second": "s", "ampere": "A", "kelvin": "K", "mole": "mol",
    "steradian": "sr", "hertz": "Hz", "newton": "N", "pascal": "Pa", "joule": "J", "watt": "W", "coulomb": "C",
    "volt": "V", "farad": "F", "ohm": "Ω", "siemens": "S", "weber": "Wb", "tesla": "T", "henry": "H",
    "litre": "L", "liter": "L", "angstrom": "Å", "minute": "min", "hour": "h", "electronvolt": "eV",
    "dalton": "Da", "amu": "u", "atmosphere": "atm", "calorie": "cal", "percent": "%",
}
_PREFIXES = {"T": 1e12, "G": 1e9, "M": 1e6, "k": 1e3, "d": 1e-1, "c": 1e-2, "m": 1e-3, "µ": 1e-6, "μ": 1e-6,
             "n": 1e-9, "p": 1e-12, "f": 1e-15}
_PREFIX_NAMES = {"tera": "T", "giga": "G", "mega": "M", "kilo": "k", "deci": "d", "centi": "c", "milli": "m",
                 "micro": "µ", "nano": "n", "pico": "p", "femto": "f"}

# One factor: a symbol or name and its power (m^2, m**2, s^-1, s-1, m2: superscripts already plain digits).
_FACTOR = re.compile(r"(°\s?[CF]|℃|%|[^\W\d_]+)(?:\s*(?:\^|\*\*)\s*\(?([-+]?\d+)\)?|(-?\d)(?!\d))?(?!\w)")
_TIMES = re.compile(r"[ \t]*[·⋅∙*][ \t]*|[ \t]+")
_OVER = re.compile(r"[ \t]*/[ \t]*|[ \t]+per[ \t]+")


def _base(spec: str) -> dict[str, int]:
    return {m[1]: int(m[2] or 1) for m in re.finditer(r"([^\s\d-]+)(-?\d*)", spec)}


def _one(token: str) -> Unit | None:
    """A unit symbol, maybe with an SI prefix (km, µs), or name (kilometres)."""
    token = re.sub(r"\s", "", token).replace("℃", "°C")
    word = token.lower()
    for w in (word, word.removesuffix("s")):
        for name, sym in [("", ""), *_PREFIX_NAMES.items()]:
            if w.startswith(name) and w[len(name) :] in _NAMES:
                scale, base = _UNITS[_NAMES[w[len(name) :]]]
                return _PREFIXES.get(sym, 1) * scale, _base(base)
    if token in _UNITS:
        return _UNITS[token][0], _base(_UNITS[token][1])
    if token[:1] in _PREFIXES and token[1:] in _UNITS:
        scale, base = _UNITS[token[1:]]
        return _PREFIXES[token[0]] * scale, _base(base)
    return None


def _read(text: str) -> tuple[Unit | None, int]:
    """(the unit written at the start of text, or None when there is none this module knows or it breaks off
    after `/` or `per`; where it ends)."""
    scale, powers, pos, sign, group, end = 1.0, {}, 0, 1, False, 0
    if m := re.match(r"[ \t]*(?:per[ \t]+)?", text):
        sign, pos = (-1 if "per" in m[0] else 1), m.end()
    while True:
        if sign < 0 and text.startswith("(", pos):  # J/(mol·K)
            group, pos = True, pos + 1
        factor = _FACTOR.match(text, pos)
        one = _one(factor[1]) if factor else None
        if one is None:
            return (None, 0) if sign < 0 or group or not end else ((scale, powers), end)
        power = sign * int(factor[2] or factor[3] or 1)
        scale *= one[0] ** power
        for b, p in one[1].items():
            powers[b] = powers.get(b, 0) + p * power
        pos = end = factor.end()
        if group and text.startswith(")", pos):
            group, pos, end, sign = False, pos + 1, pos + 1, 1
        if m := _OVER.match(text, pos):
            sign, pos = -1, m.end()
        elif m := _TIMES.match(text, pos):
            sign, pos = (-1 if group else 1), m.end()
        else:
            return (scale, powers), end


def read_unit(text: str) -> tuple[Unit | None, bool]:
    """(the unit written at the start of text, as a scale and SI base powers, or None; whether a word or a
    unit sign is written there at all, so None with True is a word this module does not know as a unit)."""
    text = re.sub(r"\\[,;: ]|~", " ", text.replace("\\cdot", "·"))  # LaTeX: $6.6 \times 10^{-34}\,\mathrm{J\,s}$
    text = re.sub(r"\\(?:text|mathrm|rm)\s*|[{}$]", "", text)
    unit, _ = _read(text)
    return unit, bool(re.match(r"[ \t]*(?:per[ \t]+)?(?:[^\W\d_]|[°%℃])", text))


def parse_unit(spec: str) -> Unit | None:
    """The unit a whole string writes ('' = a pure number), or None when it cannot be read as a whole."""
    if not spec.strip():
        return 1.0, {}
    unit, end = _read(spec.strip())
    return unit if end == len(spec.strip()) else None


def same_unit(a: Unit, b: Unit) -> bool:
    """Equal units: the same SI base powers and scale."""
    return ({k: p for k, p in a[1].items() if p} == {k: p for k, p in b[1].items() if p}
            and math.isclose(a[0], b[0], rel_tol=1e-9))
