"""CODATA constants from NIST's `allascii.txt` into the `constant` table.

The listing is fixed width: name [0:60], value [60:85], uncertainty [85:110], unit [110:].
Values print digit groups with spaces; `...` marks truncated exact values; `(exact)` marks exact ones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ingest.l1base import BuildContext, BuildError


@dataclass(frozen=True)
class Constant:
    name: str
    value: float
    uncertainty: float
    unit: str
    value_text: str
    exact: bool
    truncated: bool


def parse(path: Path) -> tuple[str, list[Constant]]:
    """Return (adjustment year, constants). Raises BuildError on any unparsable row."""
    lines = path.read_text(encoding="utf-8").splitlines()
    year = re.search(r"(\d{4}) CODATA adjustment", "\n".join(lines[:10]))
    if not year:
        raise BuildError(f"{path}: no 'CODATA adjustment' year in the header")
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith("-----")) + 1
    except StopIteration:
        raise BuildError(f"{path}: no '-----' line before the data rows") from None

    rows = []
    for n, line in enumerate(lines[start:], start + 1):
        if not line.strip():
            continue
        name, val, unc, unit = line[:60].strip(), line[60:85].strip(), line[85:110].strip(), line[110:].strip()
        exact = unc == "(exact)"
        try:
            value = float(val.replace(" ", "").replace("...", ""))
            uncertainty = 0.0 if exact else float(unc.replace(" ", ""))
        except ValueError:
            raise BuildError(f"{path}:{n}: cannot parse constant row: {line!r}") from None
        if not name:
            raise BuildError(f"{path}:{n}: constant row without a name: {line!r}")
        rows.append(Constant(name, value, uncertainty, unit, val, exact, val.endswith("...")))
    if not rows:
        raise BuildError(f"{path}: no constants found")
    return year.group(1), rows


def load(ctx: BuildContext) -> None:
    for entry, path in ctx.files("codata"):
        year, rows = parse(path)
        source_id = ctx.add_source(entry, year)
        ctx.con.executemany(
            "INSERT INTO constant(name, value, uncertainty, unit, value_text, exact, truncated, source_id)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(c.name, c.value, c.uncertainty, c.unit, c.value_text, c.exact, c.truncated, source_id) for c in rows],
        )
        ctx.report["constants"] = ctx.report.get("constants", 0) + len(rows)
