"""PubChem compound subset into the `compound` and `compound_name` tables.

Subset: CIDs in CID-MeSH, plus CIDs linked from a stored Wikipedia infobox (ctx.linked_cids).
Each compound gets its title, formula, monoisotopic mass, a molar mass computed from the formula with
standard atomic weights (ingest/reference/atomic_weights.tsv), and its title plus first synonyms as names.
"""

from __future__ import annotations

import bisect
import gzip
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

from haisenberg.l1.text import normalize, parse_formula
from ingest.l1base import BuildContext, BuildError

ATOMIC_WEIGHTS = Path(__file__).parent / "reference" / "atomic_weights.tsv"
NAMES_INDEXED = 20  # synonyms per compound in compound_name (after the title)
NAMES_SHOWN = 10  # synonyms per compound kept for display
ISOTOPE_TOLERANCE = 0.001  # max difference between computed and PubChem monoisotopic mass
CHUNK = 1 << 24


def load_atomic_weights(path: Path = ATOMIC_WEIGHTS) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """(standard atomic weight, most-abundant-isotope mass) per element symbol; None when blank."""
    weights, isotopes = {}, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or line.startswith("symbol\t") or not line.strip():
            continue
        symbol, _z, weight, iso = (line.split("\t") + ["", ""])[:4]
        weights[symbol] = float(weight) if weight else None
        isotopes[symbol] = float(iso) if iso else None
    return weights, isotopes


def molar_mass(formula: str, pubchem_mono: float, weights, isotopes) -> tuple[float | None, str | None]:
    """(g/mol, None) or (None, reason). The charge suffix is ignored (electron mass neglected)."""
    parsed = parse_formula(formula)
    if parsed is None:
        return None, f"formula not understood: {formula}"
    counts, _charge = parsed
    missing = [s for s in counts if weights.get(s) is None]
    if missing:
        return None, f"no standard atomic weight for {', '.join(missing)}"
    mono = sum(isotopes[s] * n for s, n in counts.items())
    if abs(mono - pubchem_mono) > ISOTOPE_TOLERANCE:
        return None, "isotope-labelled: PubChem monoisotopic mass differs from the plain formula's"
    return round(sum(weights[s] * n for s, n in counts.items()), 4), None


def stream_selected(path: Path, cids: list[int], cap: int) -> dict[int, list[str]]:
    """Up to `cap` rows (text after the CID) for each CID in the sorted list `cids`.

    The Extras files are sorted by CID, so each decompressed chunk only needs a `find` per
    selected CID in its range instead of a Python loop over ~120M lines.
    """
    out: dict[int, list[str]] = {}
    tail, prev_last = b"", -1
    with gzip.open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK)
            data = tail + chunk
            if not data:
                break
            cut = data.rfind(b"\n") + 1 if chunk else len(data)
            if cut == 0:
                tail = data
                continue
            buf, tail = b"\n" + data[:cut], data[cut:]
            first = int(buf[1 : buf.index(b"\t", 1)])
            start_last = buf.rfind(b"\n", 0, len(buf) - 1) + 1
            last = int(buf[start_last : buf.index(b"\t", start_last)])
            if first < prev_last:
                raise BuildError(f"{path}: rows are not sorted by CID")
            prev_last = last
            i, pos = bisect.bisect_left(cids, first), 0
            while i < len(cids) and cids[i] <= last:
                key = b"\n%d\t" % cids[i]
                p = buf.find(key, pos)
                while p >= 0:
                    end = buf.find(b"\n", p + 1)
                    end = len(buf) if end < 0 else end
                    rows = out.setdefault(cids[i], [])
                    if len(rows) < cap:
                        rows.append(buf[p + len(key) : end].rstrip(b"\r").decode("utf-8", "replace"))
                    pos = end
                    p = end if buf.startswith(key, end) else -1
                i += 1
            if not chunk:
                break
    return out


def _stream_job(args):
    return stream_selected(*args)


def load(ctx: BuildContext) -> None:
    (mesh_entry, mesh_path), = ctx.files("pubchem-mesh")
    mesh = {int(line.split("\t", 1)[0]) for line in mesh_path.read_text(encoding="utf-8").splitlines() if line}
    linked = ctx.linked_cids
    selected = sorted(mesh | linked.keys())

    files = {use: ctx.files(use)[0] for use in ("pubchem-mass", "pubchem-title", "pubchem-synonyms")}
    jobs = [
        (files["pubchem-mass"][1], selected, 1),
        (files["pubchem-title"][1], selected, 1),
        (files["pubchem-synonyms"][1], selected, NAMES_INDEXED),
    ]
    with ProcessPoolExecutor(max_workers=3) as pool:
        mass, titles, synonyms = pool.map(_stream_job, jobs)

    for entry, path in [(mesh_entry, mesh_path), *files.values()]:
        ctx.add_source(entry, date.fromtimestamp(path.stat().st_mtime).isoformat())

    weights, isotopes = load_atomic_weights()
    compounds, names = [], []
    skipped_mass = skipped_title = 0
    for cid in selected:
        if cid not in mass:
            skipped_mass += 1
            continue
        if cid not in titles:
            skipped_title += 1
            continue
        formula, mono = mass[cid][0].split("\t")[:2]
        mono = float(mono)
        mm, note = molar_mass(formula, mono, weights, isotopes)
        syns = synonyms.get(cid, [])
        by = "both" if cid in mesh and cid in linked else ("mesh" if cid in mesh else "wikipedia")
        title = titles[cid][0]
        compounds.append((cid, title, formula, mono, mm, note, "|".join(syns[:NAMES_SHOWN]), by, linked.get(cid)))
        names.extend((n, cid, rank) for rank, s in enumerate([title, *syns]) if (n := normalize(s)))

    ctx.con.executemany("INSERT INTO compound VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", compounds)
    ctx.con.executemany("INSERT OR IGNORE INTO compound_name(name, cid, rank) VALUES (?, ?, ?)", names)
    ctx.report["skipped_no_mass"] = skipped_mass
    ctx.report["skipped_no_title"] = skipped_title
