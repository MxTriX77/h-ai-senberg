"""Query the Layer 1 store from the command line.

    uv run haisenberg-l1 compound caffeine [-k 5] [--db FILE] [--json]
    uv run haisenberg-l1 constant "Planck constant" [-k 5] [--db FILE] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys

from haisenberg.config import ConfigError
from haisenberg.l1.store import CompoundRecord, ConstantRecord, L1Error, L1Store


def _show_constant(r: ConstantRecord) -> str:
    flags = " (exact)" if r.exact else f" ± {r.uncertainty:g}"
    return f"{r.name} = {r.value_text} {r.unit}".rstrip() + flags


def _show_compound(r: CompoundRecord) -> str:
    mass = f"{r.molar_mass:.3f} g/mol" if r.molar_mass is not None else f"molar mass n/a ({r.molar_mass_note})"
    also = f" | {', '.join(r.synonyms[:4])}" if r.synonyms else ""
    return f"CID {r.cid} {r.title}: {r.formula}, {mass}{also}"


def _citation_line(c) -> str:
    return f"    [{c.source} {c.version}] {c.license} | {c.url}"


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="haisenberg-l1", description="Query the Layer 1 store.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_ in [
        ("compound", "PubChem compound by CID, name, CAS number or formula"),
        ("constant", "CODATA constant by name"),
    ]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("query")
        p.add_argument("-k", type=int, default=5, help="maximum number of records (default 5)")
        p.add_argument("--db", help="store path (default: retrieval.layer1_encyclopedia_db)")
        p.add_argument("--json", action="store_true", help="print the records as a JSON array")
    args = parser.parse_args(argv)

    try:
        store = L1Store(args.db)
        lookup = {"compound": store.lookup_compound, "constant": store.lookup_constant}[args.command]
        records = lookup(args.query, args.k)
    except (L1Error, ConfigError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps([r.to_dict() for r in records], ensure_ascii=False, indent=2))
    elif not records:
        print("no results")
    else:
        show = {"compound": _show_compound, "constant": _show_constant}[args.command]
        for r in records:
            print(show(r))
            print(_citation_line(r.citation))
    return 0


if __name__ == "__main__":
    sys.exit(main())
