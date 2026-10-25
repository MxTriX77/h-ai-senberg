"""Query the Layer 2 literature store from the command line.

    uv run haisenberg-l2 search "coupled equilibria" [-k 5] [--domain chemistry] [--db FILE] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys

from haisenberg.config import ConfigError, load_config
from haisenberg.l2.store import L2Error, L2Store


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="haisenberg-l2", description="Query the Layer 2 literature store.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("search", help="passages from textbooks, full articles, Q&A and definitions by keywords")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5, help="maximum number of passages (default 5)")
    p.add_argument("--domain", help="only passages of this domain")
    p.add_argument("--db", help="store path (default: retrieval.layer2_literature_db)")
    p.add_argument("--json", action="store_true", help="print the records as a JSON array")
    args = parser.parse_args(argv)

    try:
        store = L2Store(args.db or load_config().retrieval.layer2_literature_db)
        records = store.search(args.query, args.k, args.domain)
    except (L2Error, ConfigError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps([r.to_dict() for r in records], ensure_ascii=False, indent=2))
    elif not records:
        print("no results")
    else:
        for r in records:
            c = r.citation
            print(f"{r.label()}  [{', '.join(r.domains)}; bm25 {r.score}]")
            print(f"    [{c.source} {c.version}] {c.license} | {c.url}")
            print("    " + r.text.replace("\n", "\n    "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
