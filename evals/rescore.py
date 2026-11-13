"""Rescore recorded eval runs from their stored answers, without a model (README "Evals").

    uv run python -m evals.rescore RUN [RUN ...] [--dry-run]

A run id names a folder under ${DATA_DIR:-data}/evals (or give the folder's path). Each item log is scored again
against the run's eval files as they are now (correct, scores, unmarked_contradiction), in place: the first
rescore keeps the previous scores as `scores_before_rescore`; then summary.json is rewritten. A log whose item is
gone or whose prompt differs from the item's current one stays as it was. Prints per run the
parts right before → after (and the line-only total) and how many verdict parts were read from the answer body.
An item of the general schema gets its code check again (`check`: the python tests, run again when the config
allows them, and the reference value signals; the first rescore keeps the previous one as `check_before_rescore`),
and the line then also gives the checks passed before → after.
--dry-run prints the same and writes nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from evals.checks import run_check
from evals.run import log_scores, read_logs, run_dir, run_items, write_summary
from haisenberg.config import ConfigError, load_config


def _parts(scores: dict[str, Any] | None, key: str = "parts") -> list[bool]:
    """The final answer's parts (or its line-only parts: the parts where a score has none)."""
    return list(scores["final"].get(key, scores["final"]["parts"]).values()) if scores else []


def _body(scores: dict[str, Any] | None) -> int:
    return sum(s == "body" for s in scores["final"].get("sources", {}).values()) if scores else 0


def _passed(check: dict[str, Any] | None) -> int:
    return int(bool(check and check["final"]["passed"]))


def rescore(folder: Path, write: bool = True, python: Any = None) -> dict[str, int]:
    """Rescore every item log of a run whose item is in its eval files; returns the counts the line prints.
    `python`: the config's `evals.python_tests` settings; without them a python_tests check stays as stored (the
    reference value signals are computed again either way)."""
    man = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    items = run_items(folder)
    counts = {"items": 0, "skipped": 0, "before": 0, "after": 0, "line_only": 0, "total": 0, "body": 0,
              "checks": 0, "checks_before": 0, "checks_after": 0}
    for log in read_logs(folder):
        item = items.get(log["id"])
        if item is None or log["messages"][-1]["content"] != item["question"]:
            counts["skipped"] += 1  # the item left its eval file or its prompt changed: the log stays as it was
            continue
        before = log.get("scores_before_rescore", log["scores"])
        new = log_scores(log["answer"], log["draft"], item, log["error"], log["verification"])
        counts["items"] += 1
        counts["before"] += sum(_parts(before))
        counts["after"] += sum(_parts(new["scores"]))
        counts["line_only"] += sum(_parts(new["scores"], "line_parts"))
        counts["total"] += len(_parts(new["scores"]))
        counts["body"] += _body(new["scores"])
        if "category" in item and not log["error"]:  # the general schema: its code check, on the stored texts
            kind = item["reference"]["check"]["type"]
            stored = log.get("check")
            check = stored if kind == "python_tests" and python is None else \
                run_check(item, log["answer"], log["draft"], python)
            counts["checks"] += check is not None
            counts["checks_before"] += _passed(log.get("check_before_rescore", stored))
            counts["checks_after"] += _passed(check)
            new = {**new, "check": check, "check_before_rescore": log.get("check_before_rescore", stored),
                   "correct": check["final"]["passed"] if check else None}
        if write:
            log = {**log, "scores_before_rescore": before, **new}
            (folder / "items" / f"{log['id']}.json").write_text(json.dumps(log, ensure_ascii=False, indent=2),
                                                                encoding="utf-8")
    if write:
        write_summary(folder, man["mode"])
    return counts


def line(name: str, c: dict[str, int]) -> str:
    return (f"{name}: parts {c['before']} → {c['after']} of {c['total']} (line-only {c['line_only']}),"
            f" {c['body']} verdict parts read from the body, {c['items']} items"
            + (f", code checks passed {c['checks_before']} → {c['checks_after']} of {c['checks']}"
               if c.get("checks") else "")
            + (f", {c['skipped']} skipped (item gone or prompt changed)" if c["skipped"] else ""))


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="run ids (folders under ${DATA_DIR:-data}/evals) or run folders")
    parser.add_argument("--dry-run", action="store_true", help="print the new scores, write nothing")
    args = parser.parse_args(argv)
    total: dict[str, int] = {}
    try:
        python = load_config().evals.python_tests  # the limits of the python tests, when they are to run again
    except ConfigError as e:
        python = None
        print(f"note: no config ({e}): python tests are not run again", file=sys.stderr)
    try:
        for run in args.runs:
            c = rescore(run_dir(run), write=not args.dry_run, python=python)
            print(line(run_dir(run).name, c), flush=True)
            total = {k: total.get(k, 0) + v for k, v in c.items()}
    except (OSError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if len(args.runs) > 1:
        print(line("total", total))
    return 0


if __name__ == "__main__":
    sys.exit(main())
