"""Count the numbers recorded answers state without any source, without a model (README "Evals").

    uv run python -m evals.unsourced RUN [RUN ...] [--sample N]

A run id names a folder under ${DATA_DIR:-data}/evals (or give the folder's path). For each item log the list is
computed again (haisenberg.unsourced) from the stored final answer (as scored: no source list, summary or [S#] ids),
the messages sent, and the tool results the log keeps: the result lines of every tool call in the reasoning trace
(calculator results in full; a lookup shows only its records' labels) and the record values the code checks matched
(`code:` notes of the verification report). Lookup passages are not logged, so a value only a passage states counts
here and not in the live check: the counts are an upper bound. Prints per run the answers and the mean per answer;
with --sample N, N flagged numbers drawn at random (seed 0) with their context. Runs on a holdout are skipped.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

from evals.run import answer_text, read_logs, run_dir
from haisenberg.unsourced import unsourced

_CALL = re.compile(r"> \w+\(.*\)(?:  \[re-search C\d+\])?")  # a tool call in the trace (agent, verifier re-search)
_CLAIM = re.compile(r"- C\d+ ")  # a verifier claim line right after a re-search block
_CODE = re.compile(r"[≈≠] (\S+) \(S\d+\)")  # a record value in a code-check note


def tool_results(log: dict[str, Any]) -> list[str]:
    """The tool results an item log keeps: each tool call's result lines in the trace, and the record values of the
    code checks."""
    blocks: list[list[str]] = []
    block: list[str] | None = None
    for line in (log.get("reasoning") or "").splitlines():
        if _CALL.fullmatch(line):
            block = []
            blocks.append(block)
        elif block is not None and line.startswith("- ") and not _CLAIM.match(line):
            block.append(line[2:])
        else:
            block = None
    notes = [c.get("note", "") for c in (log.get("verification") or {}).get("claims", [])]
    values = [v for note in notes if note.startswith("code:") for v in _CODE.findall(note)]
    return ["\n".join(b) for b in blocks] + ([" ".join(values)] if values else [])


def found_in(log: dict[str, Any]) -> list[dict[str, Any]]:
    conversation = [m["content"] for m in log["messages"] if isinstance(m.get("content"), str)]
    return unsourced(answer_text(log["answer"]), conversation, tool_results(log))


def on_holdout(folder: Path) -> bool:
    path = folder / "manifest.json"
    man = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return "holdout" in json.dumps(man.get("eval_files", [])) or "--holdout" in (man.get("args") or [])


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="run ids (folders under ${DATA_DIR:-data}/evals) or run folders")
    parser.add_argument("--sample", type=int, default=0, help="print N flagged numbers drawn at random, with context")
    args = parser.parse_args(argv)
    flagged: list[tuple[str, str, dict[str, Any]]] = []
    try:
        for run in args.runs:
            folder = run_dir(run)
            if on_holdout(folder):
                print(f"{folder.name}: skipped (holdout)")
                continue
            logs = [r for r in read_logs(folder) if r["answer"].strip()]
            counts = []
            for r in logs:
                found = found_in(r)
                counts.append(len(found))
                flagged += [(folder.name, r["id"], f) for f in found]
            mean = f"{sum(counts) / len(counts):.2f}" if counts else "–"
            print(f"{folder.name}: {len(logs)} answers, {sum(counts)} unsourced numbers, {mean} per answer", flush=True)
    except (OSError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if args.sample:
        print()
        for run, iid, f in random.Random(0).sample(flagged, min(args.sample, len(flagged))):
            print(f"{run} {iid}: {f['text']}  | …{f['context']}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
