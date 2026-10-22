"""Blind review of eval answers: pack anonymised answers for a reviewer, then merge the reviews into the runs.

    uv run python -m evals.review pack --runs RUN_ID... --out DIR [--seed N]
    uv run python -m evals.review merge DIR

`pack` shuffles every answer of the runs (items that failed are left out), gives each a blind id R001, ...
and writes DIR/pack.md (blind id, prompt, key, final answer: nothing about run, mode, model or trace) and
DIR/mapping.json (blind id -> run and item; not for the reviewer). The reviewer writes DIR/reviews.yaml
(README "Evals"). `merge` validates it, writes reviews.json (item id -> review) into each run folder and adds
the review metrics to its summary.json.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import yaml

from evals.run import REVIEW_COUNTS, key_of, read_logs, run_dir, run_items, write_summary

LISTS = ("defects_found", "defects_missed", "omissions", "must_not_violated", "new_errors")


def _prompt(messages: list[dict[str, str]]) -> str:
    *turns, last = messages
    return "".join(f"**{m['role']}:** {m['content']}\n\n" for m in turns) + last["content"]


def pack(runs: list[Path], out: Path, seed: int | None = None) -> int:
    """Write pack.md and mapping.json; returns the number of answers."""
    seed = random.randrange(2**32) if seed is None else seed
    entries = []
    for folder in runs:
        items = run_items(folder)
        entries += [(folder.name, r, items[r["id"]]) for r in read_logs(folder) if r["error"] is None]
    random.Random(seed).shuffle(entries)
    mapping: dict[str, Any] = {"seed": seed, "runs": {f.name: str(f.resolve()) for f in runs}, "answers": {}}
    blocks = [f"# Review pack\n\n{len(entries)} answers. Review each against its key and write one block per answer"
              " to reviews.yaml.\n"]
    for n, (run, r, item) in enumerate(entries, 1):
        blind = f"R{n:03d}"
        mapping["answers"][blind] = {"run": run, "item": r["id"]}
        key = yaml.safe_dump(key_of(item), allow_unicode=True, sort_keys=False)
        blocks.append(f"---\n\n## {blind}\n\n### Prompt\n\n{_prompt(r['messages'])}\n\n### Key\n\n```yaml\n{key}```\n\n"
                      f"### Answer\n\n{r['answer']}\n")
    out.mkdir(parents=True, exist_ok=True)
    (out / "pack.md").write_text("\n".join(blocks), encoding="utf-8")
    (out / "mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(entries)


def check_reviews(reviews: Any, keys: dict[str, dict[str, Any]]) -> list[str]:
    """What is wrong with reviews.yaml, one line per problem (empty when it is valid). `keys`: blind id -> key."""
    if not isinstance(reviews, list):
        return ["reviews.yaml must be a list of review blocks (- blind_id: R001 ...)"]
    problems, seen = [], set()
    for n, rv in enumerate(reviews, 1):
        blind = rv.get("blind_id") if isinstance(rv, dict) else None
        if blind not in keys:
            problems.append(f"block {n}: unknown or missing blind_id {blind!r}")
            continue
        if blind in seen:
            problems.append(f"{blind}: reviewed twice")
        seen.add(blind)
        key, score = keys[blind], rv.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 10:
            problems.append(f"{blind}: score must be a number from 0 to 10, got {score!r}")
        for field in LISTS:
            if not isinstance(rv.get(field, []), list):
                problems.append(f"{blind}: {field} must be a list")
        for field in REVIEW_COUNTS:
            v = rv.get(field, 0)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                problems.append(f"{blind}: {field} must be a whole number >= 0, got {v!r}")
        planted = {d.get("id") for d in key.get("defects") or []}
        named = [d for f in ("defects_found", "defects_missed") if isinstance(rv.get(f), list) for d in rv[f]]
        if unknown := [d for d in named if d not in planted]:
            problems.append(f"{blind}: defects not in the key: {unknown} (key has {sorted(planted) or 'none'})")
        verdicts = rv.get("verdicts", {})
        if not isinstance(verdicts, dict):
            problems.append(f"{blind}: verdicts must be a mapping, e.g. {{A: incorrect}}")
        elif unknown := [v for v in verdicts if v not in (key.get("verdicts") or {})]:
            problems.append(f"{blind}: verdicts for items not in the key: {unknown}")
        if isinstance(rv.get("new_errors"), list) and not all(isinstance(e, dict) and e.get("tag") for e in rv["new_errors"]):
            problems.append(f"{blind}: each new_errors entry needs a tag, e.g. {{tag: F4, what: ...}}")
    if missing := sorted(set(keys) - seen):
        problems.append(f"no review for {', '.join(missing)}")
    return problems


def merge(folder: Path) -> list[Path]:
    """Validate folder/reviews.yaml and merge it into every run of the pack; returns the run folders."""
    mapping = json.loads((folder / "mapping.json").read_text(encoding="utf-8"))
    reviews = yaml.safe_load((folder / "reviews.yaml").read_text(encoding="utf-8"))
    runs = {name: Path(path) for name, path in mapping["runs"].items()}
    items = {name: run_items(path) for name, path in runs.items()}
    answers = mapping["answers"]
    keys = {blind: key_of(items[a["run"]][a["item"]]) for blind, a in answers.items()}
    if problems := check_reviews(reviews, keys):
        raise ValueError(f"{folder / 'reviews.yaml'}:\n" + "\n".join(f"- {p}" for p in problems))
    for name, path in runs.items():
        mine = {answers[rv["blind_id"]]["item"]: rv for rv in reviews if answers[rv["blind_id"]]["run"] == name}
        (path / "reviews.json").write_text(json.dumps(mine, ensure_ascii=False, indent=2), encoding="utf-8")
        write_summary(path, json.loads((path / "manifest.json").read_text(encoding="utf-8"))["mode"])
    return list(runs.values())


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pack", help="write pack.md and mapping.json for a blind reviewer")
    p.add_argument("--runs", nargs="+", required=True, help="run ids (folders under ${DATA_DIR:-data}/evals) or run folders")
    p.add_argument("--out", type=Path, required=True, help="pack folder")
    p.add_argument("--seed", type=int, help="shuffle seed (default: random; kept in mapping.json)")
    m = sub.add_parser("merge", help="validate DIR/reviews.yaml and merge it into the runs")
    m.add_argument("dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "pack":
            n = pack([run_dir(r) for r in args.runs], args.out, args.seed)
            print(f"{n} answers in {args.out / 'pack.md'} (mapping.json is not for the reviewer)")
        else:
            for path in merge(args.dir):
                print(f"reviews merged into {path}")
    except (OSError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
