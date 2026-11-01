"""Compare eval runs as markdown: one row per run and set, and with --base paired deltas on the same items.

    uv run python -m evals.report RUN_ID... [--base RUN_ID]

A run id names a folder under ${DATA_DIR:-data}/evals (or give the folder's path). Review metrics show when the run has
been reviewed (evals.review merge).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from evals.run import read_logs, run_dir


def _f(v: Any) -> str:
    return "–" if v is None else f"{v:g}" if isinstance(v, float) else str(v)


def _d(v: float | None) -> str:
    return "–" if v is None else f"{v:+g}"


def _rows(runs: list[tuple[str, dict]]) -> list[tuple[str, str, str, dict]]:
    return [(run, name, summary["mode"], s) for run, summary in runs
            for name, s in [*summary["sets"].items(), ("all", summary["all"])]]


def code_table(runs: list[tuple[str, dict]]) -> list[str]:
    lines = ["| run | set | mode | items | errors | empty | solved | facts strict / loose | parts | parts line-only"
             " | verdict parts | verdict parts read from the body | missing lines | false / true corrections"
             " | with artefacts | tool-call leaks | S6 ok: artefacts / chars / latency | latency p50 / p90 s"
             " | unsourced numbers per answer |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for run, name, mode, s in _rows(runs):
        c = s.get("conversation")
        conv = f"{c['artefact_free']} / {c['max_chars']} / {c['max_latency']} of {c['items']}" if c else "–"
        # summaries written before the strict fact score have no fact counts
        facts = f"{s['facts_correct']} / {s['loose_correct']} of {s['facts']}" if s.get("facts") else "–"
        # summaries written before body-first verdict reading have no line-only total and no body count
        line_only = f"{s['line_parts_right']}/{s['parts_total']}" if "line_parts_right" in s else "–"
        lines.append(
            f"| {run} | {name} | {mode} | {s['items']} | {s['errors']} | {s['empty_answers']}"
            f" | {s['correct']}/{s['scored']} ({_f(s['correct_pct'])} %) | {facts} | {s['parts_right']}/{s['parts_total']}"
            f" ({_f(s['parts_pct'])} %) | {line_only} | {s['verdict_parts_right']}/{s['verdict_parts_total']}"
            f" | {_f(s.get('body_verdict_parts'))}"
            f" | {s['missing_lines']}"
            f" | {s['false_corrections']} / {s['true_corrections']} | {s['with_artefacts']} | {s['tool_call_leaks']}"
            f" | {conv} | {_f(s['latency_median_s'])} / {_f(s['latency_p90_s'])}"
            f" | {_f(s.get('unsourced_per_answer'))} |")  # none in summaries written before the flag
    return lines


def review_table(runs: list[tuple[str, dict]]) -> list[str]:
    reviewed = [(run, summary["review"] | {"mode": summary["mode"]}) for run, summary in runs if "review" in summary]
    if not reviewed:
        return []
    lines = ["| run | set | mode | reviewed | mean score | defects found | verdicts right | omissions"
             " | unsupported additions | unsupported as verified | final vs analysis | must_not violated | new errors |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for run, name, mode, s in _rows(reviewed):
        errors = ", ".join(f"{tag} {n}" for tag, n in sorted(s["new_errors"].items())) or "–"
        lines.append(
            f"| {run} | {name} | {mode} | {s['reviewed']} | {_f(s['mean_score'])} | {_f(s['defects_found_pct'])} %"
            f" | {_f(s['verdict_pct'])} % | {s['omissions']} | {s['unsupported_additions']}"
            f" | {s['unsupported_as_verified']} | {s['final_vs_analysis']} | {s['must_not_violations']} | {errors} |")
    return lines


def _parts_right(r: dict) -> int:
    return sum(r["scores"]["final"]["parts"].values()) if r["scores"] else 0


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def delta_table(base: Path, runs: list[Path]) -> list[str]:
    """Per run and set, on the items both runs have: solved, parts, better/worse items, artefacts, latency,
    and the mean review score when both runs were reviewed."""
    old = {r["id"]: r for r in read_logs(base)}
    old_reviews = _reviews(base)
    lines = [f"Paired against {base.name}:", "",
             "| run | set | paired | solved base → run | Δ parts right | items better / worse"
             " | with artefacts base → run | Δ latency p50 s | Δ review score |",
             "|---|---|---|---|---|---|---|---|---|"]
    for folder in runs:
        logs, reviews = read_logs(folder), _reviews(folder)
        for name in sorted({r["set"] for r in logs}):
            pairs = [(old[r["id"]], r) for r in logs if r["set"] == name and r["id"] in old]
            if not pairs:
                continue
            gain = [_parts_right(b) - _parts_right(a) for a, b in pairs]
            p50 = [_median([x["timings"]["total_s"] for x in side if x["error"] is None]) for side in zip(*pairs)]
            scored = [(old_reviews[a["id"]]["score"], reviews[b["id"]]["score"]) for a, b in pairs
                      if a["id"] in old_reviews and b["id"] in reviews]
            score = round(statistics.mean(y - x for x, y in scored), 2) if scored else None
            lines.append(
                f"| {folder.name} | {name} | {len(pairs)} | {sum(bool(a['correct']) for a, _ in pairs)} →"
                f" {sum(bool(b['correct']) for _, b in pairs)} | {sum(gain):+d} | {sum(g > 0 for g in gain)} /"
                f" {sum(g < 0 for g in gain)} | {sum(bool(a['artefacts']) for a, _ in pairs)} →"
                f" {sum(bool(b['artefacts']) for _, b in pairs)}"
                f" | {_d(None if None in p50 else round(p50[1] - p50[0], 2))} | {_d(score)} |")
    return lines


def _reviews(folder: Path) -> dict[str, dict]:
    path = folder / "reviews.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def report(runs: list[Path], base: Path | None = None) -> str:
    summaries = [(f.name, json.loads((f / "summary.json").read_text(encoding="utf-8"))) for f in runs]
    blocks = [code_table(summaries), review_table(summaries)]
    if base is not None:
        blocks.append(delta_table(base, [f for f in runs if f.resolve() != base.resolve()]))
    return "\n\n".join("\n".join(b) for b in blocks if b) + "\n"


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="run ids (folders under ${DATA_DIR:-data}/evals) or run folders")
    parser.add_argument("--base", help="run to pair the others against, item by item")
    args = parser.parse_args(argv)
    try:
        print(report([run_dir(r) for r in args.runs], run_dir(args.base) if args.base else None), end="")
    except (OSError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
