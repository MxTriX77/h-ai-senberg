"""Compare eval runs as markdown: one row per run and set, and with --base paired deltas on the same items.

    uv run python -m evals.report RUN_ID... [--base RUN_ID]

A run id names a folder under ${DATA_DIR:-data}/evals (or give the folder's path). Runs of the older sets get the
table of answer-line metrics; runs with a stage log get the measurement tables (seconds by request type against
the budgets, time and tokens per stage, answers not given, automatic drift and the code checks). Review metrics
show when the run has been reviewed (evals.review merge).
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
    """The answer-line metrics; runs whose items are all of the general schema have none and are left out."""
    runs = [(run, summary) for run, summary in runs if not summary.get("general")]
    if not runs:
        return []
    lines = ["| run | set | mode | items | errors | empty | solved | facts strict / loose | parts | parts line-only"
             " | verdict parts | verdict parts read from the body | missing lines | false / true corrections"
             " | with artefacts | tool-call leaks | S6 ok: artefacts / chars / latency | latency p50 / p90 s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
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
            f" | {conv} | {_f(s['latency_median_s'])} / {_f(s['latency_p90_s'])} |")
    return lines


def _per_answer(n: int, of: int) -> str:
    return f"{n} of {of}" if of else "–"


def time_table(runs: list[tuple[str, dict]]) -> list[str]:
    """Seconds per answer by request type against its budget, for runs that record request types."""
    lines = ["| run | mode | request type | items | median s | p90 s | budget s | items within budget"
             " | median within budget |", "|---|---|---|---|---|---|---|---|---|"]
    for run, summary in runs:
        for kind, t in (summary["all"].get("request_types") or {}).items():
            within = "–" if t["budget_s"] is None else _per_answer(t["within_budget"], t["items"])
            median = "–" if t["median_within_budget"] is None else "yes" if t["median_within_budget"] else "NO"
            lines.append(f"| {run} | {summary['mode']} | {kind} | {t['items']} | {_f(t['median_s'])} | {_f(t['p90_s'])}"
                         f" | {_f(t['budget_s'])} | {within} | {median} |")
    return lines if len(lines) > 2 else []


def stage_table(runs: list[tuple[str, dict]]) -> list[str]:
    """Where the seconds went: every stage of the run with its records, seconds, share of the run's total seconds
    and tokens (model calls of a backend that reports usage)."""
    lines = ["| run | mode | stage | records | seconds | % of total seconds | prompt tokens | completion tokens |",
             "|---|---|---|---|---|---|---|---|"]
    for run, summary in runs:
        total = summary["all"].get("total_s")
        for name, s in (summary["all"].get("stages") or {}).items():
            share = _f(round(100 * s["seconds"] / total, 1)) if total else "–"
            lines.append(f"| {run} | {summary['mode']} | {name} | {s['records']} | {_f(s['seconds'])} | {share}"
                         f" | {_f(s['prompt_tokens'])} | {_f(s['completion_tokens'])} |")
    return lines if len(lines) > 2 else []


def check_table(runs: list[tuple[str, dict]]) -> list[str]:
    """Answers not given, automatic drift and the code checks, per run and set, for runs with a stage log."""
    lines = ["| run | set | mode | items | errors | no answer | cut at the token limit | drift (science call on a"
             " non-science item) | python tests passed | failed / no code / timeout / error | broken / repaired by"
             " the suite | reference values found: exact | numeric | lost / gained by the suite | latency p50 / p90 s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for run, name, mode, s in _rows([(run, summary) for run, summary in runs if "stages" in summary["all"]]):
        drift, tests, found = s.get("drift"), s.get("python_tests"), s.get("reference_values") or {}
        exact, numeric = found.get("exact"), found.get("numeric")
        lines.append(
            f"| {run} | {name} | {mode} | {s['items']} | {s['errors']} | {_f(s.get('no_answer'))}"
            f" | {_f(s.get('truncated'))}"
            f" | {_per_answer(drift['auto'], drift['items']) if drift else '–'}"
            f" | {_per_answer(tests['passed'], tests['items']) if tests else '–'}"
            f" | {'{failed} / {no_code} / {timeout} / {error}'.format(**tests) if tests else '–'}"
            f" | {'{broken} / {repaired}'.format(**tests) if tests else '–'}"
            f" | {_per_answer(exact['found'], exact['items']) if exact else '–'}"
            f" | {_per_answer(numeric['found'], numeric['items']) if numeric else '–'}"
            f" | {sum(v['lost'] for v in found.values())} / {sum(v['gained'] for v in found.values())}"
            f" | {_f(s['latency_median_s'])} / {_f(s['latency_p90_s'])} |")
    return lines if len(lines) > 2 else []


def general_review_table(runs: list[tuple[str, dict]]) -> list[str]:
    """The review metrics of runs reviewed with a format 2 pack (items of the general schema)."""
    reviewed = [(run, summary["review"] | {"mode": summary["mode"]}) for run, summary in runs
                if "falsehood_kinds" in summary.get("review", {}).get("all", {})]
    if not reviewed:
        return []
    lines = ["| run | set | mode | reviewed | confident falsehoods | per answer | central | mean score"
             " | domain drift | needless \"I don't know\" | needless doubts | falsehood kinds |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for run, name, mode, s in _rows(reviewed):
        kinds = ", ".join(f"{kind} {n}" for kind, n in sorted(s["falsehood_kinds"].items())) or "–"
        lines.append(
            f"| {run} | {name} | {mode} | {s['reviewed']} | {s['confident_falsehoods']}"
            f" | {_f(s['falsehoods_per_answer'])} | {s['central_falsehoods']} | {_f(s['mean_score'])}"
            f" | {_per_answer(s['domain_drift'], s['drift_judged'])}"
            f" | {_per_answer(s['needless_idk'], s['idk_judged'])} | {s['needless_doubts']} | {kinds} |")
    return lines


def review_table(runs: list[tuple[str, dict]]) -> list[str]:
    reviewed = [(run, summary["review"] | {"mode": summary["mode"]}) for run, summary in runs
                if "review" in summary and "falsehood_kinds" not in summary["review"]["all"]]
    if not reviewed:
        return []
    lines = ["| run | set | mode | reviewed | confident falsehoods per answer | mean score | defects found"
             " | verdicts right | omissions | unsupported additions | unsupported as verified | final vs analysis"
             " | must_not violated | new errors |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for run, name, mode, s in _rows(reviewed):
        errors = ", ".join(f"{tag} {n}" for tag, n in sorted(s["new_errors"].items())) or "–"
        lines.append(
            f"| {run} | {name} | {mode} | {s['reviewed']} | {_f(s.get('falsehoods_per_answer'))}"
            f" | {_f(s['mean_score'])} | {_f(s['defects_found_pct'])} %"
            f" | {_f(s['verdict_pct'])} % | {s['omissions']} | {s['unsupported_additions']}"
            f" | {s['unsupported_as_verified']} | {s['final_vs_analysis']} | {s['must_not_violations']} | {errors} |")
    return lines


def _parts_right(r: dict) -> int:
    return sum(r["scores"]["final"]["parts"].values()) if r["scores"] else 0


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _passed(logs: list[dict], kinds: tuple[str, ...]) -> str:
    """How many of the logs' code checks of these kinds passed, as "n of m"; "–" without any."""
    checks = [r["check"] for r in logs if r.get("check") and r["check"]["type"] in kinds]
    return _per_answer(sum(c["final"]["passed"] for c in checks), len(checks)) if checks else "–"


def general_delta_table(base: Path, runs: list[Path]) -> list[str]:
    """Per run and set (category), on the items of the general schema both runs have: automatic drift, the
    code checks, answers not given, latency, and, when both runs were reviewed, the confident falsehoods and the
    mean review score."""
    old = {r["id"]: r for r in read_logs(base) if r.get("category")}
    old_reviews = _reviews(base)
    lines = [f"Paired against {base.name} (items of the general schema):", "",
             "| run | set | paired | drift base → run | python tests passed base → run"
             " | reference values found base → run | no answer base → run | Δ latency p50 s"
             " | confident falsehoods base → run | Δ review score |", "|---|---|---|---|---|---|---|---|---|---|"]
    for folder in runs:
        logs, reviews = [r for r in read_logs(folder) if r.get("category")], _reviews(folder)
        for name in [*sorted({r["set"] for r in logs}), "all"]:
            pairs = [(old[r["id"]], r) for r in logs if name in (r["set"], "all") and r["id"] in old]
            if not pairs:
                continue
            a, b = [x for x, _ in pairs], [y for _, y in pairs]
            drift = " → ".join(str(sum(bool(r.get("drift_auto")) for r in side)) for side in (a, b))
            none = " → ".join(str(sum(r["error"] is None and not r["draft"].strip() for r in side)) for side in (a, b))
            p50 = [_median([x["timings"]["total_s"] for x in side if x["error"] is None]) for side in (a, b)]
            both = [(old_reviews[x["id"]], reviews[y["id"]]) for x, y in pairs
                    if x["id"] in old_reviews and y["id"] in reviews]
            score = round(statistics.mean(y["score"] - x["score"] for x, y in both), 2) if both else None
            lies = (f"{sum(x['confident_falsehoods'] for x, _ in both)} →"
                    f" {sum(y['confident_falsehoods'] for _, y in both)} ({len(both)})" if both else "–")
            lines.append(
                f"| {folder.name} | {name} | {len(pairs)} | {drift}"
                f" | {_passed(a, ('python_tests',))} → {_passed(b, ('python_tests',))}"
                f" | {_passed(a, ('exact', 'numeric'))} → {_passed(b, ('exact', 'numeric'))} | {none}"
                f" | {_d(None if None in p50 else round(p50[1] - p50[0], 2))} | {lies} | {_d(score)} |")
    return lines if len(lines) > 4 else []


def delta_table(base: Path, runs: list[Path]) -> list[str]:
    """Per run and set, on the items both runs have: solved, parts, better/worse items, artefacts, latency,
    and, when both runs were reviewed, the confident falsehoods base → run and the mean review score. Items of
    the general schema have their own table (general_delta_table)."""
    old = {r["id"]: r for r in read_logs(base) if not r.get("category")}
    old_reviews = _reviews(base)
    lines = [f"Paired against {base.name}:", "",
             "| run | set | paired | solved base → run | Δ parts right | items better / worse"
             " | with artefacts base → run | Δ latency p50 s | confident falsehoods base → run | Δ review score |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for folder in runs:
        logs, reviews = [r for r in read_logs(folder) if not r.get("category")], _reviews(folder)
        for name in sorted({r["set"] for r in logs}):
            pairs = [(old[r["id"]], r) for r in logs if r["set"] == name and r["id"] in old]
            if not pairs:
                continue
            gain = [_parts_right(b) - _parts_right(a) for a, b in pairs]
            p50 = [_median([x["timings"]["total_s"] for x in side if x["error"] is None]) for side in zip(*pairs)]
            scored = [(old_reviews[a["id"]]["score"], reviews[b["id"]]["score"]) for a, b in pairs
                      if a["id"] in old_reviews and b["id"] in reviews]
            score = round(statistics.mean(y - x for x, y in scored), 2) if scored else None
            graded = [(old_reviews[a["id"]], reviews[b["id"]]) for a, b in pairs if a["id"] in old_reviews
                      and b["id"] in reviews and "confident_falsehoods" in old_reviews[a["id"]]
                      and "confident_falsehoods" in reviews[b["id"]]]
            falsehoods = (f"{sum(x['confident_falsehoods'] for x, _ in graded)} →"
                          f" {sum(y['confident_falsehoods'] for _, y in graded)} ({len(graded)})" if graded else "–")
            lines.append(
                f"| {folder.name} | {name} | {len(pairs)} | {sum(bool(a['correct']) for a, _ in pairs)} →"
                f" {sum(bool(b['correct']) for _, b in pairs)} | {sum(gain):+d} | {sum(g > 0 for g in gain)} /"
                f" {sum(g < 0 for g in gain)} | {sum(bool(a['artefacts']) for a, _ in pairs)} →"
                f" {sum(bool(b['artefacts']) for _, b in pairs)}"
                f" | {_d(None if None in p50 else round(p50[1] - p50[0], 2))} | {falsehoods} | {_d(score)} |")
    return lines if len(lines) > 4 else []


def _reviews(folder: Path) -> dict[str, dict]:
    path = folder / "reviews.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def report(runs: list[Path], base: Path | None = None) -> str:
    summaries = [(f.name, json.loads((f / "summary.json").read_text(encoding="utf-8"))) for f in runs]
    blocks = [code_table(summaries), check_table(summaries), time_table(summaries), stage_table(summaries),
              review_table(summaries), general_review_table(summaries)]
    if base is not None:
        others = [f for f in runs if f.resolve() != base.resolve()]
        blocks += [delta_table(base, others), general_delta_table(base, others)]
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
