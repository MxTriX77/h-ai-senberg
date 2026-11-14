"""Measurements of an eval run beyond the answer-line scores: where the seconds went (stage log), seconds per
request type against the budgets, domain drift (automatic part) and the code checks (README "Evals")."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any

from evals.items import IN_BASE

ITEM_FIELDS = ("category", "request_type", "lang", "probes", "offline_base", "split")
CHECK_STATUSES = ("passed", "failed", "no_code", "timeout", "error")
SIGNALS = ("exact", "numeric")  # checks that are quick signals: the reference value found in the free text


def item_fields(q: dict[str, Any], stage_log: list[dict[str, Any]], total_s: float, settings: Any) -> dict[str, Any]:
    """The measurement fields of an item log. `settings`: the config's `evals` section (None: no budgets and no
    drift, e.g. when the runner is used without a config). An item of the general schema carries its category,
    request type, language, probes, `offline_base`, split and, when it has one, `family` into the log; `budget_s`
    and `within_budget` follow its request type; `science_calls` counts the model's own calls of a science tool (stage `tool`) and
    `science_research` the verifier's re-searches with one (stage `research`: they check claims, they are no
    drift); `drift_auto` says whether an item outside the science categories has a science call (None for an item
    of a science category or without a category: drift does not apply)."""
    general = "category" in q
    fields: dict[str, Any] = {k: q[k] for k in ITEM_FIELDS} if general else {}
    if general and "family" in q:
        fields["family"] = q["family"]
    if settings is None:
        return {**fields, "science_calls": None, "science_research": None, "drift_auto": None,
                **({"budget_s": None, "within_budget": None} if general else {})}
    science = Counter(s["stage"] for s in stage_log if s["kind"] == "tool" and s.get("tool") in settings.science_tools)
    calls = science["tool"]
    outside = general and q["category"] not in settings.science_categories
    fields |= {"science_calls": calls, "science_research": science["research"],
               "drift_auto": calls > 0 if outside else None}
    if general:
        budget = settings.budgets_s.get(q["request_type"])
        fields |= {"budget_s": budget, "within_budget": None if budget is None else total_s <= budget}
    return fields


def _p90(values: list[float]) -> float | None:
    return sorted(values)[math.ceil(0.9 * len(values)) - 1] if values else None


def _total(records: list[dict[str, Any]], field: str) -> int | None:
    """The sum of a token field over the records that have it; None when none has."""
    values = [r[field] for r in records if r.get(field) is not None]
    return sum(values) if values else None


def summary_blocks(logs: list[dict[str, Any]]) -> dict[str, Any]:
    """The measurement blocks of a summary, over the logs that have a stage log (none for older runs): answers
    the model did not give, answers cut at the token limit, time and tokens per stage, seconds per request type
    against its budget, automatic drift, and the code checks."""
    logs = [r for r in logs if "stages" in r]
    if not logs:
        return {}
    ok = [r for r in logs if r["error"] is None]
    blocks: dict[str, Any] = {
        "no_answer": sum(not r["draft"].strip() for r in ok),
        "truncated": sum(r.get("finish_reason") == "length" for r in ok),
        "total_s": round(sum(r["timings"]["total_s"] for r in logs), 2),
    }
    by_stage: dict[str, list[dict[str, Any]]] = {}
    for r in logs:
        for s in r["stages"]:
            by_stage.setdefault(s["stage"], []).append(s)
    blocks["stages"] = {name: {"records": len(recs), "seconds": round(sum(s["seconds"] or 0 for s in recs), 2),
                               "prompt_tokens": _total(recs, "prompt_tokens"),
                               "completion_tokens": _total(recs, "completion_tokens")}
                        for name, recs in by_stage.items()}
    types: dict[str, Any] = {}
    for kind in sorted({r["request_type"] for r in ok if r.get("request_type")}):
        rows = [r for r in ok if r.get("request_type") == kind]
        seconds, budget = [r["timings"]["total_s"] for r in rows], rows[0].get("budget_s")
        median = statistics.median(seconds)
        types[kind] = {"items": len(rows), "median_s": median, "p90_s": _p90(seconds), "budget_s": budget,
                       "within_budget": None if budget is None else sum(bool(r.get("within_budget")) for r in rows),
                       "median_within_budget": None if budget is None else median <= budget}
    if types:
        blocks["request_types"] = types
    if outside := [r for r in ok if r.get("drift_auto") is not None]:  # the items outside the science categories
        blocks["drift"] = {"items": len(outside), "auto": sum(r["drift_auto"] for r in outside),
                           "science_calls": sum(r["science_calls"] for r in outside),
                           "science_research": sum(r["science_research"] for r in outside)}
    checked = [r for r in ok if r.get("check")]
    tests = [r["check"] for r in checked if r["check"]["type"] == "python_tests"]
    if tests:
        status = Counter(c["final"]["status"] for c in tests)
        blocks["python_tests"] = {
            "items": len(tests), **{s: status.get(s, 0) for s in CHECK_STATUSES},
            "draft_passed": sum(c["draft"]["passed"] for c in tests),
            "broken": sum(c["draft"]["passed"] and not c["final"]["passed"] for c in tests),
            "repaired": sum(c["final"]["passed"] and not c["draft"]["passed"] for c in tests)}
    signals = {}
    for kind in SIGNALS:
        if rows := [r["check"] for r in checked if r["check"]["type"] == kind]:
            signals[kind] = {"items": len(rows), "found": sum(c["final"]["passed"] for c in rows),
                             "draft_found": sum(c["draft"]["passed"] for c in rows),
                             "lost": sum(c["draft"]["passed"] and not c["final"]["passed"] for c in rows),
                             "gained": sum(c["final"]["passed"] and not c["draft"]["passed"] for c in rows)}
    if signals:
        blocks["reference_values"] = signals
    return blocks


def review_summary_general(rows: list[tuple[dict[str, Any], dict[str, Any]]], science: set[str]) -> dict[str, Any]:
    """Review metrics of (review of format 2, item log) pairs. `science`: the science categories: domain drift is
    judged on the other answers only, a needless "I don't know" only where the offline base holds the facts."""
    reviews = [rv for rv, _ in rows]
    lies = [e for rv in reviews for e in rv["falsehoods"]]
    outside = [rv for rv, log in rows if log.get("category") not in science]
    in_base = [rv for rv, log in rows if log.get("offline_base") in IN_BASE]
    return {
        "reviewed": len(rows),
        "confident_falsehoods": len(lies),
        "falsehoods_per_answer": round(len(lies) / len(rows), 2) if rows else None,
        "central_falsehoods": sum(bool(e["central"]) for e in lies),
        "falsehood_kinds": dict(Counter(e["kind"] for e in lies)),
        "mean_score": round(statistics.mean(rv["score"] for rv in reviews), 2) if rows else None,
        "domain_drift": sum(bool(rv["domain_drift"]) for rv in outside),
        "drift_judged": len(outside),
        "needless_idk": sum(bool(rv["needless_idk"]) for rv in in_base),
        "idk_judged": len(in_base),
        "needless_doubts": sum(len(rv["doubts"]) for rv in reviews),
    }
