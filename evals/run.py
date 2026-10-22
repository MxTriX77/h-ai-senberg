"""Run the eval set through the real pipeline (tools, tool loop, verifier) and score it.

    uv run python -m evals.run [--limit N] [--ids ID,ID] [--questions FILE] [--out FILE]

The backend, store and settings come from the config and environment, exactly as for the chat
(`API_BASE_URL`, `DATA_DIR`, `MODEL_PROFILE`, ...). The JSON report is rewritten after every question.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

from haisenberg.agent import Agent
from haisenberg.config import ConfigError, load_config
from haisenberg.l1.text import to_hill
from haisenberg.llm import LLMClient, UpstreamError
from haisenberg.tools import Knowledge
from haisenberg.verifier import read_formulas, read_numbers

QUESTIONS = Path(__file__).resolve().parent / "questions.yaml"
KINDS = ("number", "formula", "text")


def load_questions(path: Path = QUESTIONS) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """(source key -> license note, questions); raises ValueError naming the bad question."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    sources, questions = data["sources"], data["questions"]
    seen: set[str] = set()
    for q in questions:
        qid = q.get("id")
        if not qid or qid in seen:
            raise ValueError(f"{path}: missing or duplicate id {qid!r}")
        seen.add(qid)
        if q.get("source") not in sources or q.get("domain") not in ("chemistry", "physics") or not q.get("question"):
            raise ValueError(f"{path}: question {qid}: needs a known source, a domain and the question")
        answer = q.get("answer") or {}
        kinds = [k for k in KINDS if k in answer]
        if len(kinds) != 1:
            raise ValueError(f"{path}: question {qid}: answer needs exactly one of {KINDS}")
        if "number" in answer and ("abs_tol" in answer) == ("rel_tol" in answer):
            raise ValueError(f"{path}: question {qid}: a number answer needs abs_tol or rel_tol")
        if "text" in answer and not (isinstance(answer["text"], list) and answer["text"]):
            raise ValueError(f"{path}: question {qid}: text answer must be a non-empty list")
    return sources, questions


def answer_text(answer: str) -> str:
    """The answer without its source list, verification summary and [S#] markers."""
    text = answer.split("\n\nSources:\n", 1)[0]
    text = re.sub(r"\*Verification:[^\n]*\*", "", text)
    return re.sub(r"\[S\d+(?:\s*[,;]\s*S?\d+)*\]", "", text)


def is_correct(answer: str, gold: dict[str, Any]) -> bool:
    text = answer_text(answer)
    if "number" in gold:
        value = float(gold["number"])
        tol = gold["abs_tol"] if "abs_tol" in gold else gold["rel_tol"] * abs(value)
        return any(abs(n.value - value) <= tol + 1e-12 * abs(value) for n in read_numbers(text))
    if "formula" in gold:
        want = to_hill(gold["formula"])
        return any(hill == want for hill, _, _ in read_formulas(text, need_digit=False))
    return any(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", text, re.I) for w in gold["text"])


def pick(questions: list[dict], ids: list[str] | None = None, limit: int | None = None) -> list[dict]:
    """By id; or `limit` questions shared out evenly between the source/domain groups and spread
    evenly over each group (a small run still covers every kind of question)."""
    if ids:
        wanted = set(ids)
        return [q for q in questions if q["id"] in wanted]
    if not limit or limit >= len(questions):
        return questions
    groups: dict[str, list[dict]] = {}
    for q in questions:
        groups.setdefault(f"{q['source']}/{q['domain']}", []).append(q)
    quota = dict.fromkeys(groups, 0)
    while sum(quota.values()) < limit:
        for g in groups:
            if quota[g] < len(groups[g]) and sum(quota.values()) < limit:
                quota[g] += 1
    chosen = {id(groups[g][i * len(groups[g]) // n]) for g, n in quota.items() for i in range(n)}
    return [q for q in questions if id(q) in chosen]


async def run_one(agent: Agent, q: dict[str, Any]) -> dict[str, Any]:
    t0, answer, report, error = time.monotonic(), "", {}, None
    try:
        async for item in agent.run([{"role": "user", "content": q["question"]}]):
            answer += item.get("content", "")
            report = item.get("verification", report)
    except (UpstreamError, httpx.HTTPError) as e:
        error = str(e)
    claims = report.get("claims", [])
    correct = error is None and is_correct(answer, q["answer"])
    all_supported = bool(claims) and all(c["outcome"] == "supported" for c in claims)
    return {
        "id": q["id"],
        "question": q["question"],
        "known": q["answer"],
        "answer": answer,
        "correct": correct,
        "unmarked_contradiction": error is None and not correct and all_supported,
        "verification": report,
        "seconds": round(time.monotonic() - t0, 1),
        "error": error,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    counts = {o: sum(r["verification"].get("counts", {}).get(o, 0) for r in results)
              for o in ("supported", "unverified", "corrected", "removed")}
    claims = sum(counts.values())
    return {
        "questions": n,
        "correct_pct": round(100 * sum(r["correct"] for r in results) / n, 1) if n else 0.0,
        "supported_claims_pct": round(100 * counts["supported"] / claims, 1) if claims else 0.0,
        "unmarked_contradictions": sum(r["unmarked_contradiction"] for r in results),
        "claims": claims,
        **{f"{o}_claims": c for o, c in counts.items()},
        "errors": sum(r["error"] is not None for r in results),
        "mean_seconds": round(sum(r["seconds"] for r in results) / n, 1) if n else 0.0,
    }


async def run_eval(agent: Agent, questions: list[dict], out: Path | None = None, meta: dict | None = None) -> dict:
    results: list[dict] = []
    for i, q in enumerate(questions, 1):
        r = await run_one(agent, q)
        results.append(r)
        labels = r["verification"].get("counts", {})
        mark = "ok  " if r["correct"] else ("ERR " if r["error"] else "WRONG")
        print(f"[{i}/{len(questions)}] {mark} {q['id']:<11} {r['seconds']:6.1f}s  claims {labels}"
              + ("  UNMARKED" if r["unmarked_contradiction"] else ""), flush=True)
        report = {**(meta or {}), "summary": summarize(results), "results": results}
        if out:
            out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ids", help="comma-separated question ids")
    parser.add_argument("--limit", type=int, help="run N questions, taken in turn from each source/domain group")
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--out", type=Path, help="report path (default data/evals/run-<timestamp>.json)")
    args = parser.parse_args(argv)
    try:
        cfg = load_config()
        _, questions = load_questions(args.questions)
    except (ConfigError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    chosen = pick(questions, args.ids.split(",") if args.ids else None, args.limit)
    out = args.out or cfg.root / "data" / "evals" / f"run-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    knowledge = Knowledge.open(cfg)
    if not knowledge.tools:
        print("warning: no knowledge tools (store missing?): answers are neither grounded nor verified", file=sys.stderr)
    meta = {"model": cfg.model_backend.model_id, "backend": cfg.model_backend.api_base,
            "started": datetime.now().isoformat(timespec="seconds")}

    async def go() -> dict:
        llm = LLMClient(cfg.model_backend)
        try:
            return await run_eval(Agent(cfg, llm, knowledge), chosen, out, meta)
        finally:
            await llm.aclose()

    report = asyncio.run(go())
    s = report["summary"]
    print(f"\n{s['questions']} questions: {s['correct_pct']}% correct, {s['supported_claims_pct']}% of {s['claims']} claims"
          f" supported ({s['unverified_claims']} unverified, {s['corrected_claims']} corrected, {s['removed_claims']}"
          f" removed), {s['unmarked_contradictions']} unmarked contradictions, {s['errors']} errors,"
          f" {s['mean_seconds']} s per question")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
