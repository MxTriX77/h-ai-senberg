"""Run the eval sets through the real pipeline (tools, tool loop, verifier) and score them.

    uv run python -m evals.run [--set facts|reasoning|all] [--holdout] [--limit N] [--ids ID,ID]
                               [--questions FILE] [--problems FILE] [--out FILE]

Two sets: fact questions (evals/questions.yaml) and multi-step reasoning problems
(evals/reasoning.yaml, scored on their labelled answer lines; held-out problems run only with
--holdout). The backend, store and settings come from the config and environment, exactly as for the
chat (`API_BASE_URL`, `DATA_DIR`, `MODEL_PROFILE`, ...). The JSON report is rewritten after every item.
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
PROBLEMS = Path(__file__).resolve().parent / "reasoning.yaml"
KINDS = ("number", "formula", "text")
PART_KINDS = ("number", "text", "verdict", "ids")
CATEGORIES = ("facts", "reasoning")
SETS = ("facts", "reasoning", "all")
_LINE = re.compile(r"^[\s>*_`#+-]*(A\d+)[\s*_`]*[:：][\s*_`]*(.*)$", re.I | re.M)


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
        q["category"] = "facts"
    return sources, questions


def load_problems(path: Path = PROBLEMS) -> list[dict[str, Any]]:
    """The reasoning problems; raises ValueError naming the bad problem and part."""
    problems = yaml.safe_load(path.read_text(encoding="utf-8"))["problems"]
    seen: set[str] = set()
    for p in problems:
        pid = p.get("id")
        if not pid or pid in seen:
            raise ValueError(f"{path}: missing or duplicate id {pid!r}")
        seen.add(pid)
        if p.get("domain") not in ("chemistry", "physics") or not p.get("question") or not p.get("topic") \
                or p.get("lang") not in ("en", "ru") or not p.get("parts"):
            raise ValueError(f"{path}: problem {pid}: needs a domain, a topic, lang en|ru, the question and parts")
        labels: set[str] = set()
        for part in p["parts"]:
            label, kinds = part.get("label"), [k for k in PART_KINDS if k in part]
            where = f"{path}: problem {pid}: part {label}"
            if not label or label in labels or len(kinds) != 1:
                raise ValueError(f"{where}: needs a unique label and exactly one of {PART_KINDS}")
            labels.add(label)
            if kinds[0] == "number" and ("abs_tol" in part) == ("rel_tol" in part):
                raise ValueError(f"{where}: a number needs abs_tol or rel_tol")
            if kinds[0] != "number" and not (isinstance(part[kinds[0]], list) and part[kinds[0]]):
                raise ValueError(f"{where}: {kinds[0]} must be a non-empty list")
        p["category"] = "reasoning"
    return problems


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


def answer_lines(answer: str) -> dict[str, str]:
    """Label (A1, A2, ...) -> the text of the last line that starts with that label and a colon."""
    return {m.group(1).upper(): m.group(2).strip() for m in _LINE.finditer(answer_text(answer))}


def part_correct(part: dict[str, Any], line: str | None, lang: str = "en") -> bool:
    if line is None:
        return False
    line = re.sub(r"[*_`]", "", line).strip()
    if "number" in part:
        # Decimal comma: always in Russian; in English only when not exactly three digits follow (1,900).
        comma = r"(\d),(\d)" if lang == "ru" else r"(\d),(\d{1,2}(?!\d)|\d{4,})"
        numbers = read_numbers(re.sub(comma, r"\1.\2", line))
        value = float(part["number"])
        tol = part["abs_tol"] if "abs_tol" in part else part["rel_tol"] * abs(value)
        return bool(numbers) and abs(numbers[0].value - value) <= tol + 1e-12 * abs(value)
    if "text" in part:
        return any(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", line, re.I) for w in part["text"])
    if "verdict" in part:
        start = re.sub(r"^(?:the|a|an)\s+", "", re.sub(r"^\W+", "", line.lower()))
        return any(re.match(rf"{re.escape(str(w).lower())}(?!\w)", start) for w in part["verdict"])
    return {i.upper() for i in re.findall(r"(?<!\w)S\d+(?!\w)", line, re.I)} == {str(i).upper() for i in part["ids"]}


def score_parts(answer: str, problem: dict[str, Any]) -> list[bool]:
    lines = answer_lines(answer)
    return [part_correct(p, lines.get(p["label"].upper()), problem["lang"]) for p in problem["parts"]]


def select(facts: list[dict], problems: list[dict], which: str = "all", holdout: bool = False) -> list[dict]:
    """The items of a run: held-out problems only with `holdout` (and then nothing else)."""
    if holdout:
        return [p for p in problems if p.get("holdout")]
    return (facts if which in ("facts", "all") else []) + \
           ([p for p in problems if not p.get("holdout")] if which in ("reasoning", "all") else [])


def pick(questions: list[dict], ids: list[str] | None = None, limit: int | None = None) -> list[dict]:
    """By id; or `limit` items shared out evenly between the groups (source/domain for facts,
    reasoning/domain for problems) and spread evenly over each group (a small run still covers every kind)."""
    if ids:
        wanted = set(ids)
        return [q for q in questions if q["id"] in wanted]
    if not limit or limit >= len(questions):
        return questions
    groups: dict[str, list[dict]] = {}
    for q in questions:
        groups.setdefault(f"{q.get('source', 'reasoning')}/{q['domain']}", []).append(q)
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
    category = q.get("category", "facts")
    extra: dict[str, Any] = {}
    if category == "reasoning":
        parts = score_parts(answer, q) if error is None else [False] * len(q["parts"])
        correct, extra["parts"] = all(parts), {p["label"]: ok for p, ok in zip(q["parts"], parts)}
    else:
        correct = error is None and is_correct(answer, q["answer"])
    all_supported = bool(claims) and all(c["outcome"] == "supported" for c in claims)
    return {
        "id": q["id"],
        "category": category,
        "question": q["question"],
        "known": q["answer"] if category == "facts" else q["parts"],
        "answer": answer,
        "correct": correct,
        **extra,
        "unmarked_contradiction": error is None and not correct and all_supported,
        "verification": report,
        "seconds": round(time.monotonic() - t0, 1),
        "error": error,
    }


def summarize(results: list[dict[str, Any]], by_category: bool = True) -> dict[str, Any]:
    n = len(results)
    counts = {o: sum(r["verification"].get("counts", {}).get(o, 0) for r in results)
              for o in ("supported", "unverified", "corrected", "removed")}
    claims = sum(counts.values())
    summary: dict[str, Any] = {
        "questions": n,
        "correct_pct": round(100 * sum(r["correct"] for r in results) / n, 1) if n else 0.0,
        "supported_claims_pct": round(100 * counts["supported"] / claims, 1) if claims else 0.0,
        "unmarked_contradictions": sum(r["unmarked_contradiction"] for r in results),
        "claims": claims,
        **{f"{o}_claims": c for o, c in counts.items()},
        "consistency_fixes": sum(len(r["verification"].get("consistency", [])) for r in results),
        "arithmetic_fixes": sum(len(r["verification"].get("arithmetic", [])) for r in results),
        "errors": sum(r["error"] is not None for r in results),
        "mean_seconds": round(sum(r["seconds"] for r in results) / n, 1) if n else 0.0,
    }
    parts = [ok for r in results for ok in r.get("parts", {}).values()]
    if parts:
        summary["parts_pct"] = round(100 * sum(parts) / len(parts), 1)
    if by_category:
        cats = {c: [r for r in results if r.get("category", "facts") == c] for c in CATEGORIES}
        summary["by_category"] = {c: summarize(rs, by_category=False) for c, rs in cats.items() if rs}
    return summary


async def run_eval(agent: Agent, questions: list[dict], out: Path | None = None, meta: dict | None = None) -> dict:
    results: list[dict] = []
    for i, q in enumerate(questions, 1):
        r = await run_one(agent, q)
        results.append(r)
        labels = r["verification"].get("counts", {})
        mark = "ok  " if r["correct"] else ("ERR " if r["error"] else "WRONG")
        parts = " parts " + "".join("+" if ok else "-" for ok in r["parts"].values()) if "parts" in r else ""
        print(f"[{i}/{len(questions)}] {mark} {q['id']:<16} {r['seconds']:6.1f}s{parts}  claims {labels}"
              + ("  UNMARKED" if r["unmarked_contradiction"] else ""), flush=True)
        report = {**(meta or {}), "summary": summarize(results), "results": results}
        if out:
            out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", choices=SETS, default="all", help="facts, reasoning or all (default)")
    parser.add_argument("--holdout", action="store_true", help="run only the held-out reasoning problems")
    parser.add_argument("--ids", help="comma-separated question or problem ids")
    parser.add_argument("--limit", type=int, help="run N items, taken in turn from each group")
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--problems", type=Path, default=PROBLEMS)
    parser.add_argument("--out", type=Path, help="report path (default data/evals/run-<timestamp>.json)")
    args = parser.parse_args(argv)
    try:
        cfg = load_config()
        _, facts = load_questions(args.questions)
        problems = load_problems(args.problems)
    except (ConfigError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    items = select(facts, problems, args.set, args.holdout)
    chosen = pick(items, args.ids.split(",") if args.ids else None, args.limit)
    out = args.out or cfg.root / "data" / "evals" / f"run-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    knowledge = Knowledge.open(cfg)
    if not knowledge.tools:
        print("warning: no knowledge tools (store missing?): answers are neither grounded nor verified", file=sys.stderr)
    meta = {"model": cfg.model_backend.model_id, "backend": cfg.model_backend.api_base, "set": args.set,
            "holdout": args.holdout, "started": datetime.now().isoformat(timespec="seconds")}

    async def go() -> dict:
        llm = LLMClient(cfg.model_backend)
        try:
            return await run_eval(Agent(cfg, llm, knowledge), chosen, out, meta)
        finally:
            await llm.aclose()

    report = asyncio.run(go())
    print()
    for name, s in [("all", report["summary"]), *report["summary"].get("by_category", {}).items()]:
        parts = f" ({s['parts_pct']}% of parts)" if "parts_pct" in s else ""
        print(f"{name}: {s['questions']} items, {s['correct_pct']}% correct{parts}, {s['supported_claims_pct']}% of"
              f" {s['claims']} claims supported ({s['unverified_claims']} unverified, {s['corrected_claims']} corrected,"
              f" {s['removed_claims']} removed), {s['consistency_fixes']} consistency fixes,"
              f" {s['arithmetic_fixes']} arithmetic fixes,"
              f" {s['unmarked_contradictions']} unmarked contradictions, {s['errors']} errors,"
              f" {s['mean_seconds']} s per item")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
