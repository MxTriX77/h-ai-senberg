"""Synthetic items of the general schema for the tests (the real sets are written blind and stay out of the
repository), and runs made of them without a model."""

import copy
import json
from pathlib import Path

import yaml

REFERENCE = {"answer": "About 1,370 books.", "key_points": ["30 x 365 x 40 / 320 = 1,368.75"],
             "check": {"type": "none"}, "sources": ["computed (Python)"]}
ITEM = {"id": "reasoning-dev-001", "split": "dev", "category": "reasoning", "lang": "en", "request_type": "analytic",
        "turns": [], "prompt": "If I read 30 pages a day, how many books is that in 40 years?", "reference": REFERENCE,
        "must_not": ["A total far from 1,370 presented as the result"], "offline_base": "n/a", "probes": [],
        "notes": ""}
MERGE_TESTS = ("assert merge([[1, 3], [2, 6]]) == [[1, 6]]\n"
               "assert merge([]) == []\nassert merge([[5, 7], [1, 2]]) == [[1, 2], [5, 7]]\n")
MERGE = ("def merge(intervals):\n    out = []\n    for a, b in sorted(intervals):\n        if out and a <= out[-1][1]:\n"
         "            out[-1][1] = max(out[-1][1], b)\n        else:\n            out.append([a, b])\n    return out\n")


def item(iid: str = "reasoning-dev-001", check: dict | None = None, **over) -> dict:
    """One valid item; `check` replaces the reference's check, other keywords replace top-level fields. The
    category follows the id's first part unless given."""
    it = copy.deepcopy(ITEM) | {"id": iid, "category": iid.split("-")[0]} | over
    if check is not None:
        it["reference"] = {**it["reference"], "check": check}
    return it


def ready(it: dict) -> dict:
    """The item as the loader hands it to the runner (its set is its category, its question the prompt)."""
    return {**it, "set": it["category"], "question": it["prompt"], "file": None}


def coding_item(iid: str = "coding-dev-001", **over) -> dict:
    return item(iid, {"type": "python_tests", "entry_point": "merge", "tests": MERGE_TESTS},
                prompt="Write merge(intervals) that merges overlapping intervals.", **over)


def write_items(path: Path, *items: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"items": list(items)}, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def general_log(it: dict, answer: str = "About 1,370 books.", mode: str = "config", total: float = 2.0,
                **over) -> dict:
    """An item log as the runner writes it for an item of the general schema, without running anything."""
    log = {"id": it["id"], "set": it["category"], "mode": mode,
           "messages": [*it["turns"], {"role": "user", "content": it["prompt"]}], "draft": answer, "answer": answer,
           "reasoning": "SECRET-TRACE", "verification": {}, "timings": {"total_s": total}, "model_calls": 1,
           "chars": {}, "error": None, "correct": None, "scores": None, "artefacts": [],
           "unmarked_contradiction": False, "finish_reason": "stop", "stages": [],
           **{k: it[k] for k in ("category", "request_type", "lang", "probes", "offline_base", "split")},
           "science_calls": 0, "drift_auto": None if it["category"] == "science" else False,
           "budget_s": None, "within_budget": None, "check": None}
    return log | over


def write_general_run(folder: Path, mode: str, logs: list[dict], eval_file: Path, write_summary) -> Path:
    """A run folder with these logs, a manifest naming the eval file, and its summary."""
    (folder / "items").mkdir(parents=True)
    for r in logs:
        (folder / "items" / f"{r['id']}.json").write_text(json.dumps(r), encoding="utf-8")
    man = {"run_id": folder.name, "mode": mode, "model_id": "secret-model",
           "eval_files": [{"path": str(eval_file), "sha256": None}],
           "evals": {"science_categories": ["science"]}}
    (folder / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    write_summary(folder, mode)
    return folder
