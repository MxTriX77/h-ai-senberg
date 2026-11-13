"""Blind review of eval answers: pack anonymised answers for reviewers, merge their reviews into the runs, and
compare the arms.

    uv run python -m evals.review pack --runs [ARM=]RUN_ID... --out DIR [--seed N] [--parts N]
    uv run python -m evals.review merge DIR
    uv run python -m evals.review stats DIR [--base ARM]

`pack` gives every answer of the runs (items that failed are left out) a blind id R001, ... and writes the pack:
blind id, prompt, reference, final answer; nothing about run, arm, mode, model or trace. The mapping (blind id ->
run and item) is not for the reviewer. Runs of items of the general schema give a pack of format 2: all answers
of one item go to the same part (`--parts`: one reviewer grades every arm of an item), shuffled inside it; a run
named ARM=RUN belongs to that arm, and the runs of one arm are its repeats; the reviewer's brief
(evals/reviewer-brief.md) is copied into the pack, and the mapping is written next to the pack folder
(DIR.mapping.json), so the folder handed to reviewers holds nothing that unblinds them. Runs of the older sets
give the earlier pack (pack.md, key, reviews.yaml, DIR/mapping.json; README "Evals"). `merge` validates the
reviews, writes reviews.json (item id -> review) into each run folder and adds the review metrics to its
summary.json. `stats` prints, for a format 2 pack, the review metrics per arm and the paired comparison of every
arm with the base arm, item by item, in total and by category, request type, language and probe.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

from evals.items import IN_BASE
from evals.run import REVIEW_COUNTS, key_of, read_logs, run_dir, run_items, write_summary

LISTS = ("defects_found", "defects_missed", "omissions", "must_not_violated", "new_errors", "falsehoods")
BRIEF = Path(__file__).resolve().parent / "reviewer-brief.md"  # the reviewer's brief of format 2, copied into a pack
KINDS = ("false_fact", "invented", "invalid_inference", "self_contradiction")  # what a confident falsehood is
NO_ANSWER = "(no answer)"  # shown for an answer the model did not give, in every arm alike


def mapping_path(folder: Path) -> Path:
    """Where the mapping of a format 2 pack is written: next to the pack folder (DIR.mapping.json), not in it."""
    folder = folder.resolve()
    return folder.parent / f"{folder.name}.mapping.json"


def mapping_file(folder: Path) -> Path:
    """The mapping of a pack: next to the folder (format 2), else inside it (the older format)."""
    path = mapping_path(folder)
    return path if path.is_file() else folder / "mapping.json"


def _prompt(messages: list[dict[str, str]]) -> str:
    *turns, last = messages
    return "".join(f"**{m['role']}:** {m['content']}\n\n" for m in turns) + last["content"]


def pack(runs: list[Path], out: Path, seed: int | None = None, parts: int = 1,
         arms: list[str | None] | None = None) -> int:
    """Write the pack and its mapping; returns the number of answers. Runs of items of the general schema give
    a pack of format 2 (`parts` part files, `arms`: an arm name per run, None for a run that is its own arm; the
    mapping next to the folder, see mapping_path); runs of the older sets give pack.md and mapping.json in the
    folder as before. A mix of the two is refused (ValueError)."""
    seed = random.randrange(2**32) if seed is None else seed
    entries = []
    for folder in runs:
        items = run_items(folder)
        entries += [(folder.name, r, items[r["id"]]) for r in read_logs(folder) if r["error"] is None]
    general = ["category" in item for _, _, item in entries]
    if any(general):
        if not all(general):
            raise ValueError("the runs mix items of the general schema with older sets: pack them apart")
        return _pack_general(runs, arms or [None] * len(runs), entries, out, seed, parts)
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


def _check_line(check: dict[str, Any], log: dict[str, Any]) -> str:
    """What a block says about the item's check: the reference value of an exact or numeric check (a hint for the
    reviewer, who reads the answer), or how the code check of this answer ended."""
    kind, value = check.get("type"), check.get("value")
    shown = ", ".join(str(v) for v in value) if isinstance(value, list) else str(value)
    if kind == "exact":
        return f"Reference value: {shown}\n"
    if kind == "numeric":
        return f"Reference value: {shown} (tolerance {check['tolerance']})\n"
    if kind == "python_tests" and log.get("check"):
        return f"Code check (the item's tests run against this answer's code): {log['check']['final']['status']}\n"
    return ""


def _block(blind: str, log: dict[str, Any], item: dict[str, Any]) -> str:
    """One answer of a format 2 pack: the item's attributes, the conversation, the reference with its hints and
    the answer. Nothing about the run, and no item id."""
    ref, (*turns, last) = item["reference"], log["messages"]
    head = (f"category: {item['category']} | request type: {item['request_type']} | language: {item['lang']}"
            f" | offline base: {item['offline_base']} | probes: {', '.join(item['probes']) or 'none'}")
    earlier = "".join(f"**{m['role']}:** {m['content']}\n\n" for m in turns)
    earlier = f"### Earlier turns\n\n{earlier}" if turns else ""
    points = "".join(f"- {p}\n" for p in ref["key_points"])
    sources = "".join(f"- {s}\n" for s in ref["sources"])
    hints = "".join(f"- {m}\n" for m in item["must_not"]) or "- (none listed)\n"
    answer = log["answer"] if log["draft"].strip() else NO_ANSWER
    return (f"---\n\n## {blind}\n\n{head}\n\n{earlier}### Request\n\n{last['content']}\n\n"
            f"### Reference\n\n{ref['answer'].strip()}\n\n"
            + (f"Key points:\n{points}\n" if points else "") + _check_line(ref["check"], log)
            + (f"Sources:\n{sources}" if sources else "")
            + (f"\nNotes: {item['notes'].strip()}\n" if item["notes"].strip() else "")
            + f"\n### Watch for (hints, not a checklist)\n\n{hints}\n### Answer\n\n{answer}\n")


def _pack_general(runs: list[Path], arms: list[str | None], entries: list[tuple[str, dict, dict]], out: Path,
                  seed: int, parts: int) -> int:
    """Write a pack of format 2: every answer of one item in the same part, shuffled inside the part."""
    if parts < 1:
        raise ValueError("--parts must be at least 1")
    rng = random.Random(seed)
    item_ids = sorted({log["id"] for _, log, _ in entries})
    rng.shuffle(item_ids)
    part_of = {iid: n % parts for n, iid in enumerate(item_ids)}
    repeats: dict[str, int] = {}
    mapping: dict[str, Any] = {"format": 2, "seed": seed, "parts": parts, "runs": {}, "answers": {}, "items": {}}
    science: dict[str, set[str]] = {}
    for folder, arm in zip(runs, arms):
        arm = arm or folder.name
        repeats[arm] = repeats.get(arm, 0) + 1
        mapping["runs"][folder.name] = {"path": str(folder.resolve()), "arm": arm, "repeat": repeats[arm]}
        man = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        science[folder.name] = set((man.get("evals") or {}).get("science_categories") or [])
    for run, _, item in entries:
        mapping["items"].setdefault(item["id"], {
            "category": item["category"], "request_type": item["request_type"], "lang": item["lang"],
            "probes": item["probes"], "offline_base": item["offline_base"],
            "science": item["category"] in science[run]})
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for part in range(parts):
        mine = [e for e in entries if part_of[e[1]["id"]] == part]
        rng.shuffle(mine)
        reviews = "reviews.yaml" if parts == 1 else f"reviews-part{part + 1:02d}.yaml"
        blocks = [f"# Review pack, part {part + 1} of {parts}\n\n{len(mine)} answers. Read brief.md first; write one"
                  f" block per answer to {reviews}.\n"]
        for run, log, item in mine:
            n += 1
            blind = f"R{n:03d}"
            mapping["answers"][blind] = {"run": run, "item": log["id"], "part": part + 1}
            blocks.append(_block(blind, log, item))
        name = "pack.md" if parts == 1 else f"part{part + 1:02d}.md"
        (out / name).write_text("\n".join(blocks), encoding="utf-8")
    shutil.copyfile(BRIEF, out / "brief.md")
    mapping_path(out).write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    return n


def read_reviews(folder: Path) -> Any:
    """The review blocks of every reviews*.yaml of a pack folder, as one list (a file that is not a list is
    handed on as it is, for the checker to name)."""
    blocks: list[Any] = []
    for path in sorted(folder.glob("reviews*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return data
        blocks += data
    return blocks


def check_reviews_general(reviews: Any, mapping: dict[str, Any]) -> list[str]:
    """What is wrong with the reviews of a format 2 pack, one line per problem (empty when they are valid)."""
    if not isinstance(reviews, list):
        return ["a reviews file must be a list of review blocks (- blind_id: R001 ...)"]
    answers, items = mapping["answers"], mapping["items"]
    problems, seen = [], set()
    for n, rv in enumerate(reviews, 1):
        blind = rv.get("blind_id") if isinstance(rv, dict) else None
        if blind not in answers:
            problems.append(f"block {n}: unknown or missing blind_id {blind!r}")
            continue
        if blind in seen:
            problems.append(f"{blind}: reviewed twice")
        seen.add(blind)
        item, score = items[answers[blind]["item"]], rv.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 1 <= score <= 10:
            problems.append(f"{blind}: score must be a number from 1 to 10, got {score!r}")
        lies = rv.get("falsehoods")
        if not isinstance(lies, list) or not all(
                isinstance(e, dict) and e.get("kind") in KINDS and isinstance(e.get("central"), bool)
                and isinstance(e.get("what"), str) and e["what"].strip() for e in lies):
            problems.append(f"{blind}: falsehoods must be a list of {{kind: {'|'.join(KINDS)}, central: true|false,"
                            " what: one line}")
        elif rv.get("confident_falsehoods") != len(lies) or isinstance(rv.get("confident_falsehoods"), bool):
            problems.append(f"{blind}: confident_falsehoods ({rv.get('confident_falsehoods')!r}) must equal the"
                            f" number of falsehoods listed ({len(lies)})")
        doubts = rv.get("doubts")
        if not isinstance(doubts, list) or not all(isinstance(d, str) and d.strip() for d in doubts):
            problems.append(f"{blind}: doubts must be a list of one-line texts")
        elif rv.get("needless_doubts") != len(doubts) or isinstance(rv.get("needless_doubts"), bool):
            problems.append(f"{blind}: needless_doubts ({rv.get('needless_doubts')!r}) must equal the number of"
                            f" doubts listed ({len(doubts)})")
        for field in ("domain_drift", "needless_idk"):
            if not isinstance(rv.get(field), bool):
                problems.append(f"{blind}: {field} must be true or false")
        if rv.get("domain_drift") is True and item["science"]:
            problems.append(f"{blind}: domain_drift cannot be true: the item's category ({item['category']}) is"
                            " science")
        if rv.get("needless_idk") is True and item["offline_base"] not in IN_BASE:
            problems.append(f"{blind}: needless_idk cannot be true: the item's offline base is"
                            f" {item['offline_base']!r}, not {' or '.join(IN_BASE)}")
        if not isinstance(rv.get("notes", ""), str):
            problems.append(f"{blind}: notes must be a text")
    if missing := sorted(set(answers) - seen):
        problems.append(f"no review for {', '.join(missing)}")
    return problems


def _valid_reviews(folder: Path, mapping: dict[str, Any]) -> list[dict[str, Any]]:
    """The reviews of a format 2 pack; raises ValueError listing every problem."""
    reviews = read_reviews(folder)
    if problems := check_reviews_general(reviews, mapping):
        raise ValueError(f"{folder / 'reviews*.yaml'}:\n" + "\n".join(f"- {p}" for p in problems))
    return reviews


def _merge_general(folder: Path, mapping: dict[str, Any]) -> list[Path]:
    reviews, answers = _valid_reviews(folder, mapping), mapping["answers"]
    runs = {name: Path(run["path"]) for name, run in mapping["runs"].items()}
    for name, path in runs.items():
        mine = {answers[rv["blind_id"]]["item"]: {**rv, "format": 2} for rv in reviews
                if answers[rv["blind_id"]]["run"] == name}
        (path / "reviews.json").write_text(json.dumps(mine, ensure_ascii=False, indent=2), encoding="utf-8")
        write_summary(path, json.loads((path / "manifest.json").read_text(encoding="utf-8"))["mode"])
    return list(runs.values())


def sign_test(better: int, worse: int) -> float | None:
    """Two-sided sign test: the chance of a split at least this uneven when better and worse are equally likely;
    None without any."""
    n, k = better + worse, min(better, worse)
    return None if not n else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def _f(value: float | None) -> str:
    return "–" if value is None else f"{value:.2f}".rstrip("0").rstrip(".")


def _count(units: list[dict[str, Any]], field: str, keep: list[bool]) -> str:
    """`field` true among the units that `keep` says can be judged on it, as "n of m"; "–" when none can."""
    judged = [u for u, ok in zip(units, keep) if ok]
    return f"{sum(bool(u[field]) for u in judged)} of {len(judged)}" if judged else "–"


def _lies(review: dict[str, Any]) -> int:
    return len(review["falsehoods"])


def _score(review: dict[str, Any]) -> float:
    return review["score"]


def _sides(base: list[float], arm: list[float]) -> tuple[int, int, int]:
    """How many items the arm has below the base, above it and level with it."""
    below, above = sum(y < x for x, y in zip(base, arm)), sum(y > x for x, y in zip(base, arm))
    return below, above, len(base) - below - above


def stats(folder: Path, base: str | None = None) -> str:
    """The review metrics of a format 2 pack as markdown: one row per arm, in the order of the pack's runs, and
    for every other arm the paired comparison with the base arm (given, or the first), on the items both have, in
    total and by category, request type, language and probe. An item is one pair: each arm's falsehoods and score
    on it are averaged over its repeats, and the items with fewer and more falsehoods, and with a higher and lower
    score, get a sign test each (repeats of an item are not independent). Raises ValueError when the reviews are
    not valid."""
    mapping = json.loads(mapping_file(folder).read_text(encoding="utf-8"))
    if mapping.get("format") != 2:
        raise ValueError(f"{folder}: stats needs a pack of items of the general schema (format 2)")
    by_blind = {rv["blind_id"]: rv for rv in _valid_reviews(folder, mapping)}
    items, arms = mapping["items"], {}
    for blind, a in mapping["answers"].items():
        run = mapping["runs"][a["run"]]
        arms.setdefault(run["arm"], {})[a["item"], run["repeat"]] = by_blind[blind]
    order = dict.fromkeys(run["arm"] for run in mapping["runs"].values())  # the order of --runs, not the blind ids
    arms = {arm: arms[arm] for arm in order if arm in arms}
    if base is not None and base not in arms:
        raise ValueError(f"no arm {base!r} in the pack; arms: {', '.join(arms)}")
    base = base or next(iter(arms))

    def judged(keys: list[tuple[str, int]]) -> tuple[list[bool], list[bool]]:
        return ([not items[i]["science"] for i, _ in keys], [items[i]["offline_base"] in IN_BASE for i, _ in keys])

    def means(units: dict[tuple[str, int], dict[str, Any]], ids: list[str], value) -> list[float]:
        """Per item, the mean of `value` over the arm's answers to it (its repeats)."""
        per_item: dict[str, list[float]] = {}
        for (i, _), u in units.items():
            per_item.setdefault(i, []).append(value(u))
        return [sum(per_item[i]) / len(per_item[i]) for i in ids]

    lines = ["| arm | answers | confident falsehoods | per answer | central | mean score | domain drift"
             " | needless \"I don't know\" | needless doubts |", "|---|---|---|---|---|---|---|---|---|"]
    for arm, units in arms.items():
        keys, rows = list(units), list(units.values())
        outside, in_base = judged(keys)
        lies = sum(len(u["falsehoods"]) for u in rows)
        lines.append(f"| {arm} | {len(rows)} | {lies} | {_f(lies / len(rows))}"
                     f" | {sum(e['central'] for u in rows for e in u['falsehoods'])}"
                     f" | {_f(sum(u['score'] for u in rows) / len(rows))} | {_count(rows, 'domain_drift', outside)}"
                     f" | {_count(rows, 'needless_idk', in_base)} | {sum(len(u['doubts']) for u in rows)} |")
    groups = [("category", lambda it: [it["category"]]), ("request type", lambda it: [it["request_type"]]),
              ("language", lambda it: [it["lang"]]), ("probe", lambda it: it["probes"])]
    for arm, units in arms.items():
        if arm == base:
            continue
        mine = {i for i, _ in units}
        both = list(dict.fromkeys(i for i, _ in arms[base] if i in mine))  # the items both arms answered
        if not both:
            lines += ["", f"Paired: {base} → {arm}: no item is in both arms"]
            continue
        lines += ["", f"Paired: {base} → {arm}, {len(both)} items (an item is one pair: each arm's falsehoods and"
                      " score on it averaged over its repeats)", "",
                  "| group | items | answers | confident falsehoods | fewer / more / tied | sign test p | mean score"
                  " | higher / lower / tied | sign test p | domain drift | needless \"I don't know\""
                  " | needless doubts |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        rows = [("all", both)]
        for label, values in groups:
            names = sorted({v for i in both for v in values(items[i])})
            rows += [(f"{label}: {name}", [i for i in both if name in values(items[i])]) for name in names]
        for label, ids in rows:
            keep = set(ids)
            ka, kb = [k for k in arms[base] if k[0] in keep], [k for k in units if k[0] in keep]
            a, b = [arms[base][k] for k in ka], [units[k] for k in kb]
            fewer, more, tied = _sides(means(arms[base], ids, _lies), means(units, ids, _lies))
            lower, higher, level = _sides(means(arms[base], ids, _score), means(units, ids, _score))
            (outside_a, in_base_a), (outside_b, in_base_b) = judged(ka), judged(kb)
            lines.append(
                f"| {label} | {len(ids)} | {len(a)} → {len(b)} | {sum(map(_lies, a))} → {sum(map(_lies, b))}"
                f" | {fewer} / {more} / {tied} | {_f(sign_test(fewer, more))}"
                f" | {_f(sum(map(_score, a)) / len(a))} → {_f(sum(map(_score, b)) / len(b))}"
                f" | {higher} / {lower} / {level} | {_f(sign_test(higher, lower))}"
                f" | {_count(a, 'domain_drift', outside_a)} → {_count(b, 'domain_drift', outside_b)}"
                f" | {_count(a, 'needless_idk', in_base_a)} → {_count(b, 'needless_idk', in_base_b)}"
                f" | {sum(len(u['doubts']) for u in a)} → {sum(len(u['doubts']) for u in b)} |")
    return "\n".join(lines) + "\n"


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
        listed = rv.get("falsehoods", [])
        if isinstance(listed, list) and not all(isinstance(e, dict) and e.get("tag") for e in listed):
            problems.append(f"{blind}: each falsehoods entry needs a tag, e.g. {{tag: F12, central: true, what: ...}}")
        elif isinstance(listed, list) and "confident_falsehoods" in rv and rv["confident_falsehoods"] != len(listed):
            problems.append(f"{blind}: confident_falsehoods ({rv['confident_falsehoods']}) must equal the number of"
                            f" falsehoods listed ({len(listed)})")
    if missing := sorted(set(keys) - seen):
        problems.append(f"no review for {', '.join(missing)}")
    return problems


def merge(folder: Path) -> list[Path]:
    """Validate the reviews of a pack (format 2: every reviews*.yaml; else reviews.yaml) and merge them into
    every run of the pack; returns the run folders."""
    mapping = json.loads(mapping_file(folder).read_text(encoding="utf-8"))
    if mapping.get("format") == 2:
        return _merge_general(folder, mapping)
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
    p = sub.add_parser("pack", help="write the pack for blind reviewers and its mapping")
    p.add_argument("--runs", nargs="+", required=True, metavar="[ARM=]RUN",
                   help="run ids (folders under ${DATA_DIR:-data}/evals) or run folders; ARM=RUN puts a run into an"
                        " arm (general items: the runs of one arm are its repeats, in this order)")
    p.add_argument("--out", type=Path, required=True, help="pack folder")
    p.add_argument("--seed", type=int, help="shuffle seed (default: random; kept in the mapping)")
    p.add_argument("--parts", type=int, default=1, help="general items: part files, one per reviewer (default 1);"
                                                        " every answer of an item is in the same part")
    m = sub.add_parser("merge", help="validate the reviews in DIR and merge them into the runs")
    m.add_argument("dir", type=Path)
    s = sub.add_parser("stats", help="general items: review metrics per arm and paired against the base arm")
    s.add_argument("dir", type=Path)
    s.add_argument("--base", help="the arm the others are compared with (default: the first in the pack's --runs)")
    args = parser.parse_args(argv)
    try:
        if args.command == "pack":
            named = [r.partition("=")[::2] if "=" in r else ("", r) for r in args.runs]
            n = pack([run_dir(run) for _, run in named], args.out, args.seed, args.parts,
                     [arm or None for arm, _ in named])
            print(f"{n} answers in {args.out}; the mapping, not for the reviewer: {mapping_file(args.out)}")
        elif args.command == "stats":
            print(stats(args.dir, args.base), end="")
        else:
            for path in merge(args.dir):
                print(f"reviews merged into {path}")
    except (OSError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
