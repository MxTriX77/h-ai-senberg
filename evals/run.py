"""Run the eval sets through the real pipeline (tools, tool loop, verifier) and score them.

    uv run python -m evals.run [--set facts|reasoning|all] [--holdout] [--authored DIR ...] [--split SPLIT]
                               [--sets S3,S4] [--mode A|B|C|D] [--run-id ID] [--retry-errors]
                               [--limit N] [--ids ID,ID] [--questions FILE] [--problems FILE]

Sets: S1 fact questions (evals/questions.yaml), S2 multi-step reasoning problems (evals/reasoning.yaml, scored on
their labelled answer lines; held-out problems run only with --holdout) and authored items (S3-S6, U) from batch
folders (--authored, README "Evals"). --mode switches pipeline components off in-process (evals/modes.yaml).
The backend, store and settings come from the config and environment, exactly as for the chat (`API_BASE_URL`,
`DATA_DIR`, `MODEL_PROFILE`, ...). A run is a folder ${DATA_DIR:-data}/evals/<run-id>/: manifest.json, one log
per item in items/ (written when the item is done, so an interrupted run resumes with the same --run-id) and
summary.json.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import yaml
from dotenv import load_dotenv

from haisenberg.agent import DRAFT_MARK, Agent
from haisenberg.config import PROFILE_ENV, Config, ConfigError, load_config
from haisenberg.l1.text import to_hill
from haisenberg.llm import LLMClient, UpstreamError
from haisenberg.tools import Knowledge
from haisenberg.verifier import Number, read_formulas, read_numbers

QUESTIONS = Path(__file__).resolve().parent / "questions.yaml"
PROBLEMS = Path(__file__).resolve().parent / "reasoning.yaml"
MODES = Path(__file__).resolve().parent / "modes.yaml"
ROOT = Path(__file__).resolve().parents[1]
KINDS = ("number", "formula", "text")
FACT_LABEL = "A1"  # the answer line a fact question's answer target asks for (questions.yaml answer_target)
PART_KINDS = ("number", "text", "verdict", "ids")
SETS = ("facts", "reasoning", "all")
AUTHORED_SETS = ("S3", "S4", "S5", "S6", "U")
SPLITS = ("dev", "holdout", "monitor")
OUTCOMES = ("supported", "unverified", "corrected", "removed")
REVIEW_COUNTS = ("unsupported_additions", "unsupported_as_verified", "final_vs_analysis")
NVIDIA_SMI = ["nvidia-smi", "--query-gpu=name,power.limit,power.draw,memory.total", "--format=csv,noheader"]
_LOOKALIKE = "АВСЕНКМОРТХавсенкмортх"  # Cyrillic letters a model may write in a label instead of the Latin ones
_LATIN = str.maketrans(_LOOKALIKE, "ABCEHKMOPTXabcehkmoptx")
_LINE = re.compile(rf"^[\s>*_`#+-]*([A-Za-z{_LOOKALIKE}]\d+)[\s*_`]*[:：][\s*_`]*(.*)$", re.M)
_NEXT = re.compile(rf"(?<=\s)[*_`]*([A-Za-z{_LOOKALIKE}]\d+)[\s*_`]*[:：][\s*_`]*")  # a further label on that line
# Items a task lists (`C1.`, `У3)`, `S4:` at a line start, with the rest of the line: a line whose rest is a
# `<placeholder>` is a closing answer line, not an item), an item id in a placeholder, an answer line that begins
# with an item id after markdown markers (`**C1:**`, `### C1 —`, `| C1 |`), with the rest of the line, and quoted
# text (a claim restated, never read as a verdict).
_LISTED = re.compile(r"^[\s>*_`#•-]*([A-ZА-ЯЁ]{1,2}\d+)[*_`]*[.):：][ \t*_`]*(.*)$", re.M)
_ID = re.compile(r"(?<!\w)([A-ZА-ЯЁ]{1,2}\d+)(?!\w)")
_START = re.compile(r"^[\s>*_`#|+-]*([A-ZА-ЯЁ]{1,2}\d+)(?!\w)(.*)$")
_VERDICT_LINE = re.compile(r"^[\s>*_`#|+-]*(?:verdict|status|conclusion|вердикт|статус|вывод)[\s*_`]*[:：](.*)$", re.I)
_RULE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
_QUOTED = re.compile(r'"[^"\n]*"|“[^”\n]*”|«[^»\n]*»|„[^“”\n]*[“”]')
# Protocol artefacts: text the pipeline writes for itself that must not reach the final answer. The patterns follow
# the code that writes it: [S#] ids (verifier._CITES), the draft marker (agent.DRAFT_MARK), the trace headings of
# Verifier.verify and Verifier._consistency, the summary line and outcome labels of verifier.render and the
# claim lines of the trace; and raw tool-call markup of the chat template that leaked into the answer text.
ARTEFACTS = {
    "source_id": re.compile(r"\[S\d+(?:\s*[,;]\s*S?\d+)*\]"),
    "draft_marker": re.compile(re.escape(DRAFT_MARK.strip())),
    "heading": re.compile(r"Verification: (?:\d+ claims?|no checkable claims)|Consistency review(?: skipped| failed)?:"
                          r"|Arithmetic: \d+ calculations? recomputed|Check [\w-]+: \d+ findings?"),
    "outcome_label": re.compile(r"\*\((?:unverified|corrected from [^)]*)\)\*"
                                r"|\bC\d+ (?:supported|unverified|corrected|removed|judge:)"),
    "tool_call": re.compile(r"<tool_call>|<function=|<parameter="),
}


def load_questions(path: Path = QUESTIONS) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """(source key -> license note, questions, each with the file's answer target appended); raises ValueError
    naming the bad question."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    sources, questions, target = data["sources"], data["questions"], data.get("answer_target") or ""
    if f"{FACT_LABEL}:" not in target:
        raise ValueError(f"{path}: needs an answer_target asking for the {FACT_LABEL}: line")
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
        q.update(set="S1", file=path, question=f"{q['question']}\n\n{target}")
    return sources, questions


def check_parts(parts: list[dict[str, Any]], where: str) -> None:
    """Raises ValueError naming the bad part: a unique label and exactly one well-formed kind each."""
    labels: set[str] = set()
    for part in parts:
        label, kinds = part.get("label"), [k for k in PART_KINDS if k in part]
        here = f"{where}: part {label}"
        if not label or label in labels or len(kinds) != 1:
            raise ValueError(f"{here}: needs a unique label and exactly one of {PART_KINDS}")
        labels.add(label)
        if kinds[0] == "number" and ("abs_tol" in part) == ("rel_tol" in part):
            raise ValueError(f"{here}: a number needs abs_tol or rel_tol")
        if kinds[0] != "number" and not (isinstance(part[kinds[0]], list) and part[kinds[0]]):
            raise ValueError(f"{here}: {kinds[0]} must be a non-empty list")


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
        check_parts(p["parts"], f"{path}: problem {pid}")
        p.update(set="S2", file=path)
    return problems


def load_items(path: Path) -> list[dict[str, Any]]:
    """The authored items of one file (README "Evals"); raises ValueError naming the bad item."""
    items = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("items") or []
    seen: set[str] = set()
    for it in items:
        iid = it.get("id")
        if not iid or iid in seen:
            raise ValueError(f"{path}: missing or duplicate id {iid!r}")
        seen.add(iid)
        where = f"{path}: item {iid}"
        if it.get("set") not in AUTHORED_SETS or it.get("split") not in SPLITS or not it.get("prompt") \
                or it.get("lang", "en") not in ("en", "ru"):
            raise ValueError(f"{where}: needs set {'|'.join(AUTHORED_SETS)}, split {'|'.join(SPLITS)}, lang en|ru"
                             " and the prompt")
        turns = it.get("turns") or []
        if not all(isinstance(t, dict) and t.get("role") in ("user", "assistant") and isinstance(t.get("content"), str)
                   for t in turns):
            raise ValueError(f"{where}: turns must be a list of {{role: user|assistant, content: text}}")
        key = it.get("key") or {}
        check_parts(key.get("parts") or [], where)
        conv = key.get("conversation")
        if conv is not None and not all(isinstance(conv.get(k), (int, float)) for k in ("max_chars", "max_latency_s")):
            raise ValueError(f"{where}: key.conversation needs max_chars and max_latency_s")
        it.update(turns=turns, question=it["prompt"], parts=key.get("parts") or [], lang=it.get("lang", "en"), file=path)
    return items


def load_authored(folder: Path) -> list[dict[str, Any]]:
    """Every item of every *.yaml in a batch folder; raises ValueError on a bad item or a duplicate id."""
    files = sorted(folder.glob("*.yaml"))
    if not files:
        raise ValueError(f"{folder}: no *.yaml item files")
    items = [it for f in files for it in load_items(f)]
    if dupes := sorted(i for i, n in Counter(it["id"] for it in items).items() if n > 1):
        raise ValueError(f"{folder}: duplicate ids {dupes}")
    return items


def load_file(path: Path) -> list[dict[str, Any]]:
    """The items of any eval file: fact questions, reasoning problems or authored items."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "questions" in data:
        return load_questions(path)[1]
    return load_problems(path) if "problems" in data else load_items(path)


def key_of(item: dict[str, Any]) -> dict[str, Any]:
    """What the answer is graded against: an authored item's key, a question's answer or a problem's parts."""
    return item.get("key") or {k: item[k] for k in ("answer", "parts") if k in item}


def answer_text(answer: str) -> str:
    """The answer without its source list, verification summary and [S#] markers."""
    text = answer.split("\n\nSources:\n", 1)[0]
    text = re.sub(r"\*Verification:[^\n]*\*", "", text)
    return re.sub(r"\[S\d+(?:\s*[,;]\s*S?\d+)*\]", "", text)


# Small counts written as words, read as digits so a text key matches either way ("two up" = "2 up"). Language
# data (English, Russian with the gender and genitive forms a count takes), not a subject list.
_COUNTS = {w: str(n) for n, words in enumerate((
    "zero ноль нуль", "one один одна одно одного одной", "two два две двух", "three три трёх трех",
    "four четыре четырёх четырех", "five пять пяти", "six шесть шести", "seven семь семи", "eight восемь восьми",
    "nine девять девяти", "ten десять десяти")) for w in words.split()}
_COUNT = re.compile(rf"(?<!\w)(?:{'|'.join(_COUNTS)})(?!\w)", re.I)


def _digits(text: str) -> str:
    return _COUNT.sub(lambda m: _COUNTS[m.group().lower()], text)


def text_match(keys: list[str], text: str) -> bool:
    """An accepted word or phrase occurs in the text as a whole word, ignoring case; counts as digits or words."""
    text = _digits(text)
    return any(re.search(rf"(?<!\w){re.escape(_digits(w))}(?!\w)", text, re.I) for w in keys)


def is_correct(answer: str, gold: dict[str, Any]) -> bool:
    text = answer_text(answer)
    if "number" in gold:
        value = float(gold["number"])
        tol = gold["abs_tol"] if "abs_tol" in gold else gold["rel_tol"] * abs(value)
        return any(abs(n.value - value) <= tol + 1e-12 * abs(value) for n in read_numbers(text))
    if "formula" in gold:
        want = to_hill(gold["formula"])
        return any(hill == want for hill, _, _ in read_formulas(text, need_digit=False))
    return text_match(gold["text"], text)


def answer_lines(answer: str) -> dict[str, str]:
    """Label (A1, A2, ...) -> the text of the last line that starts with that label and a colon (a Cyrillic
    look-alike letter in the label, such as А, reads as the Latin one). A line holding further labels of its
    letter after a space (`A1: 22.0 g A2: 16.8 L`) is split there: each label gets the text up to the next."""
    lines: dict[str, str] = {}
    for m in _LINE.finditer(answer_text(answer)):
        label, text, start = m.group(1).translate(_LATIN).upper(), m.group(2), 0
        for n in _NEXT.finditer(text):
            if (nxt := n.group(1).translate(_LATIN).upper())[0] == label[0]:
                lines[label] = text[start:n.start()].strip()
                label, start = nxt, n.end()
        lines[label] = text[start:].strip()
    return lines


def _figures(n: Number) -> int:
    """Significant figures the number is written with (0.024: 2; 3: 1; 1,900: 4)."""
    return math.floor(math.log10(abs(n.value))) - round(math.log10(2 * n.half)) + 1 if n.value else 0


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
        if not numbers:
            return False
        n, eps = numbers[0], 1e-12 * abs(value)
        # Within tolerance, or the key rounded to the answer's own precision when that has 2+ significant
        # figures (0.0237 written as 0.024; 2.61 written as 3 stays wrong).
        return abs(n.value - value) <= tol + eps or (_figures(n) >= 2 and abs(n.value - value) <= n.half + eps)
    if "text" in part:
        return text_match(part["text"], line)
    if "verdict" in part:
        start = re.sub(r"^(?:the|a|an)\s+", "", re.sub(r"^\W+", "", line.lower()))
        return any(re.match(rf"{re.escape(str(w).lower())}(?!\w)", start) for w in part["verdict"])
    return {i.upper() for i in re.findall(r"(?<!\w)S\d+(?!\w)", line, re.I)} == {str(i).upper() for i in part["ids"]}


def score_parts(answer: str, problem: dict[str, Any]) -> list[bool]:
    lines = answer_lines(answer)
    return [part_correct(p, lines.get(p["label"].upper()), problem["lang"]) for p in problem["parts"]]


def _item_id(text: str) -> str:
    return text.translate(_LATIN).upper()


def _sections(answer: str, ids: set[str], labels: set[str]) -> dict[str, list[tuple[bool, list[str]]]]:
    """Item id -> its sections in the answer, each (table row?, lines): the start line's text after the id, then
    the lines up to the next item start, a `---` line, a heading or a closing answer line. A row is one line. The
    answer lines at the end (blank lines between) are never part of a section, even when a label is an item id."""
    found: dict[str, list[tuple[bool, list[str]]]] = {}
    section: list[str] | None = None
    lines = answer_text(answer).splitlines()
    end = len(lines)
    while end and (not lines[end - 1].strip() or
                   ((m := _LINE.match(lines[end - 1])) is not None and _item_id(m.group(1)) in labels)):
        end -= 1
    for line in lines[:end]:
        start = _START.match(line)
        if start and (iid := _item_id(start.group(1))) in ids:
            row, section = line.lstrip().startswith("|"), [start.group(2)]
            found.setdefault(iid, []).append((row, section))
            section = None if row else section
        elif _RULE.match(line) or line.lstrip().startswith("#") or \
                ((m := _LINE.match(line)) is not None and _item_id(m.group(1)) in labels):
            section = None
        elif section is not None:
            section.append(line)
    return found


def _readings(row: bool, lines: list[str], parts: list[dict[str, Any]], lang: str) -> set[tuple[bool, ...]]:
    """The verdicts a section gives, each as its score on every verdict part: its first labelled verdict line
    (`Verdict: X`), else the text right after the id (each cell of a table row) without its quoted text (the
    claim restated). Text that starts with none of the verdict words gives none."""
    labelled = [m.group(1) for line in lines if (m := _VERDICT_LINE.match(line))]
    texts = labelled[:1] or [_QUOTED.sub("", t) for t in (lines[0].split("|") if row else lines[:1])]
    return {right for t in texts if any(right := tuple(part_correct(p, t, lang) for p in parts))}


def body_verdicts(answer: str, item: dict[str, Any]) -> dict[str, bool]:
    """Verdict part label -> right, read from the answer body, for each verdict part anchored to exactly one of
    the items the task lists (its label is that item, `C1: <Correct / Incorrect>`, and/or its placeholder names
    it, `A4: <verdict for C1>`), when that item's sections give one verdict (none, or two different ones: left
    to the answer line)."""
    prompt, verdict_parts = item.get("question") or "", [p for p in item["parts"] if "verdict" in p]
    labels = {_item_id(p["label"]) for p in item["parts"]}
    listed = {_item_id(m.group(1)) for m in _LISTED.finditer(prompt) if not m.group(2).startswith("<")}
    placeholders, sections = answer_lines(prompt), _sections(answer, listed, labels)
    found: dict[str, bool] = {}
    for n, p in enumerate(verdict_parts):
        label = _item_id(p["label"])
        named = {i for m in _ID.finditer(placeholders.get(label, "")) if (i := _item_id(m.group(1))) in listed}
        named |= {label} & listed
        if len(named) == 1:
            readings = {r for row, lines in sections.get(named.pop(), [])
                        for r in _readings(row, lines, verdict_parts, item["lang"])}
            if len(readings) == 1:
                found[p["label"]] = readings.pop()[n]
    return found


def fact_correct(line: str | None, gold: dict[str, Any]) -> bool:
    """A fact answer scored strictly on its answer line (None: no line, wrong): a number or text as a reasoning
    part, a formula as is_correct does."""
    return is_correct(line, gold) if line is not None and "formula" in gold else part_correct(gold, line)


def score(answer: str, item: dict[str, Any]) -> dict[str, Any] | None:
    """Code score of an answer: part label -> right, the labels without an answer line, the verdict parts.
    A verdict part anchored to one listed item is read from that item's section in the answer body when it gives
    one verdict (body_verdicts); `line_parts` keeps every part scored on its line and `sources` says where each
    part was read (body or line). A fact question is one part, `answer`, scored on its answer line, and also has
    `loose_correct` (any match in the answer). None when the item has no code key (graded by the reviewer only)."""
    if "answer" in item:
        line, gold = answer_lines(answer).get(FACT_LABEL), item["answer"]
        right = fact_correct(line, gold)
        return {"correct": right, "parts": {"answer": right}, "missing": ["answer"] if line is None else [],
                "verdicts": {}, "loose_correct": is_correct(answer, gold)}
    if item.get("parts"):
        lines = answer_lines(answer)
        line_parts = {p["label"]: ok for p, ok in zip(item["parts"], score_parts(answer, item))}
        body = body_verdicts(answer, item)
        parts = line_parts | body
        missing = [p["label"] for p in item["parts"] if p["label"].upper() not in lines]
        verdicts = {p["label"]: parts[p["label"]] for p in item["parts"] if "verdict" in p}
    else:
        return None
    return {"correct": all(parts.values()), "parts": parts, "line_parts": line_parts,
            "sources": {label: "body" if label in body else "line" for label in parts}, "missing": missing,
            "verdicts": verdicts}


def artefacts(text: str) -> list[str]:
    """The kinds of protocol artefacts in a text."""
    return [kind for kind, pattern in ARTEFACTS.items() if pattern.search(text)]


def select(facts: list[dict], problems: list[dict], which: str | None = "all", holdout: bool = False) -> list[dict]:
    """The items of a run: held-out problems only with `holdout` (and then nothing else)."""
    if holdout:
        return [p for p in problems if p.get("holdout")]
    return (facts if which in ("facts", "all") else []) + \
           ([p for p in problems if not p.get("holdout")] if which in ("reasoning", "all") else [])


def choose(items: list[dict], split: str | None = None, sets: list[str] | None = None) -> list[dict]:
    """Authored items of `split` only (S1/S2 have no split), then only the items of `sets`."""
    return [i for i in items if (not split or i.get("split", split) == split) and (not sets or i["set"] in sets)]


def pick(questions: list[dict], ids: list[str] | None = None, limit: int | None = None) -> list[dict]:
    """By id; or `limit` items shared out evenly between the groups (source/domain for facts, set/domain for
    the others) and spread evenly over each group (a small run still covers every kind)."""
    if ids:
        wanted = set(ids)
        return [q for q in questions if q["id"] in wanted]
    if not limit or limit >= len(questions):
        return questions
    groups: dict[str, list[dict]] = {}
    for q in questions:
        groups.setdefault(f"{q.get('source') or q.get('set')}/{q.get('domain')}", []).append(q)
    quota = dict.fromkeys(groups, 0)
    while sum(quota.values()) < limit:
        for g in groups:
            if quota[g] < len(groups[g]) and sum(quota.values()) < limit:
                quota[g] += 1
    chosen = {id(groups[g][i * len(groups[g]) // n]) for g, n in quota.items() for i in range(n)}
    return [q for q in questions if id(q) in chosen]


# --- modes and manifest -----------------------------------------------------------------------------


def load_modes(path: Path = MODES) -> dict[str, dict[str, dict[str, Any]]]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def apply_mode(cfg: Config, overrides: dict[str, dict[str, Any]]) -> None:
    """Override config fields in place (section -> field -> value), validated like the config file."""
    for section, fields in overrides.items():
        sec = getattr(cfg, section)
        setattr(cfg, section, type(sec).model_validate({**sec.model_dump(), **fields}))


def file_sha256(path: Path, cache: Path | None = None) -> str | None:
    """The file's sha256 (None when it is missing); with `cache`, reused while its size and mtime are unchanged."""
    if not path.is_file():
        return None
    st, key = path.stat(), str(path.resolve())
    known = json.loads(cache.read_text(encoding="utf-8")) if cache and cache.is_file() else {}
    if (hit := known.get(key)) and (hit["size"], hit["mtime_ns"]) == (st.st_size, st.st_mtime_ns):
        return hit["sha256"]
    if cache:
        print(f"sha256 of {path} (once: cached in {cache})", flush=True)
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if cache:
        known[key] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": digest}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(known, indent=2), encoding="utf-8")
    return digest


def gpu_info() -> list[dict[str, str]] | None:
    """nvidia-smi's name, power limit and draw, and memory per GPU; None when it is not available."""
    try:
        out = subprocess.run(NVIDIA_SMI, capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    fields = ("name", "power_limit", "power_draw", "memory_total")
    return [dict(zip(fields, (v.strip() for v in line.split(",")))) for line in out.splitlines() if line.strip()]


def wsl_path(path: str) -> str:
    """A Windows path (C:/x, also with backslashes) as WSL sees it (/mnt/c/x); other paths unchanged."""
    m = re.match(r"([A-Za-z]):[/\\](.*)", path)
    return f"/mnt/{m.group(1).lower()}/{m.group(2).replace(chr(92), '/')}" if m else path


def _git(root: Path) -> dict[str, Any] | None:
    """Commit, dirty flag and, when dirty, the sha256 of the uncommitted changes. Line endings are compared as
    Windows git checks them out (core.autocrlf), so a Windows checkout read from WSL does not look all modified.
    A worktree made by Windows git has a `gitdir: C:/...` file that Linux git cannot follow: then git runs with
    that gitdir converted to /mnt/c/..."""
    git = ["git", "-c", "core.autocrlf=true", "--no-optional-locks"]
    for attempt in range(2):
        try:
            if attempt:
                gitdir = (root / ".git").read_text(encoding="utf-8").strip().removeprefix("gitdir:").strip()
                git += [f"--git-dir={wsl_path(gitdir)}", f"--work-tree={root}"]
            sha = subprocess.run([*git, "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True)
            status = subprocess.run([*git, "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True)
            dirty = bool(status.stdout.strip())
            return {"sha": sha.stdout.strip(), "dirty": dirty, "diff_sha256": _changes(git, root) if dirty else None}
        except (OSError, subprocess.SubprocessError):
            continue
    return None


def _changes(git: list[str], root: Path) -> str:
    """sha256 of the uncommitted changes: the diff against the commit, and each untracked (not ignored) file's
    name and content."""
    diff, untracked = (subprocess.run([*git, *args], cwd=root, capture_output=True, check=True).stdout
                       for args in (["diff", "HEAD", "--binary"], ["ls-files", "--others", "--exclude-standard", "-z"]))
    digest = hashlib.sha256(diff)
    for name in sorted(filter(None, untracked.split(b"\0"))):
        digest.update(name + b"\0" + (root / name.decode()).read_bytes())
    return digest.hexdigest()


def served_model(cfg: Config) -> Path:
    """The model file the backend serves: llama-server's /props `model_path` when it answers (best effort),
    else the configured path (MODEL_PATH), else the copy scripts/wsl/run.sh uses ($MODELS_DIR/<name>)."""
    base = re.sub(r"/v1$", "", cfg.model_backend.api_base.rstrip("/"))
    try:
        if path := httpx.get(f"{base}/props", timeout=2).json().get("model_path"):
            return Path(path)
    except (httpx.HTTPError, ValueError, AttributeError):
        pass
    configured = cfg.llama_server.model_path
    copy = Path(os.environ.get("MODELS_DIR") or Path.home() / ".local/share/h-ai-senberg/models") / configured.name
    return copy if not configured.is_file() and copy.is_file() else configured


def _rel(path: Path) -> str:
    """A path relative to the repository when it is inside it (runs move between Windows and WSL)."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def store_identity(path: Path) -> dict[str, Any]:
    """Path, size and mtime of a knowledge store file, or of a store folder the total size and newest mtime of
    its files; size and mtime are None when it is missing."""
    stats = [p.stat() for p in ([path] if path.is_file() else path.rglob("*")) if p.is_file()]
    if not stats:
        return {"path": str(path), "size": None, "mtime": None}
    return {"path": str(path), "size": sum(s.st_size for s in stats),
            "mtime": datetime.fromtimestamp(max(s.st_mtime for s in stats)).isoformat(timespec="seconds")}


def manifest(cfg: Config, run_id: str, mode: str, files: list[Path], args: list[str], cache: Path) -> dict[str, Any]:
    """What a run depends on: code, config after the mode, model file, knowledge setting and stores (see
    store_identities), eval files, GPU, sampling."""
    backend, model = cfg.model_backend, served_model(cfg)
    return {
        "run_id": run_id,
        "git": _git(cfg.root),
        "date": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "model_profile": os.environ.get(PROFILE_ENV),
        "model_id": backend.model_id,
        "backend": backend.api_base,
        "config_sha256": hashlib.sha256(cfg.model_dump_json().encode()).hexdigest(),
        "model_file": {"path": str(model), "sha256": file_sha256(model, cache)},
        "knowledge": cfg.retrieval.knowledge,
        "stores": store_identities(cfg),
        "eval_files": [{"path": _rel(f), "sha256": file_sha256(f)} for f in files],
        "seed": None,  # the runner does not shuffle; --limit picks the same items every time
        "gpu": gpu_info(),
        "context_tokens": cfg.context_tokens,
        "sampling": {"temperature": backend.temperature, "max_tokens": backend.max_tokens, "seed": backend.seed,
                     **backend.extra_body},
        "args": args,
    }


def store_identities(cfg: Config) -> dict[str, dict[str, Any]]:
    """Every store of the registry (kind, whether the knowledge setting enables it, path, size, mtime), then every
    other store path of the retrieval settings; raw_dir holds the downloads they are built from."""
    r = cfg.retrieval
    enabled = r.knowledge_sets[r.knowledge]
    found = {name: {"kind": s.kind, "enabled": name in enabled, **store_identity(s.path)}
             for name, s in r.stores.items()}
    found.update({name: store_identity(path) for name, path in r if isinstance(path, Path) and name != "raw_dir"})
    return found


def identity(man: dict[str, Any]) -> dict[str, Any]:
    """The manifest fields that define a run's experiment, besides its eval files (see open_run)."""
    git = man.get("git") or {}
    return {"mode": man.get("mode"), "config_sha256": man.get("config_sha256"), "commit": git.get("sha"),
            "dirty": git.get("dirty"), "diff_sha256": git.get("diff_sha256"),
            "model_sha256": (man.get("model_file") or {}).get("sha256"), "stores": man.get("stores"),
            "sampling": man.get("sampling")}


def open_run(folder: Path, man: dict[str, Any]) -> None:
    """Write the manifest of a new run. A resumed run must match the stored manifest on every identity field, and
    each of its eval files must be stored with the same sha256 (fewer files are fine); else ValueError naming
    what differs."""
    path = folder / "manifest.json"
    if path.is_file():
        old = json.loads(path.read_text(encoding="utf-8"))
        was, now = identity(old), identity(man)
        changed = [field for field in now if now[field] != was[field]]
        stored = {(f["path"], f["sha256"]) for f in old.get("eval_files") or []}
        if any((f["path"], f["sha256"]) not in stored for f in man.get("eval_files") or []):
            changed.append("eval_files")
        if changed:
            raise ValueError(f"run {folder.name} differs from its manifest in {', '.join(changed)}; use a new --run-id")
        return
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")


def runs_dir() -> Path:
    """Where runs live: the evals folder of the data folder, ${DATA_DIR:-data} (relative to the repository)."""
    load_dotenv(ROOT / ".env", override=False)
    return ROOT / (os.environ.get("DATA_DIR") or "data") / "evals"


def run_dir(run: str) -> Path:
    """A run folder: the path itself when it is a folder, else <data>/evals/<run>."""
    return Path(run) if Path(run).is_dir() else runs_dir() / run


def read_logs(folder: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((folder / "items").glob("*.json"))]


def run_items(folder: Path) -> dict[str, dict[str, Any]]:
    """Item id -> item (with its key) from the eval files a run's manifest lists."""
    man = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    return {i["id"]: i for f in man["eval_files"] for i in load_file(ROOT / f["path"])}


# --- running and scoring -------------------------------------------------------------------------------


def log_scores(answer: str, draft: str, q: dict[str, Any], error: str | None, report: dict[str, Any]) -> dict:
    """The score fields of an item log: correct, scores (final and draft) and unmarked_contradiction (wrong
    although every claim is labelled supported)."""
    final, claims = score(answer, q), report.get("claims", [])
    all_supported = bool(claims) and all(c["outcome"] == "supported" for c in claims)
    return {"correct": final["correct"] if final else None,
            "scores": {"final": final, "draft": score(draft, q)} if final else None,
            "unmarked_contradiction": error is None and bool(final) and not final["correct"] and all_supported}


async def run_one(agent: Agent, q: dict[str, Any], mode: str = "config") -> dict[str, Any]:
    messages = [*q.get("turns", []), {"role": "user", "content": q["question"]}]
    t0, calls = time.monotonic(), agent.llm.calls
    answer = draft = trace = ""
    report: dict[str, Any] = {}
    timings: dict[str, float] = {}
    error = None
    try:
        async for item in agent.run(messages):
            if text := item.get("reasoning_content"):
                at = round(time.monotonic() - t0, 2)
                timings.setdefault("first_reasoning_s", at)
                if text == DRAFT_MARK:  # the model's first answer token follows
                    timings.setdefault("first_answer_s", at)
                trace += text
            answer += item.get("content", "")
            draft = item.get("draft", draft)
            report = item.get("verification", report)
    except (UpstreamError, httpx.HTTPError) as e:
        error = str(e)
    timings["total_s"] = round(time.monotonic() - t0, 2)
    found = artefacts(answer)
    log = {
        "id": q["id"],
        "set": q["set"],
        "mode": mode,
        "messages": messages,
        "draft": draft,
        "answer": answer,
        "reasoning": trace,
        "verification": report,
        "timings": timings,
        "model_calls": agent.llm.calls - calls,
        "chars": {"reasoning": len(trace), "draft": len(draft), "answer": len(answer)},
        "error": error,
        **log_scores(answer, draft, q, error, report),
        "artefacts": found,
    }
    if conv := (q.get("key") or {}).get("conversation"):  # S6
        log["conversation"] = {"artefact_free": not found, "max_chars": len(answer) <= conv["max_chars"],
                               "max_latency": timings["total_s"] <= conv["max_latency_s"]}
    return log


def _pct(n: int, total: int) -> float | None:
    return round(100 * n / total, 1) if total else None


def summarize(logs: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in logs if r["error"] is None]
    scored = [r["scores"] for r in logs if r["scores"]]
    parts = [v for s in scored for v in s["final"]["parts"].values()]
    verdicts = [v for s in scored for v in s["final"]["verdicts"].values()]
    # logs scored before body-first reading have no line_parts or sources: every part was read from its line
    line_parts = [v for s in scored for v in s["final"].get("line_parts", s["final"]["parts"]).values()]
    pairs = [(s["draft"]["parts"][k], v) for r in ok if (s := r["scores"]) for k, v in s["final"]["parts"].items()]
    latency = sorted(r["timings"]["total_s"] for r in ok)
    counts = {o: sum(r["verification"].get("counts", {}).get(o, 0) for r in logs) for o in OUTCOMES}
    claims = sum(counts.values())
    solved = sum(s["final"]["correct"] for s in scored)
    facts = [s["final"] for s in scored if "loose_correct" in s["final"]]
    summary: dict[str, Any] = {
        "items": len(logs),
        "errors": len(logs) - len(ok),
        "empty_answers": sum(not r["answer"].strip() for r in ok),
        "scored": len(scored),
        "correct": solved,
        "correct_pct": _pct(solved, len(scored)),
        "facts": len(facts),
        "facts_correct": sum(f["correct"] for f in facts),
        "loose_correct": sum(f["loose_correct"] for f in facts),
        "parts_right": sum(parts),
        "parts_total": len(parts),
        "parts_pct": _pct(sum(parts), len(parts)),
        "line_parts_right": sum(line_parts),
        "body_verdict_parts": sum(src == "body" for s in scored for src in s["final"].get("sources", {}).values()),
        "verdict_parts_right": sum(verdicts),
        "verdict_parts_total": len(verdicts),
        "verdict_pct": _pct(sum(verdicts), len(verdicts)),
        "missing_lines": sum(len(r["scores"]["final"]["missing"]) for r in ok if r["scores"]),
        "draft_parts_right": sum(d for d, _ in pairs),
        "false_corrections": sum(d and not f for d, f in pairs),
        "true_corrections": sum(f and not d for d, f in pairs),
        "with_artefacts": sum(bool(r["artefacts"]) for r in ok),
        "tool_call_leaks": sum("tool_call" in r["artefacts"] for r in ok),
        "artefacts": dict(Counter(k for r in ok for k in r["artefacts"])),
        "latency_median_s": statistics.median(latency) if latency else None,
        "latency_p90_s": latency[math.ceil(0.9 * len(latency)) - 1] if latency else None,
        "model_calls": sum(r["model_calls"] for r in logs),
        "claims": claims,
        "supported_claims_pct": _pct(counts["supported"], claims),
        **{f"{o}_claims": c for o, c in counts.items()},
        "unmarked_contradictions": sum(r["unmarked_contradiction"] for r in logs),
        "consistency_fixes": sum(len(r["verification"].get("consistency", [])) for r in logs),
        "arithmetic_fixes": sum(len(r["verification"].get("arithmetic", [])) for r in logs),
        "check_fixes": sum(c["fixed"] is not None for r in logs for c in r["verification"].get("checks", [])),
    }
    if conv := [r["conversation"] for r in logs if "conversation" in r]:
        summary["conversation"] = {"items": len(conv), **{k: sum(c[k] for c in conv) for k in conv[0]}}
    return summary


def review_summary(rows: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    """Review metrics of (review, key) pairs."""
    planted = sum(len(k.get("defects") or []) for _, k in rows)
    expected = [(r.get("verdicts", {}).get(item), str(v["verdict"])) for r, k in rows
                for item, v in (k.get("verdicts") or {}).items()]
    right = sum(str(got).strip().lower() == want.strip().lower() for got, want in expected if got is not None)
    return {
        "reviewed": len(rows),
        "mean_score": round(statistics.mean(r["score"] for r, _ in rows), 2) if rows else None,
        "defects_found_pct": _pct(sum(len(r.get("defects_found", [])) for r, _ in rows), planted),
        "verdict_pct": _pct(right, len(expected)),
        "omissions": sum(len(r.get("omissions", [])) for r, _ in rows),
        **{field: sum(r.get(field, 0) for r, _ in rows) for field in REVIEW_COUNTS},
        "must_not_violations": sum(len(r.get("must_not_violated", [])) for r, _ in rows),
        "new_errors": dict(Counter(e["tag"] for r, _ in rows for e in r.get("new_errors", []))),
    }


def write_summary(folder: Path, mode: str) -> dict[str, Any]:
    """summary.json of every item log in the run folder: all items and each set, and the review metrics when
    the run has reviews.json (evals.review merge)."""
    logs = read_logs(folder)
    set_of = {r["id"]: r["set"] for r in logs}
    sets = sorted(set(set_of.values()))
    summary = {"run_id": folder.name, "mode": mode, "all": summarize(logs),
               "sets": {s: summarize([r for r in logs if r["set"] == s]) for s in sets}}
    if (path := folder / "reviews.json").is_file():
        reviews, items = json.loads(path.read_text(encoding="utf-8")), run_items(folder)
        rows = [(rv, key_of(items[iid]), set_of[iid]) for iid, rv in reviews.items() if iid in set_of]
        summary["review"] = {"all": review_summary([(rv, k) for rv, k, _ in rows]),
                             "sets": {s: review_summary([(rv, k) for rv, k, x in rows if x == s])
                                      for s in sorted({x for *_, x in rows})}}
    (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _done(log: Path, retry_errors: bool) -> bool:
    return log.is_file() and not (retry_errors and json.loads(log.read_text(encoding="utf-8"))["error"])


async def run_eval(agent: Agent, items: list[dict], folder: Path, mode: str = "config",
                   retry_errors: bool = False) -> dict[str, Any]:
    """Run the items without a log in the folder (with retry_errors also those whose log has an error), writing
    each log when its item is done; then write and return the summary of every log in the folder."""
    (folder / "items").mkdir(parents=True, exist_ok=True)
    todo = [q for q in items if not _done(folder / "items" / f"{q['id']}.json", retry_errors)]
    if len(todo) < len(items):
        print(f"{len(items) - len(todo)} items already done", flush=True)
    try:
        for i, q in enumerate(todo, 1):
            r = await run_one(agent, q, mode)
            (folder / "items" / f"{q['id']}.json").write_text(json.dumps(r, ensure_ascii=False, indent=2),
                                                               encoding="utf-8")
            mark = "ERR  " if r["error"] else "-    " if r["correct"] is None else "ok   " if r["correct"] else "WRONG"
            line = ""
            if s := r["scores"]:
                line = " parts " + "".join("+" if ok else "-" for ok in s["final"]["parts"].values())
                if s["draft"]["parts"] != s["final"]["parts"]:
                    line += " (draft " + "".join("+" if ok else "-" for ok in s["draft"]["parts"].values()) + ")"
            print(f"[{i}/{len(todo)}] {mark} {q['set']:<3} {q['id']:<16} {r['timings']['total_s']:6.1f}s{line}"
                  f"  claims {r['verification'].get('counts', {})}"
                  + (f"  ARTEFACTS {','.join(r['artefacts'])}" if r["artefacts"] else "")
                  + ("  UNMARKED" if r["unmarked_contradiction"] else ""), flush=True)
    finally:
        summary = write_summary(folder, mode)
    return summary


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    modes = load_modes()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", choices=SETS, help="S1/S2: facts, reasoning or all (default: all; none with --authored)")
    parser.add_argument("--holdout", action="store_true", help="run only the held-out reasoning problems")
    parser.add_argument("--authored", type=Path, action="append", help="batch folder of authored items (repeatable)")
    parser.add_argument("--split", choices=SPLITS, help="authored items of this split only")
    parser.add_argument("--sets", help="comma-separated sets to keep, e.g. S3,S4 (S1 facts, S2 reasoning)")
    parser.add_argument("--mode", choices=sorted(modes), help="pipeline mode (evals/modes.yaml); default: as configured")
    parser.add_argument("--run-id", help="run folder under ${DATA_DIR:-data}/evals (default <yyyymmdd-hhmm>-<mode>);"
                                         " an existing run is resumed: items with a log are skipped")
    parser.add_argument("--retry-errors", action="store_true", help="when resuming, run the items that failed again")
    parser.add_argument("--ids", help="comma-separated item ids")
    parser.add_argument("--limit", type=int, help="run N items, taken in turn from each group")
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--problems", type=Path, default=PROBLEMS)
    args = parser.parse_args(argv)
    mode = args.mode or "config"
    try:
        cfg = load_config()
        apply_mode(cfg, modes.get(mode, {}))
        _, facts = load_questions(args.questions)
        items = select(facts, load_problems(args.problems), args.set or (None if args.authored else "all"), args.holdout)
        for folder in args.authored or []:
            items += load_authored(folder)
        items = choose(items, args.split, args.sets.split(",") if args.sets else None)
        chosen = pick(items, args.ids.split(",") if args.ids else None, args.limit)
        if dupes := sorted(i for i, n in Counter(q["id"] for q in chosen).items() if n > 1):
            raise ValueError(f"duplicate item ids {dupes}")
        run_id = args.run_id or f"{datetime.now():%Y%m%d-%H%M}-{mode}"
        folder = runs_dir() / run_id
        files = sorted({q["file"] for q in chosen})
        knowledge = Knowledge.open(cfg)  # before the run folder: a missing literature store leaves nothing behind
        open_run(folder, manifest(cfg, run_id, mode, files, sys.argv[1:] if argv is None else argv,
                                  runs_dir() / ".sha-cache.json"))
    except (ConfigError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if not knowledge.tools and cfg.reasoning_policy.max_search_depth:
        print("warning: no knowledge tools (store missing?): answers are neither grounded nor verified", file=sys.stderr)

    async def go() -> dict:
        llm = LLMClient(cfg.model_backend)
        try:
            return await run_eval(Agent(cfg, llm, knowledge), chosen, folder, mode, args.retry_errors)
        finally:
            await llm.aclose()

    summary = asyncio.run(go())
    print()
    for name, s in [("all", summary["all"]), *summary["sets"].items()]:
        print(f"{name}: {s['items']} items, {s['errors']} errors, solved {s['correct']}/{s['scored']},"
              f" parts {s['parts_right']}/{s['parts_total']}, {s['missing_lines']} missing lines,"
              f" {s['false_corrections']} false / {s['true_corrections']} true corrections,"
              f" {s['with_artefacts']} with artefacts ({s['tool_call_leaks']} tool-call leaks),"
              f" median {s['latency_median_s']} s")
    print(f"run: {folder}  (uv run python -m evals.report {run_id})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
