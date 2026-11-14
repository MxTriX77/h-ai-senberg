"""Items of the general schema: free-form requests of any category, with a reference for reviewers.

A file holds `items:`; an item has an id, a split, a category (any text: no subject is required), a language, a
request type, earlier turns, the prompt, a reference (answer, key points, sources and an optional check),
`must_not`, `offline_base`, probes and notes, and maybe a `family`, its group inside the category (README "Evals").
There are no answer lines and no `set`: in logs and summaries an item's set is its category. Fields the schema does
not name are kept and ignored.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml

SPLITS = ("dev", "holdout", "monitor")
LANGS = ("en", "ru")
REQUEST_TYPES = ("chat", "factual", "analytic", "audit")
OFFLINE = ("yes", "likely", "partial", "no", "n/a")
IN_BASE = ("yes", "likely")  # the offline base holds the needed facts: "I don't know" is then needless
CHECK_TYPES = ("none", "exact", "numeric", "python_tests")
CODE_LANGUAGES = ("python", "sql")


def is_general(data: Any) -> bool:
    """A file of the general schema: its items carry a category, a request type or a reference (authored items
    of the older schema carry a set and a key)."""
    items = data.get("items") if isinstance(data, dict) else None
    return isinstance(items, list) and any(
        isinstance(it, dict) and "set" not in it and {"category", "request_type", "reference"} & set(it)
        for it in items)


def _texts(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def check_values(check: dict[str, Any]) -> list[Any]:
    """The values of an exact or numeric check as a list (texts or numbers): every one must be in the answer."""
    value = check.get("value")
    values = value if isinstance(value, list) else [value]
    return [str(v) for v in values] if check.get("type") == "exact" else [float(v) for v in values]


def _check_problem(check: Any) -> str:
    """'' or what is wrong with a `reference.check`."""
    if not isinstance(check, dict) or check.get("type") not in CHECK_TYPES:
        return f"type must be one of {'|'.join(CHECK_TYPES)}"
    kind, value = check["type"], check.get("value")
    values = value if isinstance(value, list) else [value]
    if kind == "exact" and not (values and all(_number(v) or (isinstance(v, str) and v.strip()) for v in values)):
        return "an exact check needs a value: a text or a list of texts"
    if kind == "numeric":
        if not (values and all(_number(v) for v in values)):
            return "a numeric check needs a value: a number or a list of numbers"
        if not _number(check.get("tolerance")) or check["tolerance"] < 0:
            return "a numeric check needs a tolerance of 0 or more"
    if kind == "python_tests":
        if not (isinstance(check.get("tests"), str) and check["tests"].strip()):
            return "a python_tests check needs tests"
        if not (isinstance(check.get("entry_point"), str) and check["entry_point"].strip()):
            return "a python_tests check needs an entry_point"
        if check.get("language", "python") not in CODE_LANGUAGES:
            return f"language must be one of {'|'.join(CODE_LANGUAGES)}"
    return ""


def _problem(it: dict[str, Any]) -> str:
    """'' or the first thing wrong with an item, starting with the field's name."""
    for field, allowed in (("split", SPLITS), ("lang", LANGS), ("request_type", REQUEST_TYPES)):
        if it.get(field) not in allowed:
            return f"{field} must be one of {'|'.join(allowed)}"
    for field in ("category", "prompt", *(["family"] if "family" in it else [])):  # a family is optional
        if not (isinstance(it.get(field), str) and it[field].strip()):
            return f"{field} must be a non-empty text"
    turns = it.get("turns")
    if not isinstance(turns, list) or not all(
            isinstance(t, dict) and t.get("role") in ("user", "assistant") and isinstance(t.get("content"), str)
            for t in turns):
        return "turns must be a list of {role: user|assistant, content: text}"
    ref = it.get("reference")
    if not isinstance(ref, dict) or not isinstance(ref.get("answer"), str):
        return "reference needs an answer (text)"
    for field in ("key_points", "sources"):
        if not _texts(ref.get(field)):
            return f"reference.{field} must be a list of texts"
    if why := _check_problem(ref.get("check")):
        return f"reference.check: {why}"
    for field in ("must_not", "probes"):
        if not _texts(it.get(field)):
            return f"{field} must be a list of texts"
    base = it.get("offline_base")
    if isinstance(base, bool):  # YAML reads an unquoted yes or no as a boolean
        base = "yes" if base else "no"
    if base not in OFFLINE:
        return f"offline_base must be one of {'|'.join(OFFLINE)}"
    if not isinstance(it.get("notes"), str):
        return "notes must be a text"
    return ""


def load_general(path: Path) -> list[dict[str, Any]]:
    """The items of one file of the general schema, ready for the runner (`set`: the category, `question`: the
    prompt, `file`); raises ValueError naming the file, the item and the field of the first bad item."""
    items = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("items") or []
    seen: set[str] = set()
    for it in items:
        iid = it.get("id") if isinstance(it, dict) else None
        if not isinstance(iid, str) or not iid or iid in seen:
            raise ValueError(f"{path}: missing or duplicate id {iid!r}")
        seen.add(iid)
        if why := _problem(it):
            raise ValueError(f"{path}: item {iid}: {why}")
        base = it["offline_base"]
        it.update(set=it["category"], question=it["prompt"], file=path,
                  offline_base=("yes" if base else "no") if isinstance(base, bool) else base)
    return items


def load_paths(paths: list[Path]) -> list[dict[str, Any]]:
    """Every item of the given files and folders (of a folder every *.yaml); raises ValueError on a bad item, an
    empty folder or an id that occurs twice."""
    files: list[Path] = []
    for path in paths:
        found = sorted(path.glob("*.yaml")) if path.is_dir() else [path]
        if not found:
            raise ValueError(f"{path}: no *.yaml item files")
        files += found
    items = [it for f in files for it in load_general(f)]
    if dupes := sorted(i for i, n in Counter(it["id"] for it in items).items() if n > 1):
        raise ValueError(f"duplicate ids {dupes}")
    return items


def keep(items: list[dict[str, Any]], categories: list[str] | None = None, request_types: list[str] | None = None,
         langs: list[str] | None = None, probes: list[str] | None = None) -> list[dict[str, Any]]:
    """The items that match every filter given; an item of another schema (no category) passes."""
    def ok(it: dict[str, Any]) -> bool:
        return "category" not in it or (
            (not categories or it["category"] in categories)
            and (not request_types or it["request_type"] in request_types)
            and (not langs or it["lang"] in langs)
            and (not probes or bool(set(probes) & set(it["probes"]))))
    return [it for it in items if ok(it)]


def pick_proportional(items: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    """`limit` items: each category gets a share in proportion to its size (largest remainders; one item at
    least while the limit allows), taken at even steps (the middle of each step) over the category's items
    ordered by language, request type and id, so languages and request types keep about their shares. The same
    call picks the same items."""
    if not limit or limit >= len(items):
        return items
    groups: dict[str, list[dict[str, Any]]] = {}
    for it in items:
        groups.setdefault(it["category"], []).append(it)
    share = {c: limit * len(g) / len(items) for c, g in groups.items()}
    quota = {c: int(s) for c, s in share.items()}
    for c in sorted(groups, key=lambda c: share[c] - quota[c], reverse=True)[: limit - sum(quota.values())]:
        quota[c] += 1
    for c in sorted(groups, key=lambda c: len(groups[c]), reverse=True):  # a category left out takes one from
        donor = max(quota, key=lambda d: quota[d])                         # the largest quota, while that has two
        if not quota[c] and quota[donor] > 1:
            quota[c], quota[donor] = 1, quota[donor] - 1
    chosen: set[str] = set()
    for c, group in groups.items():
        ordered = sorted(group, key=lambda it: (it["lang"], it["request_type"], it["id"]))
        chosen |= {ordered[int((i + 0.5) * len(ordered) / quota[c])]["id"] for i in range(quota[c])}
    return [it for it in items if it["id"] in chosen]
