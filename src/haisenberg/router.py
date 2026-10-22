"""Request router (D-033): how much checking a message gets. Rules in code, no model call; thresholds and cue
words in config (`router`). Casual talk is answered directly, factual questions get lookups and claim checks,
analytic or multi-part requests get the full evidence protocol."""

from __future__ import annotations

import re

from .config import RouterConfig

CASUAL, FACTUAL, ANALYTIC = "casual", "factual", "analytic"

# A listed item: a line starting with a number, a letter or a bullet, or an inline label such as "Claim A:".
_LINE_ITEM = re.compile(r"(?m)^[ \t]*(?:\(?(\d{1,2}|[A-Za-zА-Яа-я])[.)]|[-*•])[ \t]+(\S.*)$")
_LABEL_ITEM = re.compile(r"(?i)\b(?:claim|statement|item|part|question|утверждение|пункт|вопрос|часть)\s+(\w{1,3})\s*[:.)]")


def request_items(text: str) -> list[tuple[str, str]]:
    """(id, text) of the items a message lists: numbered, lettered or bulleted lines (bullets are numbered in
    order), or labels such as "Claim A:" (the text up to the next label). Ids are upper case."""
    lines = [(m.group(1) or str(i), m.group(2).strip()) for i, m in enumerate(_LINE_ITEM.finditer(text), 1)]
    labels = list(_LABEL_ITEM.finditer(text))
    if len(lines) >= len(labels):
        return [(i.upper(), t) for i, t in lines]
    ends = [m.start() for m in labels[1:]] + [len(text)]
    return [(m.group(1).upper(), text[m.end() : end].strip()) for m, end in zip(labels, ends)]


def route(text: str, cfg: RouterConfig) -> tuple[str, str]:
    """(route, reason) for the last user message."""
    if not cfg.enabled:
        return FACTUAL, "router off"
    t, low = text.strip(), text.strip().lower()
    items = len(request_items(t))
    if items >= cfg.analytic_min_items:
        return ANALYTIC, f"{items} listed items"
    if len(t) >= cfg.analytic_min_chars:
        return ANALYTIC, f"{len(t)} characters"
    digits = bool(re.search(r"\d", t))
    if any(re.search(rf"(?<!\w){re.escape(c.lower())}", low) for c in cfg.analytic_cues) and (
        digits or _sentences(t) >= 2
    ):
        return ANALYTIC, "analysis asked for"
    if len(t) <= cfg.casual_max_chars and not digits and _small_talk(low, cfg):
        return CASUAL, "small talk"
    return FACTUAL, "a question to answer"


def _small_talk(low: str, cfg: RouterConfig) -> bool:
    """A casual cue and little else: "is it ok to mix bleach and ammonia?" holds a cue but is a question."""
    rest = low
    for c in sorted(cfg.casual_cues, key=len, reverse=True):  # longest first: "tell me a joke" before "a joke"
        rest = re.sub(rf"(?<!\w){re.escape(c.lower())}(?!\w)", " ", rest)
    return rest != low and len(re.findall(r"\w+", rest)) <= cfg.casual_max_other_words


def _sentences(text: str) -> int:
    return len([s for s in re.split(r"[.!?;]+(?:\s|$)", text) if s.strip()])
