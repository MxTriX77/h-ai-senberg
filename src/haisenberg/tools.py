"""Agent tools generated from the knowledge in the Layer 1 store (D-021), and the sources they return.

The tools follow the kinds of knowledge the store reports (`L1Store.kinds()`): one `search_knowledge`
over every text source and one `lookup_<kind>` per structured kind. Their texts come from `tools.*` in
the config. Nothing here names a specific source.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import calc, compute
from .config import Config, ConfigError
from .l1.store import STOPWORDS, L1Error, L1Store, Record
from .l2.store import L2Error, L2Store, PassageRecord

log = logging.getLogger(__name__)

SEARCH = "search_knowledge"
CALCULATE = "calculate"  # built-in arithmetic tool, offered next to the knowledge tools
COMPUTE = "compute"  # built-in procedures tool (dates, lists, schedules), offered right after the calculator
CALC_TOOLS = (CALCULATE, COMPUTE)  # their rounds are calculator rounds (reasoning_policy.max_calc_rounds)
TEXT = "text"  # the kind searched by SEARCH
_CITE = re.compile(r"\[S(\d+)\]")


@dataclass
class Source:
    id: str  # "S1", "S2", ...
    kind: str
    record: Record
    passage: str  # exactly what the model was shown
    literature: bool = False  # a literature passage (Layer 2), not a reference record


class Sources:
    """The records retrieved for one answer, numbered S1, S2, ... in the order first seen."""

    def __init__(self) -> None:
        self._by_key: dict[tuple, Source] = {}
        self.items: list[Source] = []

    def add(self, kind: str, record: Record, passage: str) -> tuple[Source, bool]:
        """(source, is_new); a record seen before (same citation URL and, for a literature passage, passage
        number) keeps its id."""
        key = (record.citation.url, record.citation.locator.get("passage"))
        if key in self._by_key:
            return self._by_key[key], False
        src = Source(f"S{len(self.items) + 1}", kind, record, passage, isinstance(record, PassageRecord))
        self._by_key[key] = src
        self.items.append(src)
        return src, True

    def get(self, sid: str) -> Source | None:
        """The source with id 'S<n>', or None if it was not issued."""
        n = int(sid[1:]) if sid[:1] == "S" and sid[1:].isdigit() else 0
        return self.items[n - 1] if 0 < n <= len(self.items) else None

    def cited(self, text: str) -> list[Source]:
        """Issued sources cited as [S#] in text, in id order."""
        found = {int(n) for n in _CITE.findall(text)}
        return [s for i, s in enumerate(self.items, 1) if i in found]

    def source_list(self, text: str) -> str:
        """The source list for the sources text cites ('' when it cites none)."""
        lines = []
        for s in self.cited(text):
            c = s.record.citation
            lines.append(f"- [{s.id}] {s.record.label()}: {c.source} ({c.version}), {c.license}, {c.url}")
        return "\n\nSources:\n" + "\n".join(lines) if lines else ""


class Knowledge:
    """The store plus the tools generated from it; no store (or depth 0) means no tools. With a literature store
    (enabled by the knowledge setting), searches also return up to retrieval.literature_k of its passages."""

    def __init__(self, store: L1Store | None, cfg: Config, literature: L2Store | None = None):
        self.store = store
        self.literature = literature
        self.literature_k = cfg.retrieval.literature_k
        self.top_k = cfg.retrieval.top_k_proofs
        self.passage_chars = cfg.retrieval.passage_chars
        self.min_share = cfg.retrieval.min_term_share
        labelled = literature is not None and cfg.retrieval.label_sources  # base: no labels, no extra tool text
        self.labels = cfg.retrieval.source_labels if labelled else None
        kinds = store.kinds() if store is not None and cfg.reasoning_policy.max_search_depth > 0 else {}
        self.domains = kinds.get(TEXT, [])
        self._kind_of: dict[str, str] = {}  # tool name -> kind
        self.tools: list[dict[str, Any]] = []
        for kind in sorted(kinds, key=lambda k: (k != TEXT, k)):
            name = SEARCH if kind == TEXT else f"lookup_{kind}"
            self._kind_of[name] = kind
            self.tools.append(_tool(name, self.domains if kind == TEXT else None, cfg, literature=labelled))
        self.calculator = bool(self.tools) and CALCULATE in cfg.tools  # built in, next to the knowledge tools
        if self.calculator:
            self.tools.append(_tool(CALCULATE, None, cfg, param="expressions",
                                    schema={"type": "array", "items": {"type": "string"}}))
        self.computer = self.calculator and cfg.reasoning_policy.compute_tool  # switch; its texts are then required
        if self.computer:
            self.tools.append(_tool(COMPUTE, None, cfg, param="operations",
                                    schema={"type": "array", "items": {"type": "string"}}))

    @classmethod
    def open(cls, cfg: Config) -> Knowledge:
        """The stores the knowledge setting enables. Raises ConfigError when it enables a literature store that
        cannot be opened (an A/B run must never fall back to less knowledge)."""
        r, store, literature = cfg.retrieval, None, None
        try:
            store = L1Store(r.path("encyclopedia"), r.top_k_proofs)
        except L1Error as e:
            log.warning("no knowledge tools: %s", e)
        if (path := r.enabled("literature")) is not None:
            try:
                literature = L2Store(path)
            except L2Error as e:
                raise ConfigError(f"KNOWLEDGE={r.knowledge} (retrieval.knowledge) needs the literature store: {e}") from e
        return cls(store, cfg, literature)

    def run(self, name: str, arguments: str, sources: Sources, k: int | None = None) -> tuple[str, list[str], list[str]]:
        """Run one tool call returning at most k records (default retrieval.top_k_proofs):
        (result text for the model, one line per result for the reasoning, the records' source ids)."""
        if name == CALCULATE and self.calculator:
            return (*_each(arguments, "expressions", lambda e: calc.run_tool(json.dumps({"expression": e}))[0]), [])
        if name == COMPUTE and self.computer:
            return (*_each(arguments, "operations", compute.run), [])
        try:
            kind, query, records = self._call(name, arguments, k or self.top_k)
        except _BadCall as e:
            return f"Error: {e}", [f"error: {e}"], []
        if not records:
            return f"Nothing found for {query!r}.", ["nothing found"], []
        weak: list[str] = []
        if kind == TEXT and self.min_share > 0:  # drop hits that share little more than a word with the query
            relevant = [getattr(r, "score", None) == 0  # the store matched its title or alias
                        or term_share(query, f"{r.label()} {r.passage(query, self.passage_chars)}") >= self.min_share
                        for r in records]
            weak = [f"weak match, not used: {r.label()}" for r, ok in zip(records, relevant) if not ok]
            records = [r for r, ok in zip(records, relevant) if ok]
            if not records:
                return f"Nothing relevant found for {query!r}.", [*weak, "nothing relevant found"], []
        blocks, lines, ids = [], [], []
        for record in records:
            src, new = sources.add(kind, record, record.passage(query, self.passage_chars))
            if new:
                tag = ("" if self.labels is None
                       else f"; {self.labels.literature if src.literature else self.labels.reference}")
                blocks.append(f"[{src.id}] {record.label()} ({kind}; {', '.join(record.domains)}{tag})\n{src.passage}")
                lines.append(f"[{src.id}] {record.label()}")
            else:
                blocks.append(f"[{src.id}] {record.label()} (shown above)")
                lines.append(f"[{src.id}] {record.label()} (shown above)")
            ids.append(src.id)
        return "\n\n".join(blocks), lines + weak, ids

    def _call(self, name: str, arguments: str, k: int) -> tuple[str, str, list[Record]]:
        kind = self._kind_of.get(name)
        if kind is None:
            raise _BadCall(f"unknown tool {name!r}; available: {', '.join(self._kind_of) or 'none'}")
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            raise _BadCall("arguments are not valid JSON") from None
        if not isinstance(args, dict):
            raise _BadCall("arguments must be a JSON object")
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise _BadCall("'query' must be a non-empty string")
        try:
            if kind != TEXT:
                return kind, query, self.store.lookup(kind, query, k)
            domain = args.get("domain") or None
            if domain is not None and domain not in self.domains:
                raise _BadCall(f"unknown domain {domain!r}; use one of {', '.join(self.domains)}")
            records = self.store.search_encyclopedia(query, k, domain)
            if self.literature is not None:
                records += self.literature.search(query, self.literature_k, domain)
            return kind, query, records
        except (L1Error, L2Error, sqlite3.Error) as e:
            raise _BadCall(f"lookup failed: {e}") from e


class _BadCall(Exception):
    pass


def term_share(query: str, text: str) -> float:
    """Share of the query's content words found in the text; words of 4+ letters match on their first four
    letters (close to the store's stemming), shorter ones exactly."""
    words = [w for w in re.findall(r"\w+", query.lower()) if w not in STOPWORDS]
    if not words:
        return 1.0
    have = set(re.findall(r"\w+", text.lower()))
    stems = {h[:4] for h in have if len(h) >= 4}
    return sum((w[:4] in stems) if len(w) >= 4 else (w in have) for w in words) / len(words)


def _each(arguments: str, param: str, run: Callable[[Any], str]) -> tuple[str, list[str]]:
    """A built-in list tool (`calculate`, `compute`): every entry of `param` through `run`, one result line each."""
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return "Error: arguments are not valid JSON", ["Error: arguments are not valid JSON"]
    entries = args.get(param) if isinstance(args, dict) else None
    if isinstance(entries, str):  # a list written as text ('["2 * 3", "4 / 5"]') or one entry
        try:
            listed = json.loads(entries)
        except json.JSONDecodeError:
            listed = None
        entries = listed if isinstance(listed, list) else [entries]
    if not isinstance(entries, list) or not entries:
        lines = [f"Error: '{param}' must be a non-empty list of {param}"]
    else:
        lines = [run(e) for e in entries]
    return "\n".join(lines), lines


def _tool(name: str, domains: list[str] | None, cfg: Config, param: str = "query",
          schema: dict[str, Any] | None = None, literature: bool = False) -> dict[str, Any]:
    """OpenAI function tool: parameters from code (`schema`: the main parameter's type, default string), texts
    from tools.<name> in the config (with `literature`, its `literature` text follows the description)."""
    text = cfg.tools.get(name)
    if text is None:
        raise ConfigError(f"tools.{name}: the store offers this tool but the config has no text for it")
    properties: dict[str, dict[str, Any]] = {param: schema or {"type": "string"}}
    if domains is not None:
        properties["domain"] = {"type": "string", "enum": domains}
    for key, schema in properties.items():
        if key not in text.parameters:
            raise ConfigError(f"tools.{name}.parameters.{key}: missing description")
        schema["description"] = text.parameters[key]
    description = f"{text.description} {text.literature}" if literature and text.literature else text.description
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": [param]},
        },
    }
