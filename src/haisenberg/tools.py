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
from dataclasses import dataclass
from typing import Any

from . import calc
from .config import Config, ConfigError
from .l1.store import STOPWORDS, L1Error, L1Store, Record

log = logging.getLogger(__name__)

SEARCH = "search_knowledge"
CALCULATE = "calculate"  # built-in arithmetic tool, offered next to the knowledge tools
TEXT = "text"  # the kind searched by SEARCH
_CITE = re.compile(r"\[S(\d+)\]")


@dataclass
class Source:
    id: str  # "S1", "S2", ...
    kind: str
    record: Record
    passage: str  # exactly what the model was shown


class Sources:
    """The records retrieved for one answer, numbered S1, S2, ... in the order first seen."""

    def __init__(self) -> None:
        self._by_url: dict[str, Source] = {}
        self.items: list[Source] = []

    def add(self, kind: str, record: Record, passage: str) -> tuple[Source, bool]:
        """(source, is_new); a record seen before keeps its id."""
        key = record.citation.url
        if key in self._by_url:
            return self._by_url[key], False
        src = Source(f"S{len(self.items) + 1}", kind, record, passage)
        self._by_url[key] = src
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
    """The store plus the tools generated from it; no store (or depth 0) means no tools."""

    def __init__(self, store: L1Store | None, cfg: Config):
        self.store = store
        self.top_k = cfg.retrieval.top_k_proofs
        self.passage_chars = cfg.retrieval.passage_chars
        self.min_share = cfg.retrieval.min_term_share
        kinds = store.kinds() if store is not None and cfg.reasoning_policy.max_search_depth > 0 else {}
        self.domains = kinds.get(TEXT, [])
        self._kind_of: dict[str, str] = {}  # tool name -> kind
        self.tools: list[dict[str, Any]] = []
        for kind in sorted(kinds, key=lambda k: (k != TEXT, k)):
            name = SEARCH if kind == TEXT else f"lookup_{kind}"
            self._kind_of[name] = kind
            self.tools.append(_tool(name, self.domains if kind == TEXT else None, cfg))
        self.calculator = bool(self.tools) and CALCULATE in cfg.tools  # built in, next to the knowledge tools
        if self.calculator:
            self.tools.append(_tool(CALCULATE, None, cfg, param="expression"))

    @classmethod
    def open(cls, cfg: Config) -> Knowledge:
        try:
            store = L1Store(cfg.retrieval.layer1_encyclopedia_db, cfg.retrieval.top_k_proofs)
        except L1Error as e:
            log.warning("no knowledge tools: %s", e)
            store = None
        return cls(store, cfg)

    def run(self, name: str, arguments: str, sources: Sources, k: int | None = None) -> tuple[str, list[str], list[str]]:
        """Run one tool call returning at most k records (default retrieval.top_k_proofs):
        (result text for the model, one line per result for the reasoning, the records' source ids)."""
        if name == CALCULATE and self.calculator:
            return (*calc.run_tool(arguments), [])
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
                blocks.append(f"[{src.id}] {record.label()} ({kind}; {', '.join(record.domains)})\n{src.passage}")
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
            return kind, query, self.store.search_encyclopedia(query, k, domain)
        except (L1Error, sqlite3.Error) as e:
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


def _tool(name: str, domains: list[str] | None, cfg: Config, param: str = "query") -> dict[str, Any]:
    """OpenAI function tool: parameters from code, texts from tools.<name> in the config."""
    text = cfg.tools.get(name)
    if text is None:
        raise ConfigError(f"tools.{name}: the store offers this tool but the config has no text for it")
    properties: dict[str, dict[str, Any]] = {param: {"type": "string"}}
    if domains is not None:
        properties["domain"] = {"type": "string", "enum": domains}
    for key, schema in properties.items():
        if key not in text.parameters:
            raise ConfigError(f"tools.{name}.parameters.{key}: missing description")
        schema["description"] = text.parameters[key]
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": text.description,
            "parameters": {"type": "object", "properties": properties, "required": [param]},
        },
    }
