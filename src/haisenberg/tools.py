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

from .config import Config, ConfigError
from .l1.store import L1Error, L1Store, Record

log = logging.getLogger(__name__)

SEARCH = "search_knowledge"
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
        kinds = store.kinds() if store is not None and cfg.reasoning_policy.max_search_depth > 0 else {}
        self.domains = kinds.get(TEXT, [])
        self._kind_of: dict[str, str] = {}  # tool name -> kind
        self.tools: list[dict[str, Any]] = []
        for kind in sorted(kinds, key=lambda k: (k != TEXT, k)):
            name = SEARCH if kind == TEXT else f"lookup_{kind}"
            self._kind_of[name] = kind
            self.tools.append(_tool(name, self.domains if kind == TEXT else None, cfg))

    @classmethod
    def open(cls, cfg: Config) -> Knowledge:
        try:
            store = L1Store(cfg.retrieval.layer1_encyclopedia_db, cfg.retrieval.top_k_proofs)
        except L1Error as e:
            log.warning("no knowledge tools: %s", e)
            store = None
        return cls(store, cfg)

    def run(self, name: str, arguments: str, sources: Sources) -> tuple[str, list[str]]:
        """Run one tool call: (result text for the model, one line per result for the reasoning)."""
        try:
            kind, query, records = self._call(name, arguments)
        except _BadCall as e:
            return f"Error: {e}", [f"error: {e}"]
        if not records:
            return f"Nothing found for {query!r}.", ["nothing found"]
        blocks, lines = [], []
        for record in records:
            src, new = sources.add(kind, record, record.passage(query, self.passage_chars))
            if new:
                blocks.append(f"[{src.id}] {record.label()} ({kind}; {', '.join(record.domains)})\n{src.passage}")
                lines.append(f"[{src.id}] {record.label()}")
            else:
                blocks.append(f"[{src.id}] {record.label()} (shown above)")
                lines.append(f"[{src.id}] {record.label()} (shown above)")
        return "\n\n".join(blocks), lines

    def _call(self, name: str, arguments: str) -> tuple[str, str, list[Record]]:
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
                return kind, query, self.store.lookup(kind, query, self.top_k)
            domain = args.get("domain") or None
            if domain is not None and domain not in self.domains:
                raise _BadCall(f"unknown domain {domain!r}; use one of {', '.join(self.domains)}")
            return kind, query, self.store.search_encyclopedia(query, self.top_k, domain)
        except (L1Error, sqlite3.Error) as e:
            raise _BadCall(f"lookup failed: {e}") from e


class _BadCall(Exception):
    pass


def _tool(name: str, domains: list[str] | None, cfg: Config) -> dict[str, Any]:
    """OpenAI function tool: parameters from code, texts from tools.<name> in the config."""
    text = cfg.tools.get(name)
    if text is None:
        raise ConfigError(f"tools.{name}: the store offers this tool but the config has no text for it")
    properties: dict[str, dict[str, Any]] = {"query": {"type": "string"}}
    if domains is not None:
        properties["domain"] = {"type": "string", "enum": domains}
    for param, schema in properties.items():
        if param not in text.parameters:
            raise ConfigError(f"tools.{name}.parameters.{param}: missing description")
        schema["description"] = text.parameters[param]
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": text.description,
            "parameters": {"type": "object", "properties": properties, "required": ["query"]},
        },
    }
