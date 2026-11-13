"""Stage log of one answer, for in-process callers (the eval runner): where its seconds went.

A caller opens a log with `recording()`. While it is open, the client records every model call, the knowledge
tools every tool call and the verifier its code checks, each as one record `{stage, kind, start_s, seconds, ...}`
appended when the stage starts. Nothing is recorded unless a log is open, so the chat server pays one lookup per
stage and no request or answer differs. The log is a context variable: the caller that opened it must iterate the
agent itself (the eval runner does), and no constructor needs to pass it on.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

MODEL, TOOL, CHECK = "model", "tool", "check"  # the kinds of a record

_LOG: ContextVar[tuple[float, list[dict[str, Any]]] | None] = ContextVar("haisenberg_stage_log", default=None)


@contextmanager
def recording() -> Iterator[list[dict[str, Any]]]:
    """Open a stage log for the code run inside the block; yields the list the records are appended to."""
    records: list[dict[str, Any]] = []
    token = _LOG.set((time.monotonic(), records))
    try:
        yield records
    finally:
        _LOG.reset(token)


@contextmanager
def stage(name: str, kind: str, **info: Any) -> Iterator[dict[str, Any] | None]:
    """Time the block as one stage of the open log; yields its record (to add fields to), or None when no log is
    open. `seconds` is set when the block ends, also when it raises or its generator is closed."""
    log = _LOG.get()
    if log is None:
        yield None
        return
    opened, records = log
    start = time.monotonic()
    record: dict[str, Any] = {"stage": name, "kind": kind, "start_s": round(start - opened, 3), "seconds": None, **info}
    records.append(record)
    try:
        yield record
    finally:
        record["seconds"] = round(time.monotonic() - start, 3)
