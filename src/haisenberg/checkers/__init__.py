"""Domain checker plug-ins (D-034): code checks that know one subject, run by the verifier on every draft.
The verifier knows only this interface; which checkers run is `verifier.checkers` in the config.

A checker is a module in this package with `NAME` and `check(draft, task) -> list[Finding]` (`task`: the user's
message, "" when unknown). Add a module and its name to CHECKERS to register it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Finding:
    """One problem a checker found. With `fixed`, the span [start, end) of the draft is replaced by it;
    without, the problem is only reported (in the reasoning block and the verification report)."""

    start: int
    end: int
    text: str  # what was checked, as written
    fixed: str | None
    note: str
    doubt: str | None = None  # why the text may not be in the checker's subject: then the fix is only reported


def _registry() -> dict[str, Callable[[str, str], list[Finding]]]:
    from . import element_balance

    return {m.NAME: m.check for m in (element_balance,)}


CHECKERS = _registry()
