"""The suite's layers: everything it adds to a plain model call, each with the config switch that decides it.

An eval arm is the configured suite with layers switched off or on (`evals.run --minus`, `--plus`), or the bare
model with layers switched on (`--mode bare --plus`). The registry below names every layer, reads its state from
the config, writes it, and says what a layer cannot run without. A layer arm writes the switch of every layer, so
a switch hidden behind a master switch cannot come on by accident; what a layer needs is checked, never repaired:
the arm's name must say everything that differs from its base.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from haisenberg.checkers import CHECKERS
from haisenberg.config import Config
from haisenberg.tools import CALCULATE, COMPUTE, SEARCH, Knowledge

CONFIG, BARE = "config", "bare"  # the two bases of an arm: the suite as configured, the model alone


@dataclass(frozen=True)
class Layer:
    name: str
    adds: str  # what it adds to the bare call, in a few words
    switch: str  # the config key that decides it
    get: Callable[[Config], bool]
    put: Callable[[Config, bool], None]  # raises ValueError when the config holds no value to switch it on with
    needs: tuple[tuple[str, ...], ...] = ()  # of every group, one layer must be on
    verifier: bool = False  # runs inside the verifier: only with enable_multi_hop_verification and an open store


def _flag(section: str, field: str) -> tuple[Callable[[Config], bool], Callable[[Config, bool], None]]:
    """Getter and setter of a boolean switch `section.field` (`a.b` for a nested section)."""

    def target(cfg: Config) -> Any:
        obj: Any = cfg
        for part in section.split("."):
            obj = getattr(obj, part)
        return obj

    return (lambda cfg: bool(getattr(target(cfg), field))), (lambda cfg, on: setattr(target(cfg), field, on))


def _level(section: str, field: str) -> tuple[Callable[[Config], bool], Callable[[Config, bool], None]]:
    """Getter and setter of a switch that is a number: on above 0, off sets 0, on needs the config's own value."""

    def put(cfg: Config, on: bool) -> None:
        sec = getattr(cfg, section)
        if not on:
            setattr(sec, field, 0)
        elif not getattr(sec, field):
            raise ValueError(f"{section}.{field} is 0 in the config: set the value to switch it on with")

    return (lambda cfg: getattr(getattr(cfg, section), field) > 0), put


def _tool(name: str) -> tuple[Callable[[Config], bool], Callable[[Config, bool], None]]:
    def get(cfg: Config) -> bool:
        p = cfg.reasoning_policy
        return p.max_search_depth > 0 and name not in p.tools_off and (name != COMPUTE or p.compute_tool)

    def put(cfg: Config, on: bool) -> None:
        p = cfg.reasoning_policy
        if on and not p.max_search_depth:
            raise ValueError("reasoning_policy.max_search_depth is 0 in the config: set the tool rounds to offer"
                             " a tool")
        if name == COMPUTE:  # its own switch decides it: never listed in tools_off as well
            p.compute_tool = on
        off = set(p.tools_off) - {name} if on or name == COMPUTE else set(p.tools_off) | {name}
        p.tools_off = [t for t in cfg.tools if t in off]

    return get, put


def _thinking() -> tuple[Callable[[Config], bool], Callable[[Config, bool], None]]:
    def put(cfg: Config, on: bool) -> None:
        tb = cfg.reasoning_policy.thinking_budget
        if on and not tb.field:
            raise ValueError("reasoning_policy.thinking_budget.field is empty in the config: the backend takes no"
                             " budget")
        tb.enabled = on

    return (lambda cfg: bool(cfg.reasoning_policy.thinking_budget.field)
            and cfg.reasoning_policy.thinking_budget.enabled), put


def _checker(name: str) -> tuple[Callable[[Config], bool], Callable[[Config, bool], None]]:
    def put(cfg: Config, on: bool) -> None:
        v = cfg.verifier
        v.checkers = [c for c in CHECKERS if (c in v.checkers and c != name) or (on and c == name)]

    return (lambda cfg: name in cfg.verifier.checkers), put


def registry(cfg: Config) -> list[Layer]:
    """Every layer, in the order the suite applies them. The tool layers follow the config's `tools` entries and
    the checker layers the registered checkers."""
    tools = [f"tool.{name}" for name in cfg.tools]
    knowledge = tuple(t for t in tools if t not in (f"tool.{CALCULATE}", f"tool.{COMPUTE}"))
    rp, v, ev = "reasoning_policy", "verifier", "evidence"
    layers = [
        Layer("system_prompt", "the suite's rules as the system message", f"{rp}.use_system_prompt",
              *_flag(rp, "use_system_prompt")),
        Layer("tool_prompt", "the tool and citation instructions", f"{rp}.use_tool_prompt",
              *_flag(rp, "use_tool_prompt"), needs=(tuple(tools),)),
        *(Layer(f"tool.{name}", "the compute tool and its prompt" if name == COMPUTE else f"the {name} tool",
                f"{rp}.compute_tool" if name == COMPUTE else f"{rp}.tools_off, {rp}.max_search_depth",
                *_tool(name)) for name in cfg.tools),
        Layer("thinking_budget", "a thinking budget per call, scaled by the question",
              f"{rp}.thinking_budget.enabled", *_thinking()),
        Layer("text_tool_calls", "tool calls written as text are run or removed", f"{rp}.text_tool_calls",
              *_flag(rp, "text_tool_calls")),
        Layer("answer_retry", "an empty answer is asked for again", f"{rp}.empty_answer_retry",
              *_flag(rp, "empty_answer_retry")),
        Layer("written_calculations", "a written calculate call is run before the retry",
              f"{rp}.run_written_calculations", *_flag(rp, "run_written_calculations"),
              needs=(("text_tool_calls",), (f"tool.{CALCULATE}",))),
        Layer("weak_hit_filter", "weak search hits are dropped", "retrieval.min_term_share",
              *_level("retrieval", "min_term_share"), needs=((f"tool.{SEARCH}", "research"),)),
        Layer("arithmetic", "a × b = c recomputed and fixed", f"{v}.arithmetic_check",
              *_flag(v, "arithmetic_check"), verifier=True),
        Layer("carry_forward", "a fixed value carried to its later uses", f"{v}.carry_fixes_forward",
              *_flag(v, "carry_fixes_forward"), needs=(("arithmetic",),), verifier=True),
        *(Layer(f"checker.{name}", f"the {name} checker", f"{v}.checkers", *_checker(name), verifier=True)
          for name in CHECKERS),
        Layer("record_check", "numbers and formulas against retrieved records, in code", f"{v}.record_check",
              *_flag(v, "record_check"), needs=(knowledge,), verifier=True),
        Layer("judge", "the claim judge call", f"{v}.claim_judge", *_flag(v, "claim_judge"),
              needs=((*knowledge, "research"),), verifier=True),
        Layer("research", "re-search of unsupported claims and the second judge call", f"{v}.max_research",
              *_level(v, "max_research"), needs=(("judge",),), verifier=True),
        Layer("review", "the consistency review call", f"{v}.consistency_check",
              *_flag(v, "consistency_check"), verifier=True),
        Layer("ledger", "the evidence ledger call and its rules", f"{ev}.enabled", *_flag(ev, "enabled")),
        Layer("writer", "the answer rewritten from the settled ledger", f"{ev}.writer", *_flag(ev, "writer"),
              needs=(("ledger",),)),
        Layer("independent", "one judgment call per listed item", f"{ev}.independent",
              *_flag(ev, "independent"), needs=(("ledger",),)),
        Layer("cite_stripping", "[S#]-like markers removed from the final answer", f"{v}.strip_source_ids",
              *_flag(v, "strip_source_ids")),
    ]
    return layers


def switches(cfg: Config) -> dict[str, bool]:
    """Every layer's switch as the config has it (a verifier layer only with the verifier's master switch)."""
    master = cfg.reasoning_policy.enable_multi_hop_verification
    return {layer.name: layer.get(cfg) and (master or not layer.verifier) for layer in registry(cfg)}


def unmet(on: set[str], reg: list[Layer]) -> dict[str, list[tuple[str, ...]]]:
    """Layer -> the groups of which none is on, for every layer of `on` that lacks something it needs."""
    return {layer.name: missing for layer in reg
            if layer.name in on and (missing := [group for group in layer.needs if not on & set(group)])}


def effective(cfg: Config, knowledge: Knowledge | None = None) -> dict[str, bool]:
    """Every layer with whether it can act: its switch is on and what it needs is on. With `knowledge` (the tools
    as built) also: a tool layer only when that tool is offered, a verifier layer only with an open store."""
    reg = registry(cfg)
    on = {name for name, state in switches(cfg).items() if state}
    if knowledge is not None:
        offered = {f"tool.{t['function']['name']}" for t in knowledge.tools}
        on = {name for name in on if not name.startswith("tool.") or name in offered}
        if not knowledge.available:
            on -= {layer.name for layer in reg if layer.verifier}
    while lacking := unmet(on, reg):  # two layers may hold each other up (judge and research): the largest set stays
        on -= set(lacking)
    return {layer.name: layer.name in on for layer in reg}


def _names(given: list[str], reg: list[Layer], candidates: set[str], what: str) -> list[str]:
    """The layers a list of names stands for: a layer's own name (which must be in `candidates`), or the prefix
    of layers named `<prefix>.<x>` for those of them in `candidates`. Raises ValueError naming the bad ones."""
    known = [layer.name for layer in reg]
    out: list[str] = []
    for name in given:
        members = [k for k in known if k.startswith(f"{name}.")]
        if name in known:
            if name not in candidates:
                raise ValueError(f"layer {name} is already {what}")
            out.append(name)
        elif members:
            if not (group := [m for m in members if m in candidates]):
                raise ValueError(f"every {name} layer is already {what}")
            out += group
        else:
            raise ValueError(f"unknown layer {name!r}; known: {', '.join(known)}")
    return [k for k in known if k in out]


def apply_arm(cfg: Config, base: str = CONFIG, plus: list[str] | None = None,
              minus: list[str] | None = None) -> tuple[str, dict[str, Any]]:
    """Switch layers of `cfg` in place: from the configured suite (`base` config) or from every layer off (`base`
    bare), `plus` on and `minus` off. Returns (the arm's name, its record for the manifest). Raises ValueError
    naming the layers when a name is unknown, a layer already has that state, a layer would be on without what
    it needs, the config holds no value to switch a layer on with, or the resulting config is not valid."""
    reg = registry(cfg)
    have = {name for name, state in switches(cfg).items() if state}
    start = set() if base == BARE else have
    if base == BARE and minus:
        raise ValueError("--minus needs the configured suite: with --mode bare every layer is off already")
    names = {layer.name for layer in reg}
    plus = _names(plus or [], reg, names - start, "on")
    minus = _names(minus or [], reg, start, "off")
    if both := sorted(set(plus) & set(minus)):
        raise ValueError(f"layers both switched on and off: {', '.join(both)}")
    want = (start | set(plus)) - set(minus)
    if lacking := unmet(want, reg):
        raise ValueError("; ".join(
            f"{name} needs " + " and ".join(" or ".join(group) for group in groups) + " (switch it off too, or"
            " switch one of those on)" for name, groups in lacking.items()))
    for layer in reg:
        layer.put(cfg, layer.name in want)
    cfg.reasoning_policy.enable_multi_hop_verification = any(layer.verifier and layer.name in want for layer in reg)
    try:
        type(cfg).model_validate(cfg.model_dump())  # the whole-config checks, as for the config file
    except ValidationError as e:
        raise ValueError("; ".join(err["msg"] for err in e.errors())) from e
    name = base + (f"-plus-{'+'.join(plus)}" if plus else "") + (f"-minus-{'+'.join(minus)}" if minus else "")
    return name, {"base": base, "plus": plus, "minus": minus}


def runtime_problems(cfg: Config, knowledge: Knowledge) -> list[str]:
    """What keeps a layer arm from running as its name says, once the tools are built: a tool layer that is on
    although the store does not offer the tool, a verifier layer that is on without an open store."""
    reg, on = registry(cfg), {name for name, state in switches(cfg).items() if state}
    offered = {f"tool.{t['function']['name']}" for t in knowledge.tools}
    problems = [f"{name} is on but the knowledge store does not offer that tool"
                for name in sorted(on) if name.startswith("tool.") and name not in offered]
    if not knowledge.available:
        problems += [f"{layer.name} is on but no knowledge store is open (the agent runs the verifier only with it)"
                     for layer in reg if layer.verifier and layer.name in on]
    return problems


def table(cfg: Config, bare: bool = False) -> str:
    """Every layer with its state, switch and needs, for `--list-layers`. `bare`: the bare call, which uses no
    layer whatever the config says."""
    reg = registry(cfg)
    acts, state = (dict.fromkeys(switches(cfg), False),) * 2 if bare else (effective(cfg), switches(cfg))
    rows = [("layer", "state", "switch", "needs", "adds")]
    for layer in reg:
        shown = "on" if acts[layer.name] else "on, inert" if state[layer.name] else "off"
        needs = " and ".join(" or ".join(group) for group in layer.needs) + \
                ("; the knowledge store" if layer.verifier or layer.name.startswith("tool.") else "")
        rows.append((layer.name, shown, layer.switch, needs.lstrip("; "), layer.adds))
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    return "\n".join("  ".join([*(cell.ljust(w) for cell, w in zip(r, widths)), r[4]]).rstrip() for r in rows)
