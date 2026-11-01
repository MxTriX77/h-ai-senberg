"""Load `config/config.yaml` (+ `.env`), interpolate ${VAR} / ${VAR:-default}, validate, resolve paths."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)

CONFIG_ENV = "HAISENBERG_CONFIG"
PROFILE_ENV = "MODEL_PROFILE"  # selects config/<name>.yaml, merged over config/config.yaml
_NOT_PROFILES = {"config", "sources"}  # the other YAML files in config/
CAP_FLAG = "reasoning-budget"  # llama_server.flags: the server's hard cap on thinking tokens
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(Exception):
    """Any problem with the configuration; the message names the file, key or variable."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AppConfig(_Section):
    host: str
    port: int
    debug: bool
    keepalive_s: PositiveFloat  # streamed responses: seconds of silence before an SSE keep-alive comment


class ModelBackendConfig(_Section):
    provider: str
    api_base: str
    api_key: str
    model_id: str
    temperature: float
    seed: NonNegativeInt | None = None  # sent as `seed` when set; None (empty EVAL_SEED) = no seed
    max_tokens: int
    timeout_s: PositiveFloat | None  # read limit; None = wait as long as the backend needs
    wait_s: NonNegativeFloat  # retry a backend that refuses connections or answers 503 this long
    context_tokens: PositiveInt | None = None  # None = llama_server.ctx_size
    chars_per_token: PositiveFloat  # token estimate for the context budget
    extra_body: dict[str, Any] = {}

    @field_validator("seed", mode="before")
    @classmethod
    def _empty_seed(cls, v: Any) -> Any:
        return None if v == "" else v


class ToolCallMarkup(_Section):
    """How chat templates write a tool call as text (regular expressions; `.` also matches a newline)."""

    call: str  # one whole call, also cut off; removed from the answer
    function: str  # the tool name (group 1); a call without it is read as JSON {"name", "arguments"}
    parameter: str  # one argument: name (group 1) and value (group 2)

    @field_validator("call", "function", "parameter")
    @classmethod
    def _regex(cls, v: str) -> str:
        try:
            re.compile(v)
        except re.error as e:
            raise ValueError(f"invalid regular expression: {e}") from None
        return v


class BudgetSignal(_Section):
    """One feature of the question for the thinking budget: level 0 up to `start`, 1 from `full`, linear between."""

    start: NonNegativeFloat
    full: PositiveFloat
    weight: Annotated[float, Field(ge=0, le=1)]  # its part of the share

    @model_validator(mode="after")
    def _range(self) -> BudgetSignal:
        if self.full <= self.start:
            raise ValueError("full must be greater than start")
        return self

    def level(self, count: float) -> float:
        return min(1.0, max(0.0, (count - self.start) / (self.full - self.start)))


class ThinkingBudgetConfig(_Section):
    """Thinking tokens per reasoner call: min_tokens + (max_tokens - min_tokens) x the question's share (0-1)."""

    field: str  # request field the backend reads per call (llama-server: thinking_budget_tokens); "" = send none
    min_tokens: PositiveInt
    max_tokens: PositiveInt
    answer_tokens: PositiveInt  # room for the answer: such a call's max_tokens is its budget plus this
    chars: BudgetSignal  # characters of the question
    numbers: BudgetSignal  # numbers written in it
    items: BudgetSignal  # enumerated parts and requested answer lines (item_pattern matches)
    item_pattern: str  # regex: one match per part or answer line

    @field_validator("item_pattern")
    @classmethod
    def _regex(cls, v: str) -> str:
        try:
            re.compile(v)
        except re.error as e:
            raise ValueError(f"invalid regular expression: {e}") from None
        return v

    @model_validator(mode="after")
    def _bounds(self) -> ThinkingBudgetConfig:
        if self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens must not exceed max_tokens")
        return self


class ReasoningPolicyConfig(_Section):
    system_prompt: str
    tool_prompt: str
    enable_multi_hop_verification: bool
    max_search_depth: NonNegativeInt
    max_calc_rounds: NonNegativeInt  # rounds of calculator calls only, not counted in max_search_depth; 0 = counted
    compute_tool: bool  # offer the compute tool (procedures) next to the calculator, with compute_prompt
    compute_prompt: str  # added after the tool prompt when the compute tool is offered
    text_tool_calls: bool  # a tool call written as text is run (rounds left) or removed and the answer asked again
    tool_call_markup: ToolCallMarkup
    empty_answer_retry: bool  # an empty final answer is asked again once
    run_written_calculations: bool  # a written calculate call that cannot run as a round is run before asking again
    answer_retry_prompt: str  # the user turn that asks again for the final answer
    answer_retry_request: dict[str, Any] = {}  # fields merged into that call (e.g. thinking off)
    thinking_budget: ThinkingBudgetConfig

    @field_validator("system_prompt", "tool_prompt", "compute_prompt", "answer_retry_prompt")
    @classmethod
    def _not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v


class StoreConfig(_Section):
    """A knowledge store: its kind says which ingest script builds it and which code reads it."""

    kind: Literal["encyclopedia", "literature"]  # Layer 1 (ingest.build_l1) or Layer 2 literature (ingest.build_l2)
    path: Path


class SourceLabels(_Section):
    """What a tool result calls a source when a literature store is enabled (retrieval.label_sources)."""

    reference: str  # a Layer 1 record (encyclopedia article, compound, constant)
    literature: str  # a literature passage

    @field_validator("reference", "literature")
    @classmethod
    def _not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v


class RetrievalConfig(_Section):
    raw_dir: Path
    stores: dict[str, StoreConfig]  # every knowledge store by name
    knowledge_sets: dict[str, list[str]]  # knowledge setting -> the stores it enables
    knowledge: str  # the knowledge setting in use (the A/B switch), one of knowledge_sets
    layer2_literature_index: Path
    layer2_chunk_chars: PositiveInt  # longest passage the Layer 2 build stores
    literature_k: PositiveInt  # most literature passages added to one search result when a literature store is enabled
    top_k_proofs: PositiveInt
    passage_chars: PositiveInt
    min_term_share: Annotated[float, Field(ge=0, le=1)]  # search hits need this share of the query's words; 0 = off
    label_sources: bool  # with a literature store enabled: results say what each source is (source_labels)
    source_labels: SourceLabels

    @field_validator("knowledge_sets")
    @classmethod
    def _sets(cls, v: dict[str, list[str]], info) -> dict[str, list[str]]:
        stores = info.data.get("stores", {})
        for name, members in v.items():
            if unknown := [m for m in members if m not in stores]:
                raise ValueError(f"{name}: unknown stores {unknown}; declared: {sorted(stores)}")
            if len({stores[m].kind for m in members}) < len(members):
                raise ValueError(f"{name}: at most one store of each kind")
        return v

    @field_validator("knowledge")
    @classmethod
    def _knowledge(cls, v: str, info) -> str:
        if "knowledge_sets" in info.data and v not in info.data["knowledge_sets"]:
            raise ValueError(f"{v!r} is not one of {sorted(info.data['knowledge_sets'])}")
        return v

    def path(self, kind: str) -> Path:
        """The path of the store of this kind: the enabled one, else the first declared (what a build writes)."""
        found = self.enabled(kind) or next((s.path for s in self.stores.values() if s.kind == kind), None)
        if found is None:
            raise ConfigError(f"retrieval.stores declares no store of kind {kind!r}")
        return found

    def enabled(self, kind: str) -> Path | None:
        """The path of the store of this kind that the knowledge setting enables; None if it enables none."""
        names = self.knowledge_sets[self.knowledge]
        return next((self.stores[n].path for n in names if self.stores[n].kind == kind), None)


class ToolText(_Section):
    """Model-facing texts of one agent tool; the tools themselves are generated from the store."""

    description: str
    parameters: dict[str, str] = {}
    literature: str = ""  # added to the description when a literature store is enabled (retrieval.label_sources)

    @field_validator("description")
    @classmethod
    def _description(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v

    @field_validator("parameters")
    @classmethod
    def _parameters(cls, v: dict[str, str]) -> dict[str, str]:
        if empty := [k for k, text in v.items() if not text.strip()]:
            raise ValueError(f"empty description for {empty}")
        return v


class VerifierConfig(_Section):
    judge_prompt: str
    rel_tol: NonNegativeFloat
    unit_checks: bool  # code checks compare a number with a record value only as the same quantity (unit, property)
    arithmetic_check: bool  # recompute "a × b = c" in the draft and fix wrong results
    checkers: list[str] = []  # domain checker plug-ins run on every draft (haisenberg.checkers), e.g. element_balance
    checkers_clear_only: bool  # a checker's fix that the checker doubts (maybe not its subject) is only reported
    arithmetic_rel_tol: NonNegativeFloat  # "a × b = c" in the draft is fixed when c is off by more
    arithmetic_prefixes: bool  # c may also be scaled by an SI prefix of its chain (2452 W = 2.45 kW)
    arithmetic_answer_lines: bool  # an answer line stating exactly a fixed wrong c gets the fixed value too
    arithmetic_strict_parse: bool  # a × 10ⁿ is one number, only whole left sides, unit conversions skipped
    arithmetic_max_change: NonNegativeFloat  # a fix changing c by this factor or more, or its sign, is only reported
    skip_derived_claims: bool  # computations and labelled answer/verdict lines are never judged
    keep_uncorrected: bool  # a "contradicts" verdict without a correction marks the claim unverified, not removed
    ignore_prompt_echo: bool  # a judge correction repeating the judge's own instructions is ignored
    judge_minimal_edits: bool  # a judge correction is applied only as a minimal edit the cited passages state
    research_k: PositiveInt
    max_research: NonNegativeInt
    request: dict[str, Any] = {}  # fields merged into every judge call
    answer_style: Literal["plain", "annotated"]  # plain chat answer, or labels + ids + sources (debugging)
    consistency_check: bool  # one review call over the whole answer and the task (conclusions vs calculations...)
    consistency_prompt: str
    consistency_quotes: bool  # a review fix needs a quoted conflicting passage of the answer that it agrees with
    consistency_keep_answer_lines: bool  # the review never changes a labelled answer/verdict line
    keep_quotes: bool  # text the answer quotes from the task is never rewritten by checkers, judge or review
    reference_over_literature: bool  # a claim supported only by literature is not supported when a reference disagrees

    @field_validator("checkers")
    @classmethod
    def _known_checkers(cls, v: list[str]) -> list[str]:
        from .checkers import CHECKERS

        if unknown := [n for n in v if n not in CHECKERS]:
            raise ValueError(f"unknown checker {unknown}; available: {sorted(CHECKERS)}")
        return v

    @field_validator("judge_prompt", "consistency_prompt")
    @classmethod
    def _not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v


class EvidenceConfig(_Section):
    """Evidence ledger for requests that list items to judge (D-032); see haisenberg.evidence."""

    enabled: bool
    min_items: PositiveInt  # the last user message lists at least this many items
    strong_words: list[str]  # a statement holding one (at a word start) is a claim of proof
    ledger_prompt: str
    repair_prompt: str  # {missing}: the listed ids without an entry
    writer_prompt: str
    independent: bool  # also one judgment per listed item in its own call, without the draft or the ledger
    independent_max_items: PositiveInt  # at most this many items get one
    item_prompt: str  # {id}, {text}: the item to judge
    ledger_request: dict[str, Any] = {}  # fields merged into the ledger and repair calls
    writer_request: dict[str, Any] = {}  # fields merged into the writer call
    item_request: dict[str, Any] = {}  # fields merged into each independent judgment call

    @field_validator("ledger_prompt", "repair_prompt", "writer_prompt", "item_prompt")
    @classmethod
    def _not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v

    @field_validator("repair_prompt")
    @classmethod
    def _names_missing(cls, v: str) -> str:
        if "{missing}" not in v:
            raise ValueError("must contain {missing}")
        return v

    @field_validator("item_prompt")
    @classmethod
    def _names_the_item(cls, v: str) -> str:
        if "{id}" not in v or "{text}" not in v:
            raise ValueError("must contain {id} and {text}")
        return v


class LlamaServerConfig(_Section):
    enabled: bool
    executable: str
    model_path: Path
    host: str
    port: int
    ctx_size: int
    startup_timeout_s: float
    env: dict[str, str] = {}
    flags: dict[str, Any] = {}


class UiConfig(_Section):
    dist_dir: Path
    settings: dict[str, Any] = {}


class Config(_Section):
    app: AppConfig
    model_backend: ModelBackendConfig
    reasoning_policy: ReasoningPolicyConfig
    retrieval: RetrievalConfig
    tools: dict[str, ToolText]
    verifier: VerifierConfig
    evidence: EvidenceConfig
    llama_server: LlamaServerConfig
    ui: UiConfig

    _root: Path = PrivateAttr()

    @property
    def root(self) -> Path:
        """Project root: the parent of the folder that holds the config file."""
        return self._root

    @property
    def context_tokens(self) -> int:
        """The backend's context size in tokens: model_backend.context_tokens, else llama_server.ctx_size."""
        return self.model_backend.context_tokens or self.llama_server.ctx_size

    @model_validator(mode="after")
    def _thinking_fits(self) -> Config:
        """The largest thinking budget plus the answer room fits the context, and under a local llama-server's cap."""
        tb, server = self.reasoning_policy.thinking_budget, self.llama_server
        if not tb.field:
            return self
        top = f"reasoning_policy.thinking_budget.max_tokens ({tb.max_tokens})"
        if tb.max_tokens + tb.answer_tokens >= self.context_tokens:
            raise ValueError(f"{top} + answer_tokens ({tb.answer_tokens}) leaves no room for the prompt in the"
                             f" context ({self.context_tokens} tokens)")
        cap = server.flags.get(CAP_FLAG)
        if server.enabled and cap is not None and not isinstance(cap, bool) and cap != "":  # passed on by the launcher
            try:
                cap = int(cap)
            except (TypeError, ValueError):
                raise ValueError(f"llama_server.flags.{CAP_FLAG}: {cap!r} is not a whole number") from None
            if 0 <= cap < tb.max_tokens:  # -1: no cap
                raise ValueError(f"{top} exceeds the server cap llama_server.flags.{CAP_FLAG} ({cap})")
        return self


# (section, field) pairs resolved against the project root when relative.
_PATH_FIELDS = [
    ("retrieval", "raw_dir"),
    ("retrieval", "layer2_literature_index"),
    ("llama_server", "model_path"),
    ("ui", "dist_dir"),
]


def load_config(path: Path | str | None = None) -> Config:
    """Load, interpolate and validate the config. Raises ConfigError."""
    if path is None:
        path = os.environ.get(CONFIG_ENV) or Path.cwd() / "config" / "config.yaml"
    path = Path(path).resolve()
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    root = path.parent.parent

    load_dotenv(root / ".env", override=False)

    raw = _read_yaml(path)
    source = str(path)
    profile = os.environ.get(PROFILE_ENV)
    if profile:
        profile_path = path.parent / f"{profile}.yaml"
        if profile in _NOT_PROFILES or not profile_path.is_file():
            available = sorted(p.stem for p in path.parent.glob("*.yaml") if p.stem not in _NOT_PROFILES)
            raise ConfigError(
                f"model profile '{profile}' not found: {profile_path} (available: {', '.join(available) or 'none'})"
            )
        raw = _merge(raw, _read_yaml(profile_path))
        source = f"{path} + {profile_path.name}"

    data = _interpolate(raw, "")
    try:
        cfg = Config.model_validate(data)
    except ValidationError as e:
        problems = "; ".join(  # a whole-config check has no location: its message names the keys
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" if err["loc"] else err["msg"] for err in e.errors()
        )
        raise ConfigError(f"{source}: {problems}") from e

    for section, name in _PATH_FIELDS:
        sec = getattr(cfg, section)
        p = getattr(sec, name)
        if not p.is_absolute():
            setattr(sec, name, root / p)
    for store in cfg.retrieval.stores.values():
        if not store.path.is_absolute():
            store.path = root / store.path
    cfg._root = root
    return cfg


def _read_yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: invalid YAML: {e}") from e


def _merge(base: dict, overlay: dict) -> dict:
    """Overlay wins; nested sections merge key by key, other values are replaced."""
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _interpolate(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return {k: _interpolate(v, f"{key}.{k}" if key else str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v, f"{key}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, str):
        return _VAR.sub(lambda m: _substitute(m, key), value)
    return value


def _substitute(m: re.Match[str], key: str) -> str:
    name, default = m.group(1), m.group(2)
    val = os.environ.get(name)
    if default is not None:
        return val if val else default
    if val is None:
        raise ConfigError(f"environment variable {name} is not set (used by {key})")
    return val
