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
)

CONFIG_ENV = "HAISENBERG_CONFIG"
PROFILE_ENV = "MODEL_PROFILE"  # selects config/<name>.yaml, merged over config/config.yaml
_NOT_PROFILES = {"config", "sources"}  # the other YAML files in config/
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
    max_tokens: int
    timeout_s: PositiveFloat | None  # read limit; None = wait as long as the backend needs
    wait_s: NonNegativeFloat  # retry a backend that refuses connections or answers 503 this long
    context_tokens: PositiveInt | None = None  # None = llama_server.ctx_size
    chars_per_token: PositiveFloat  # token estimate for the context budget
    extra_body: dict[str, Any] = {}


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


class ReasoningPolicyConfig(_Section):
    system_prompt: str
    tool_prompt: str
    enable_multi_hop_verification: bool
    max_search_depth: NonNegativeInt
    max_calc_rounds: NonNegativeInt  # rounds of calculator calls only, not counted in max_search_depth; 0 = counted
    text_tool_calls: bool  # a tool call written as text is run (rounds left) or removed and the answer asked again
    tool_call_markup: ToolCallMarkup
    empty_answer_retry: bool  # an empty final answer is asked again once
    answer_retry_prompt: str  # the user turn that asks again for the final answer
    answer_retry_request: dict[str, Any] = {}  # fields merged into that call (e.g. thinking off)

    @field_validator("system_prompt", "tool_prompt", "answer_retry_prompt")
    @classmethod
    def _not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v


class StoreConfig(_Section):
    """A knowledge store: its kind says which ingest script builds it and which code reads it."""

    # Layer 1 (ingest.build_l1), Layer 2 literature (ingest.build_l2) or its vectors (ingest.embed_l2, a folder)
    kind: Literal["encyclopedia", "literature", "literature_dense"]
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


class DenseConfig(_Section):
    """The literature_dense store: how ingest.embed_l2 indexes the vectors and how a hybrid search ranks."""

    index: str  # FAISS index factory, e.g. 'IVF4096,PQ64'
    nprobe: PositiveInt  # index lists searched per query
    rerank: PositiveInt  # index hits re-scored exactly with the stored vectors
    candidates: PositiveInt  # length of the BM25 ranking and of the dense ranking that are fused
    rrf_k: NonNegativeInt  # reciprocal rank fusion: score = sum of 1 / (rrf_k + rank)
    # a passage of the dense ranking at least this similar (cosine) to the query passes min_term_share's filter
    min_similarity: Annotated[float, Field(ge=-1, le=1)]


class RetrievalConfig(_Section):
    raw_dir: Path
    stores: dict[str, StoreConfig]  # every knowledge store by name
    knowledge_sets: dict[str, list[str]]  # knowledge setting -> the stores it enables
    knowledge: str  # the knowledge setting in use (the A/B switch), one of knowledge_sets
    dense: DenseConfig
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
            kinds = {stores[m].kind for m in members}
            if len(kinds) < len(members):
                raise ValueError(f"{name}: at most one store of each kind")
            if "literature_dense" in kinds and "literature" not in kinds:  # the vectors index its passages
                raise ValueError(f"{name}: a literature_dense store needs a store of kind 'literature'")
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
    arithmetic_check: bool  # recompute "a × b = c" in the draft and fix wrong results
    checkers: list[str] = []  # domain checker plug-ins run on every draft (haisenberg.checkers), e.g. element_balance
    checkers_clear_only: bool  # a checker's fix that the checker doubts (maybe not its subject) is only reported
    arithmetic_rel_tol: NonNegativeFloat  # "a × b = c" in the draft is fixed when c is off by more
    arithmetic_prefixes: bool  # c may also be scaled by an SI prefix of its chain (2452 W = 2.45 kW)
    arithmetic_strict_parse: bool  # a × 10ⁿ is one number, only whole left sides, unit conversions skipped
    arithmetic_max_change: NonNegativeFloat  # a fix changing c by this factor or more, or its sign, is only reported
    skip_derived_claims: bool  # computations and labelled answer/verdict lines are never judged
    keep_uncorrected: bool  # a "contradicts" verdict without a correction marks the claim unverified, not removed
    ignore_prompt_echo: bool  # a judge correction repeating the judge's own instructions is ignored
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


class EmbeddingConfig(_Section):
    """The embedding model of the literature_dense store: an OpenAI-compatible embeddings backend and its own
    llama-server, started on demand when the knowledge setting enables that store."""

    api_base: str
    api_key: str
    model_id: str  # sent as `model`, the server's alias; the index records it
    query_prefix: str  # put before every search query, never before a passage
    batch: PositiveInt  # passages per request while building
    workers: PositiveInt  # parallel requests while building
    timeout_s: PositiveFloat
    server: LlamaServerConfig


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
    llama_server: LlamaServerConfig
    embedding: EmbeddingConfig
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


# (section, field) pairs resolved against the project root when relative.
_PATH_FIELDS = [
    ("retrieval", "raw_dir"),
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
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()
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
    if not cfg.embedding.server.model_path.is_absolute():
        cfg.embedding.server.model_path = root / cfg.embedding.server.model_path
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
