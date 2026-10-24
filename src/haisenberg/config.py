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


class ReasoningPolicyConfig(_Section):
    system_prompt: str
    tool_prompt: str
    enable_multi_hop_verification: bool
    max_search_depth: NonNegativeInt

    @field_validator("system_prompt", "tool_prompt")
    @classmethod
    def _not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v


class RetrievalConfig(_Section):
    raw_dir: Path
    layer1_encyclopedia_db: Path
    layer2_literature_index: Path
    top_k_proofs: PositiveInt
    passage_chars: PositiveInt
    min_term_share: Annotated[float, Field(ge=0, le=1)]  # search hits need this share of the query's words; 0 = off


class ToolText(_Section):
    """Model-facing texts of one agent tool; the tools themselves are generated from the store."""

    description: str
    parameters: dict[str, str] = {}

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
    arithmetic_rel_tol: NonNegativeFloat  # "a × b = c" in the draft is fixed when c is off by more
    arithmetic_prefixes: bool  # c may also be scaled by an SI prefix of its chain (2452 W = 2.45 kW)
    skip_derived_claims: bool  # computations and labelled answer/verdict lines are never judged
    keep_uncorrected: bool  # a "contradicts" verdict without a correction marks the claim unverified, not removed
    research_k: PositiveInt
    max_research: NonNegativeInt
    request: dict[str, Any] = {}  # fields merged into every judge call
    answer_style: Literal["plain", "annotated"]  # plain chat answer, or labels + ids + sources (debugging)
    consistency_check: bool  # one review call over the whole answer and the task (conclusions vs calculations...)
    consistency_prompt: str

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
    ("retrieval", "layer1_encyclopedia_db"),
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
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()
        )
        raise ConfigError(f"{source}: {problems}") from e

    for section, name in _PATH_FIELDS:
        sec = getattr(cfg, section)
        p = getattr(sec, name)
        if not p.is_absolute():
            setattr(sec, name, root / p)
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
