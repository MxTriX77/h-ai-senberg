import os
import re
import shutil
from pathlib import Path

import pytest
import yaml

from haisenberg.config import CONFIG_ENV, PROFILE_ENV, ConfigError, load_config

REPO = Path(__file__).resolve().parents[1]
SHIPPED = REPO / "config" / "config.yaml"


def _strings(v):
    if isinstance(v, dict):
        for x in v.values():
            yield from _strings(x)
    elif isinstance(v, str):
        yield v


PROFILES = sorted(p for p in (REPO / "config").glob("*.yaml") if p.stem not in {"config", "sources"})

# Variables referenced in config and profile values (comments excluded).
VARS = sorted(
    {
        name
        for f in [SHIPPED, *PROFILES]
        for s in _strings(yaml.safe_load(f.read_text(encoding="utf-8")))
        for name in re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)", s)
    }
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Isolate every test: no config vars set, and undo whatever load_dotenv adds."""
    saved = dict(os.environ)
    for name in VARS + [CONFIG_ENV, PROFILE_ENV]:
        monkeypatch.delenv(name, raising=False)
    yield
    os.environ.clear()
    os.environ.update(saved)


def base() -> dict:
    return yaml.safe_load(SHIPPED.read_text(encoding="utf-8"))


def write(root: Path, data: dict | None = None, env: str | None = None) -> Path:
    path = root / "config" / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data if data is not None else base()), encoding="utf-8")
    if env is not None:
        (root / ".env").write_text(env, encoding="utf-8")
    return path


def test_default_location(tmp_path, monkeypatch):
    write(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert load_config().root == tmp_path.resolve()


def test_override_by_environment(tmp_path, monkeypatch):
    alt = tmp_path / "alt"
    monkeypatch.setenv(CONFIG_ENV, str(write(alt)))
    monkeypatch.chdir(tmp_path)
    assert load_config().root == alt.resolve()


def test_missing_file(tmp_path):
    missing = tmp_path / "config" / "config.yaml"
    with pytest.raises(ConfigError, match=re.escape(str(missing.resolve()))):
        load_config(missing)


def test_real_env_beats_dotenv(tmp_path, monkeypatch):
    path = write(tmp_path, env="API_KEY=a\n")
    monkeypatch.setenv("API_KEY", "b")
    assert load_config(path).model_backend.api_key == "b"


def test_dotenv_used_when_env_unset(tmp_path):
    path = write(tmp_path, env="API_KEY=a\n")
    assert load_config(path).model_backend.api_key == "a"


def test_no_dotenv_file(tmp_path):
    assert load_config(write(tmp_path)).app.port == 8000


def test_default_used(tmp_path):
    cfg = load_config(write(tmp_path))
    assert cfg.retrieval.path("encyclopedia") == tmp_path.resolve() / "data" / "encyclopedia_fts.db"
    assert cfg.retrieval.path("literature") == tmp_path.resolve() / "data" / "literature_fts.db"


def test_variable_inside_longer_string_and_absolute_path(tmp_path, monkeypatch):
    other = tmp_path / "elsewhere" / "hdata"
    monkeypatch.setenv("DATA_DIR", other.as_posix())
    cfg = load_config(write(tmp_path / "proj"))
    assert cfg.retrieval.path("encyclopedia") == other / "encyclopedia_fts.db"
    assert cfg.retrieval.path("literature") == other / "literature_fts.db"
    assert cfg.retrieval.raw_dir == other / "raw"


def test_raw_dir_default(tmp_path):
    cfg = load_config(write(tmp_path))
    assert cfg.retrieval.raw_dir == tmp_path.resolve() / "data" / "raw"


def test_top_k_proofs_must_be_positive(tmp_path):
    data = base()
    data["retrieval"]["top_k_proofs"] = 0
    with pytest.raises(ConfigError, match=r"retrieval\.top_k_proofs"):
        load_config(write(tmp_path, data))


def test_required_variable_missing(tmp_path):
    data = base()
    data["model_backend"]["api_base"] = "${API_BASE_URL}"
    with pytest.raises(ConfigError) as e:
        load_config(write(tmp_path, data))
    assert "API_BASE_URL" in str(e.value) and "model_backend.api_base" in str(e.value)


def test_unknown_key_named(tmp_path):
    data = base()
    data["app"]["debgu"] = True
    with pytest.raises(ConfigError, match=r"app\.debgu"):
        load_config(write(tmp_path, data))


def test_wrong_type_named(tmp_path):
    data = base()
    data["llama_server"]["port"] = "eighty"
    with pytest.raises(ConfigError, match=r"llama_server\.port"):
        load_config(write(tmp_path, data))


def test_strings_coerced(tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMA_SERVER_ENABLED", "false")
    cfg = load_config(write(tmp_path))
    assert cfg.app.port == 8000 and isinstance(cfg.app.port, int)
    assert cfg.llama_server.enabled is False


def test_relative_model_path(tmp_path):
    cfg = load_config(write(tmp_path))
    assert cfg.llama_server.model_path == tmp_path.resolve() / "models" / "Qwen3.5-9B-Q4_K_M.gguf"


def test_empty_system_prompt(tmp_path):
    data = base()
    data["reasoning_policy"]["system_prompt"] = "  \n"
    with pytest.raises(ConfigError, match=r"reasoning_policy\.system_prompt"):
        load_config(write(tmp_path, data))


def test_shipped_config_fresh_clone(tmp_path):
    dest = tmp_path / "config" / "config.yaml"
    dest.parent.mkdir(parents=True)
    shutil.copy(SHIPPED, dest)
    cfg = load_config(dest)
    assert cfg.model_backend.api_base.startswith("http://127.0.0.1:")
    assert cfg.model_backend.api_key == ""
    assert cfg.reasoning_policy.system_prompt.strip()


def test_every_variable_documented():
    text = (REPO / ".env.example").read_text(encoding="utf-8")
    documented = set(re.findall(r"^#?\s*([A-Z_][A-Z0-9_]*)=", text, re.M))
    assert set(VARS) <= documented, set(VARS) - documented


def _arg(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_machine_knob_defaults_unchanged(tmp_path):
    from haisenberg.launcher import build_command

    cmd = build_command(load_config(write(tmp_path)))
    assert (_arg(cmd, "--host"), _arg(cmd, "--port"), _arg(cmd, "--ctx-size")) == ("127.0.0.1", "8080", "16384")
    assert (_arg(cmd, "--gpu-layers"), _arg(cmd, "--threads"), _arg(cmd, "--reasoning-budget")) == ("99", "8", "4096")


def test_machine_knobs_from_environment(tmp_path, monkeypatch):
    from haisenberg.launcher import build_command

    knobs = {"CTX_SIZE": "65536", "GPU_LAYERS": "40", "LLAMA_THREADS": "32", "REASONING_BUDGET": "2048", "LLAMA_PORT": "9090"}
    for k, v in knobs.items():
        monkeypatch.setenv(k, v)
    cfg = load_config(write(tmp_path))
    cmd = build_command(cfg)
    assert _arg(cmd, "--ctx-size") == "65536" and _arg(cmd, "--gpu-layers") == "40"
    assert _arg(cmd, "--threads") == "32" and _arg(cmd, "--reasoning-budget") == "2048"
    assert cfg.reasoning_policy.thinking_budget.max_tokens == 2048  # the cap is also the top of the budget
    assert _arg(cmd, "--port") == "9090"


def write_profile(root: Path, name: str, data: dict) -> None:
    path = root / "config" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_profile_overlays_model_settings(tmp_path, monkeypatch):
    path = write(tmp_path)
    write_profile(tmp_path, "big", {
        "model_backend": {"model_id": "big-model", "extra_body": {"top_k": 40}},
        "llama_server": {"model_path": "${MODEL_PATH:-models/big.gguf}", "flags": {"n-cpu-moe": 20}},
    })
    monkeypatch.setenv(PROFILE_ENV, "big")
    cfg = load_config(path)
    assert cfg.model_backend.model_id == "big-model"
    assert cfg.model_backend.extra_body == {**base()["model_backend"]["extra_body"], "top_k": 40}
    assert cfg.llama_server.model_path == tmp_path.resolve() / "models" / "big.gguf"
    assert cfg.llama_server.flags["n-cpu-moe"] == 20 and cfg.llama_server.flags["jinja"] is True
    assert cfg.model_backend.temperature == base()["model_backend"]["temperature"]


def test_no_profile_keeps_base(tmp_path):
    write_profile(tmp_path, "big", {"model_backend": {"model_id": "big-model"}})
    assert load_config(write(tmp_path)).model_backend.model_id != "big-model"


def test_missing_profile_lists_available(tmp_path, monkeypatch):
    path = write(tmp_path)
    write_profile(tmp_path, "one", {})
    monkeypatch.setenv(PROFILE_ENV, "two")
    with pytest.raises(ConfigError, match=r"'two' not found.*available: one"):
        load_config(path)


def test_profile_typo_named(tmp_path, monkeypatch):
    path = write(tmp_path)
    write_profile(tmp_path, "bad", {"model_backend": {"modle_id": "x"}})
    monkeypatch.setenv(PROFILE_ENV, "bad")
    with pytest.raises(ConfigError, match=r"bad\.yaml.*model_backend\.modle_id"):
        load_config(path)


def test_shipped_profiles_load(tmp_path, monkeypatch):
    for profile in PROFILES:
        root = tmp_path / profile.stem
        path = write(root)
        write_profile(root, profile.stem, yaml.safe_load(profile.read_text(encoding="utf-8")))
        monkeypatch.setenv(PROFILE_ENV, profile.stem)
        assert load_config(path).model_backend.model_id


@pytest.mark.parametrize("section, key, value, named", [
    ("reasoning_policy", "max_search_depth", -1, r"reasoning_policy\.max_search_depth"),
    ("reasoning_policy", "tool_prompt", None, r"reasoning_policy\.tool_prompt"),
    ("reasoning_policy", "tool_prompt", " ", r"reasoning_policy\.tool_prompt"),
    ("reasoning_policy", "max_calc_rounds", -1, r"reasoning_policy\.max_calc_rounds"),
    ("reasoning_policy", "answer_retry_prompt", " ", r"reasoning_policy\.answer_retry_prompt"),
    ("reasoning_policy", "text_tool_calls", None, r"reasoning_policy\.text_tool_calls"),
    ("reasoning_policy", "run_written_calculations", None, r"reasoning_policy\.run_written_calculations"),
    ("retrieval", "passage_chars", 0, r"retrieval\.passage_chars"),
])
def test_tool_and_search_settings_validated(tmp_path, section, key, value, named):
    data = base()
    if value is None:
        del data[section][key]
    else:
        data[section][key] = value
    with pytest.raises(ConfigError, match=named):
        load_config(write(tmp_path, data))


@pytest.mark.parametrize("change, named", [
    ({"coverage_repair_prompt": " "}, r"reasoning_policy\.coverage_repair_prompt"),
    ({"coverage_repair_prompt": "Write the missing lines."}, r"reasoning_policy\.coverage_repair_prompt: .*\{lines\}"),
    ({"coverage_repair": None}, r"reasoning_policy\.coverage_repair"),
])
def test_coverage_repair_settings(tmp_path, change, named):
    p = load_config(write(tmp_path)).reasoning_policy
    assert p.coverage_repair and "{lines}" in p.coverage_repair_prompt
    data = base()
    for key, value in change.items():
        if value is None:
            del data["reasoning_policy"][key]
        else:
            data["reasoning_policy"][key] = value
    with pytest.raises(ConfigError, match=named):
        load_config(write(tmp_path, data))


def test_tool_texts(tmp_path):
    cfg = load_config(write(tmp_path))
    assert set(cfg.tools) == {"search_knowledge", "lookup_compound", "lookup_constant", "calculate"}
    assert set(cfg.tools["calculate"].parameters) == {"expressions"}
    assert cfg.verifier.arithmetic_rel_tol == 0.005
    assert set(cfg.tools["search_knowledge"].parameters) == {"query", "domain"}
    data = base()
    data["tools"]["search_knowledge"]["description"] = ""
    with pytest.raises(ConfigError, match=r"tools\.search_knowledge\.description"):
        load_config(write(tmp_path, data))


def test_answer_robustness_settings(tmp_path):
    p = load_config(write(tmp_path)).reasoning_policy
    assert p.max_calc_rounds == 1 and p.text_tool_calls and p.empty_answer_retry and p.run_written_calculations
    assert p.answer_retry_request == {"chat_template_kwargs": {"enable_thinking": False}}
    data = base()
    data["reasoning_policy"]["tool_call_markup"]["call"] = "<tool_call>("
    with pytest.raises(ConfigError, match=r"reasoning_policy\.tool_call_markup\.call: .*invalid regular expression"):
        load_config(write(tmp_path, data))


def test_config_and_sources_are_not_profiles(tmp_path, monkeypatch):
    path = write(tmp_path)
    write_profile(tmp_path, "big", {})
    monkeypatch.setenv(PROFILE_ENV, "config")
    with pytest.raises(ConfigError, match=r"'config' not found.*\(available: big\)$"):
        load_config(path)


def test_verifier_settings(tmp_path):
    cfg = load_config(write(tmp_path))
    v = cfg.verifier
    assert v.judge_prompt.strip() and v.rel_tol >= 0 and v.research_k > 0 and v.max_research >= 0
    assert v.request["temperature"] == 0
    data = base()
    data["verifier"]["judge_prompt"] = " "
    with pytest.raises(ConfigError, match=r"verifier\.judge_prompt"):
        load_config(write(tmp_path, data))


def test_answer_style(tmp_path):
    assert load_config(write(tmp_path)).verifier.answer_style == "plain"
    data = base()
    data["verifier"]["answer_style"] = "fancy"
    with pytest.raises(ConfigError, match=r"verifier\.answer_style"):
        load_config(write(tmp_path, data))


def test_robustness_settings(tmp_path):
    cfg = load_config(write(tmp_path))
    b = cfg.model_backend
    assert (cfg.app.keepalive_s, b.timeout_s, b.wait_s, b.context_tokens, b.chars_per_token) == (1, None, 300, None, 2.0)
    assert cfg.context_tokens == cfg.llama_server.ctx_size == 16384 and b.max_tokens == 4096
    data = base()
    data["app"]["keepalive_s"] = 0
    with pytest.raises(ConfigError, match=r"app\.keepalive_s"):
        load_config(write(tmp_path, data))


def test_big_profile_budget(tmp_path, monkeypatch):
    root = tmp_path / "p"
    path = write(root)
    profile = REPO / "config" / "qwen3.5-35b-a3b-heretic.yaml"
    write_profile(root, profile.stem, yaml.safe_load(profile.read_text(encoding="utf-8")))
    monkeypatch.setenv(PROFILE_ENV, profile.stem)
    cfg = load_config(path)
    assert (cfg.context_tokens, cfg.model_backend.max_tokens) == (32768, 8192)
    tb = cfg.reasoning_policy.thinking_budget
    assert (tb.min_tokens, tb.max_tokens, tb.answer_tokens) == (4096, 8192, 4096)
    assert cfg.llama_server.flags["reasoning-budget"] == "8192"


def test_consistency_settings(tmp_path):
    v = load_config(write(tmp_path)).verifier
    assert v.consistency_check is True and v.consistency_prompt.strip()
    data = base()
    data["verifier"]["consistency_prompt"] = " "
    with pytest.raises(ConfigError, match="verifier.consistency_prompt"):
        load_config(write(tmp_path, data))


def test_conservative_check_settings(tmp_path):
    v = load_config(write(tmp_path)).verifier
    assert v.arithmetic_prefixes is v.skip_derived_claims is v.keep_uncorrected is v.arithmetic_strict_parse is True
    assert v.arithmetic_max_change == 100
    assert v.unit_checks is v.arithmetic_answer_lines is True
    for key in ("keep_uncorrected", "unit_checks", "arithmetic_answer_lines", "arithmetic_strict_parse", "arithmetic_max_change"):
        data = base()
        del data["verifier"][key]
        with pytest.raises(ConfigError, match=f"verifier.{key}"):
            load_config(write(tmp_path, data))
    data = base()
    data["verifier"]["arithmetic_max_change"] = -1
    with pytest.raises(ConfigError, match="verifier.arithmetic_max_change"):
        load_config(write(tmp_path, data))


def test_conservative_review_settings(tmp_path):
    v = load_config(write(tmp_path)).verifier
    assert v.consistency_quotes is v.ignore_prompt_echo is v.consistency_keep_answer_lines is v.keep_quotes is True
    assert v.judge_minimal_edits is True
    for key in ("consistency_quotes", "ignore_prompt_echo", "consistency_keep_answer_lines", "keep_quotes",
                "judge_minimal_edits"):
        data = base()
        del data["verifier"][key]
        with pytest.raises(ConfigError, match=f"verifier.{key}"):
            load_config(write(tmp_path, data))


def test_relevance_threshold(tmp_path):
    assert load_config(write(tmp_path)).retrieval.min_term_share == 0.5
    data = base()
    data["retrieval"]["min_term_share"] = 1.5
    with pytest.raises(ConfigError, match="retrieval.min_term_share"):
        load_config(write(tmp_path, data))


def test_checker_plugins(tmp_path):
    v = load_config(write(tmp_path)).verifier
    assert v.checkers == ["element_balance"] and v.checkers_clear_only is True
    data = base()
    data["verifier"]["checkers"] = ["astrology"]
    with pytest.raises(ConfigError, match="verifier.checkers"):
        load_config(write(tmp_path, data))
    data = base()
    del data["verifier"]["checkers_clear_only"]
    with pytest.raises(ConfigError, match="verifier.checkers_clear_only"):
        load_config(write(tmp_path, data))


def test_knowledge_switch_by_environment(monkeypatch):
    monkeypatch.delenv("KNOWLEDGE", raising=False)
    base = load_config(SHIPPED)
    assert base.retrieval.knowledge == "base"
    monkeypatch.setenv("KNOWLEDGE", "extended")
    extended = load_config(SHIPPED)
    assert extended.retrieval.knowledge == "extended"
    assert extended.model_copy(update={"retrieval": base.retrieval}).model_dump() == base.model_dump()  # nothing else
    monkeypatch.setenv("KNOWLEDGE", "everything")
    with pytest.raises(ConfigError, match="retrieval.knowledge"):
        load_config(SHIPPED)


def test_sampling_seed_from_environment(tmp_path, monkeypatch):
    path = write(tmp_path)
    assert load_config(path).model_backend.seed is None  # unset
    monkeypatch.setenv("EVAL_SEED", "")
    assert load_config(path).model_backend.seed is None
    monkeypatch.setenv("EVAL_SEED", "7")
    assert load_config(path).model_backend.seed == 7
    for bad in ("-1", "1.5", "abc"):
        monkeypatch.setenv("EVAL_SEED", bad)
        with pytest.raises(ConfigError, match="model_backend.seed"):
            load_config(path)


@pytest.mark.parametrize("change, error", [
    ({"knowledge_sets": {"base": ["encyclopedia", "nope"]}}, "unknown stores"),
    ({"stores": {"encyclopedia": {"kind": "encyclopedia", "path": "a.db"}, "other": {"kind": "encyclopedia", "path": "b.db"}},
      "knowledge_sets": {"base": ["encyclopedia", "other"]}}, "one store of each kind"),
    ({"stores": {"encyclopedia": {"kind": "graph", "path": "a.db"}}}, "retrieval.stores"),
])
def test_store_registry_validated(tmp_path, change, error):
    raw = yaml.safe_load(SHIPPED.read_text(encoding="utf-8"))
    raw["retrieval"].update(change)
    if "stores" in change and "knowledge_sets" not in change:
        raw["retrieval"]["knowledge_sets"] = {"base": ["encyclopedia"]}
    raw["retrieval"]["knowledge"] = "base"
    with pytest.raises(ConfigError, match=error):
        load_config(write(tmp_path, raw))


def test_enabled_stores_follow_the_knowledge_setting(monkeypatch):
    monkeypatch.delenv("KNOWLEDGE", raising=False)
    r = load_config(SHIPPED).retrieval
    assert {s.kind for s in r.stores.values()} == {"encyclopedia", "literature"}
    assert r.enabled("encyclopedia") == r.path("encyclopedia") and r.enabled("literature") is None
    monkeypatch.setenv("KNOWLEDGE", "extended")
    r = load_config(SHIPPED).retrieval
    assert r.enabled("literature") == r.path("literature") == r.stores["literature"].path


def test_source_trust_settings(tmp_path):
    cfg = load_config(write(tmp_path))
    r, v = cfg.retrieval, cfg.verifier
    assert r.label_sources is v.reference_over_literature is True
    assert r.source_labels.reference.strip() and r.source_labels.literature.strip()
    assert cfg.tools["search_knowledge"].literature.strip() and cfg.tools["lookup_compound"].literature == ""
    data = base()
    data["retrieval"]["source_labels"]["literature"] = " "
    with pytest.raises(ConfigError, match="retrieval.source_labels.literature"):
        load_config(write(tmp_path, data))
    for section, key in (("retrieval", "label_sources"), ("verifier", "reference_over_literature")):
        data = base()
        del data[section][key]
        with pytest.raises(ConfigError, match=f"{section}.{key}"):
            load_config(write(tmp_path, data))


def test_thinking_budget_settings(tmp_path):
    cfg = load_config(write(tmp_path))
    tb = cfg.reasoning_policy.thinking_budget
    assert (tb.field, tb.min_tokens, tb.max_tokens, tb.answer_tokens) == ("thinking_budget_tokens", 1024, 4096, 3072)
    assert (tb.chars.weight + tb.numbers.weight + tb.items.weight) == pytest.approx(1)  # share 1 when all are full
    assert cfg.llama_server.flags["reasoning-budget"] == "4096" and cfg.model_backend.max_tokens == 4096


@pytest.mark.parametrize("change, error", [
    ({"answer_tokens": 0}, r"reasoning_policy\.thinking_budget\.answer_tokens"),
    ({"min_tokens": 4096, "max_tokens": 1024}, r"reasoning_policy\.thinking_budget: .*min_tokens must not exceed max_tokens"),
    ({"chars": {"start": 2000, "full": 200, "weight": 0.4}}, r"reasoning_policy\.thinking_budget\.chars: .*full"),
    ({"items": {"start": 1, "full": 6, "weight": 1.5}}, r"reasoning_policy\.thinking_budget\.items\.weight"),
    ({"item_pattern": "("}, r"reasoning_policy\.thinking_budget\.item_pattern: .*regular expression"),
    ({"max_tokens": 6144}, r"reasoning_policy\.thinking_budget\.max_tokens \(6144\) exceeds the server cap "
                           r"llama_server\.flags\.reasoning-budget \(4096\)"),
])
def test_thinking_budget_validated(tmp_path, change, error):
    data = base()
    data["reasoning_policy"]["thinking_budget"].update(change)
    with pytest.raises(ConfigError, match=error):
        load_config(write(tmp_path, data))


def test_thinking_budget_fits_the_context(tmp_path, monkeypatch):
    monkeypatch.setenv("CTX_SIZE", "7168")  # 4096 + 3072: no room left for the prompt
    with pytest.raises(ConfigError, match=r"thinking_budget\.max_tokens \(4096\) \+ answer_tokens \(3072\) leaves no "
                                          r"room for the prompt in the context \(7168 tokens\)"):
        load_config(write(tmp_path))
    monkeypatch.setenv("CTX_SIZE", "7169")
    assert load_config(write(tmp_path)).context_tokens == 7169
    monkeypatch.delenv("CTX_SIZE")
    data = base()
    data["llama_server"]["enabled"] = False  # a remote backend: its context counts, not llama-server's cap
    data["reasoning_policy"]["thinking_budget"]["max_tokens"] = 6144
    assert load_config(write(tmp_path, data)).reasoning_policy.thinking_budget.max_tokens == 6144


def test_fixed_thinking_budget_is_the_a_arm(tmp_path):
    """The A arm on the B arm's server (cap 4096): min = max = 1024, so every answer gets today's 1024 + 3072."""
    data = base()
    data["reasoning_policy"]["thinking_budget"]["max_tokens"] = 1024
    tb = load_config(write(tmp_path, data)).reasoning_policy.thinking_budget
    assert tb.min_tokens == tb.max_tokens == 1024 and tb.field


def test_thinking_budget_off_is_the_old_setup(tmp_path, monkeypatch):
    """No field and the old server cap: the bounds are then unused and not checked."""
    from haisenberg.launcher import build_command

    monkeypatch.setenv("REASONING_BUDGET", "1024")
    data = base()
    data["reasoning_policy"]["thinking_budget"]["field"] = ""
    cfg = load_config(write(tmp_path, data))
    assert _arg(build_command(cfg), "--reasoning-budget") == "1024" and cfg.model_backend.max_tokens == 4096
    data["reasoning_policy"]["thinking_budget"]["max_tokens"] = 99999  # above the cap and the context
    assert load_config(write(tmp_path, data)).reasoning_policy.thinking_budget.field == ""
