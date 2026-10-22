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


PROFILES = sorted((REPO / "config" / "models").glob("*.yaml"))

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
    assert cfg.retrieval.layer1_encyclopedia_db == tmp_path.resolve() / "data" / "encyclopedia_fts.db"


def test_variable_inside_longer_string_and_absolute_path(tmp_path, monkeypatch):
    other = tmp_path / "elsewhere" / "hdata"
    monkeypatch.setenv("DATA_DIR", other.as_posix())
    cfg = load_config(write(tmp_path / "proj"))
    assert cfg.retrieval.layer1_encyclopedia_db == other / "encyclopedia_fts.db"
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
    assert (_arg(cmd, "--gpu-layers"), _arg(cmd, "--threads"), _arg(cmd, "--reasoning-budget")) == ("99", "8", "1024")


def test_machine_knobs_from_environment(tmp_path, monkeypatch):
    from haisenberg.launcher import build_command

    knobs = {"CTX_SIZE": "65536", "GPU_LAYERS": "40", "LLAMA_THREADS": "32", "REASONING_BUDGET": "4096", "LLAMA_PORT": "9090"}
    for k, v in knobs.items():
        monkeypatch.setenv(k, v)
    cmd = build_command(load_config(write(tmp_path)))
    assert _arg(cmd, "--ctx-size") == "65536" and _arg(cmd, "--gpu-layers") == "40"
    assert _arg(cmd, "--threads") == "32" and _arg(cmd, "--reasoning-budget") == "4096"
    assert _arg(cmd, "--port") == "9090"


def write_profile(root: Path, name: str, data: dict) -> None:
    path = root / "config" / "models" / f"{name}.yaml"
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
