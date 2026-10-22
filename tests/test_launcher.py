import sys

import httpx
import pytest

from haisenberg.launcher import LauncherError, build_command, build_env, ensure_llama_server, stop

SLEEPER = [sys.executable, "-c", "import time; time.sleep(60)"]


def health(*statuses):
    """httpx client whose /health answers the given statuses in turn (then repeats the last)."""
    seq = list(statuses)

    def handler(request):
        status = seq.pop(0) if len(seq) > 1 else seq[0]
        if status is None:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(status)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_flags_mapping(cfg):
    cfg.llama_server.flags = {"jinja": True, "reasoning-format": "deepseek", "no-ui": True, "mlock": False, "x": None}
    cmd = build_command(cfg)
    assert cmd[0] == cfg.llama_server.executable
    assert cmd[cmd.index("--model") + 1] == str(cfg.llama_server.model_path)
    assert cmd[cmd.index("--alias") + 1] == cfg.model_backend.model_id
    assert cmd[-4:] == ["--jinja", "--reasoning-format", "deepseek", "--no-ui"]
    assert "--mlock" not in cmd and "--x" not in cmd


def test_vulkan_device_env(cfg):
    cfg.llama_server.env = {"GGML_VK_VISIBLE_DEVICES": "1"}
    assert build_env(cfg)["GGML_VK_VISIBLE_DEVICES"] == "1"


def test_reuse_running_server(cfg):
    cfg.llama_server.model_path = cfg.root / "does-not-exist.gguf"
    assert ensure_llama_server(cfg, http=health(200)) is None


def test_missing_model(cfg):
    cfg.llama_server.model_path = cfg.root / "missing.gguf"
    with pytest.raises(LauncherError, match="MODEL_PATH") as e:
        ensure_llama_server(cfg, http=health(None))
    assert "missing.gguf" in str(e.value)


def test_missing_executable(cfg, tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"")
    cfg.llama_server.model_path = model
    cfg.llama_server.executable = "definitely-not-a-real-llama-server"
    with pytest.raises(LauncherError, match="LLAMA_SERVER_EXE"):
        ensure_llama_server(cfg, http=health(None))


def test_waits_through_loading(cfg):
    proc = ensure_llama_server(cfg, http=health(None, 503, 503, 503, 200), command=SLEEPER, poll_s=0.01)
    try:
        assert proc is not None and proc.poll() is None
    finally:
        stop(proc)
    assert proc.poll() is not None


def test_early_exit_reports_code(cfg):
    with pytest.raises(LauncherError, match="exited with code 1"):
        ensure_llama_server(cfg, http=health(503), command=[sys.executable, "-c", "import sys; sys.exit(1)"], poll_s=0.05)


def test_timeout_stops_process(cfg, monkeypatch):
    cfg.llama_server.startup_timeout_s = 0.3
    spawned = []
    real_popen = __import__("subprocess").Popen

    def recording_popen(*a, **kw):
        spawned.append(real_popen(*a, **kw))
        return spawned[-1]

    monkeypatch.setattr("haisenberg.launcher.subprocess.Popen", recording_popen)
    with pytest.raises(LauncherError, match="not ready"):
        ensure_llama_server(cfg, http=health(503), command=SLEEPER, poll_s=0.05)
    assert spawned[0].poll() is not None
