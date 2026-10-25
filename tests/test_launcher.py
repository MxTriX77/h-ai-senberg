import logging
import os
import signal
import subprocess
import sys
import time

import httpx
import pytest

from haisenberg.launcher import (
    LauncherError,
    Supervisor,
    build_command,
    build_env,
    describe_exit,
    embedding_view,
    ensure_llama_server,
    health_url,
    stop,
)

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
    cfg.llama_server.flags = {
        "jinja": True, "reasoning-format": "deepseek", "no-ui": True, "mlock": False, "x": None, "log-file": "",
    }
    cmd = build_command(cfg)
    assert cmd[0] == cfg.llama_server.executable
    assert cmd[cmd.index("--model") + 1] == str(cfg.llama_server.model_path)
    assert cmd[cmd.index("--alias") + 1] == cfg.model_backend.model_id
    assert cmd[-4:] == ["--jinja", "--reasoning-format", "deepseek", "--no-ui"]
    assert "--mlock" not in cmd and "--x" not in cmd and "--log-file" not in cmd


def test_vulkan_device_env(cfg):
    cfg.llama_server.env = {"GGML_VK_VISIBLE_DEVICES": "1"}
    assert build_env(cfg)["GGML_VK_VISIBLE_DEVICES"] == "1"


def test_embedding_server_command(cfg):
    view = embedding_view(cfg)
    cmd, e = build_command(view), cfg.embedding.server
    assert cmd[cmd.index("--model") + 1] == str(e.model_path)
    assert cmd[cmd.index("--alias") + 1] == cfg.embedding.model_id
    assert (cmd[cmd.index("--port") + 1], cmd[cmd.index("--device") + 1]) == (str(e.port), "none")
    assert "--embedding" in cmd and cmd[cmd.index("--pooling") + 1] == "cls"
    assert build_env(view)["CUDA_VISIBLE_DEVICES"] == ""  # no CUDA context: the reasoner keeps the GPU
    assert health_url(view) == f"http://{e.host}:{e.port}/health"
    assert build_command(cfg)[build_command(cfg).index("--alias") + 1] == cfg.model_backend.model_id  # cfg untouched


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


CHILD = """
import sys, time
from haisenberg.launcher import spawn_tied
p = spawn_tied([sys.executable, "-c", "import time; time.sleep(60)"], None)
print(p.pid, flush=True)
time.sleep(60)
"""


def _gone_within(pid: int, seconds: float) -> bool:
    if os.name == "nt":
        import ctypes

        k32 = ctypes.WinDLL("kernel32")
        handle = k32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not handle:
            return True
        try:
            return k32.WaitForSingleObject(handle, int(seconds * 1000)) == 0
        finally:
            k32.CloseHandle(handle)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.1)
    return False


def test_llama_server_dies_when_the_app_is_killed():
    app = subprocess.Popen([sys.executable, "-c", CHILD], stdout=subprocess.PIPE, text=True)
    child_pid = int(app.stdout.readline())
    app.kill()  # hard kill: no cleanup code runs in the app
    app.wait()
    assert _gone_within(child_pid, 10)


def wait_for(condition, seconds=10.0):
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.02)
    return condition()


def test_supervisor_restarts_a_dead_server(cfg, caplog):
    started = []

    def start():
        started.append(subprocess.Popen(SLEEPER))
        return started[-1]

    sup = Supervisor(cfg, start=start, pause_s=0.01)
    with caplog.at_level(logging.INFO):
        sup.start()
        try:
            started[0].kill()  # a crash or an OOM kill
            assert wait_for(lambda: sup.restarts == 1 and len(started) == 2)
            assert sup.proc is started[1] and started[1].poll() is None
        finally:
            sup.stop()
    assert started[1].poll() is not None and len(started) == 2  # stopped once, not restarted
    expected = "signal SIGKILL" if os.name != "nt" else "code 1"
    assert f"llama-server exited with {expected}" in caplog.text


def test_supervisor_retries_a_failed_restart(cfg):
    calls = []

    def start():
        calls.append(1)
        if len(calls) == 2:
            raise LauncherError("still broken")
        return subprocess.Popen([sys.executable, "-c", "pass"] if len(calls) == 1 else SLEEPER)

    sup = Supervisor(cfg, start=start, pause_s=0.01)
    sup.start()
    try:
        assert wait_for(lambda: len(calls) == 3 and sup.proc is not None and sup.proc.poll() is None)
    finally:
        sup.stop()


def test_supervisor_does_not_watch_a_reused_server(cfg):
    sup = Supervisor(cfg, start=lambda: None)
    sup.start()
    assert sup._thread is None
    sup.stop()


def test_describe_exit():
    assert describe_exit(1) == "code 1" and describe_exit(-signal.SIGTERM) == "signal SIGTERM"
