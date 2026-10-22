"""Start (or reuse) the local llama-server from config, wait until it is ready, stop it on exit."""

from __future__ import annotations

import ctypes
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

from .config import Config

log = logging.getLogger(__name__)


class LauncherError(Exception):
    """llama-server could not be started; the message names the cause and the setting to fix."""


def build_command(cfg: Config) -> list[str]:
    ls = cfg.llama_server
    cmd = [
        ls.executable,
        "--model", str(ls.model_path),
        "--host", ls.host,
        "--port", str(ls.port),
        "--ctx-size", str(ls.ctx_size),
        "--alias", cfg.model_backend.model_id,
    ]
    for name, value in ls.flags.items():
        if value is True:
            cmd.append(f"--{name}")
        elif value is not False and value is not None:
            cmd += [f"--{name}", str(value)]
    return cmd


def build_env(cfg: Config) -> dict[str, str]:
    return {**os.environ, **cfg.llama_server.env}


def health_url(cfg: Config) -> str:
    ls = cfg.llama_server
    return f"http://{ls.host}:{ls.port}/health"


def ensure_llama_server(
    cfg: Config,
    http: httpx.Client | None = None,
    command: list[str] | None = None,
    poll_s: float = 1.0,
) -> subprocess.Popen | None:
    """Return the spawned process, or None when a healthy llama-server was already running."""
    http = http or httpx.Client(timeout=2.0)
    url = health_url(cfg)
    if _healthy(http, url):
        log.info("reusing the llama-server already running at %s", url)
        return None

    ls = cfg.llama_server
    if command is None:
        if not ls.model_path.is_file():
            raise LauncherError(f"model file not found: {ls.model_path} (set MODEL_PATH or llama_server.model_path)")
        if not (shutil.which(ls.executable) or Path(ls.executable).is_file()):
            raise LauncherError(
                f"llama-server not found: {ls.executable} (set LLAMA_SERVER_EXE or llama_server.executable)"
            )
        command = build_command(cfg)

    log.info("starting llama-server: %s", " ".join(command))
    proc = spawn_tied(command, build_env(cfg))  # console inherited: load progress shows here
    deadline = time.monotonic() + ls.startup_timeout_s
    while True:
        code = proc.poll()
        if code is not None:
            raise LauncherError(f"llama-server exited with code {code} while loading")
        if _healthy(http, url):
            log.info("llama-server is ready at %s", url)
            return proc
        if time.monotonic() > deadline:
            stop(proc)
            raise LauncherError(f"llama-server was not ready after {ls.startup_timeout_s:g} s")
        time.sleep(poll_s)


def stop(proc: subprocess.Popen | None) -> None:
    """Terminate a process this launcher started; kill it if it does not exit within 10 s."""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def spawn_tied(command: list[str], env: dict[str, str] | None) -> subprocess.Popen:
    """Start a process that the OS stops when this process exits, even if it is killed
    and `stop()` never runs (for example a hard kill or a closed terminal)."""
    if sys.platform.startswith("linux"):
        # Own session: Ctrl+C reaches only the app, which then stops this process once.
        return subprocess.Popen(command, env=env, preexec_fn=_die_with_parent, start_new_session=True)
    proc = subprocess.Popen(command, env=env)
    if os.name == "nt":
        _kill_on_exit(proc)
    return proc


def _die_with_parent() -> None:
    # Runs in the child before exec: SIGTERM when the parent dies (PR_SET_PDEATHSIG = 1).
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("basic", _BasicLimits),
        ("io", ctypes.c_uint64 * 6),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


_jobs: list[int] = []  # job handles stay open for this process's lifetime


def _kill_on_exit(proc: subprocess.Popen) -> None:
    """Windows: put the child in a job that is killed when its last handle (ours) closes."""
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    k32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
    k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)

    info = _ExtendedLimits()
    info.basic.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    job = k32.CreateJobObjectW(None, None)
    ok = (
        job
        and k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))  # extended limits
        and k32.AssignProcessToJobObject(job, int(proc._handle))
    )
    if not ok:
        log.warning("could not tie llama-server to this process (WinError %s)", ctypes.get_last_error())
        return
    _jobs.append(job)


def _healthy(http: httpx.Client, url: str) -> bool:
    try:
        return http.get(url).status_code == 200
    except httpx.HTTPError:
        return False
