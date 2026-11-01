"""Start (or reuse) the local llama-server from config, wait until it is ready, restart it if it dies,
stop it on exit."""

from __future__ import annotations

import ctypes
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
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
        elif value is not False and value is not None and value != "":
            cmd += [f"--{name}", str(value)]
    return cmd


def build_env(cfg: Config) -> dict[str, str]:
    return {**os.environ, **cfg.llama_server.env}


def health_url(cfg: Config) -> str:
    ls = cfg.llama_server
    return f"http://{ls.host}:{ls.port}/health"


def embedding_view(cfg: Config) -> Config:
    """The config as this launcher sees the embedding server: llama_server = embedding.server, alias =
    embedding.model_id. Every function here and Supervisor take it unchanged."""
    backend = cfg.model_backend.model_copy(update={"model_id": cfg.embedding.model_id})
    return cfg.model_copy(update={"llama_server": cfg.embedding.server, "model_backend": backend})


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


class Supervisor:
    """Runs the llama-server this app starts: restarts it when it exits on its own (crash, kill),
    with a growing pause between failed attempts, until `stop()`."""

    def __init__(self, cfg: Config, start=None, pause_s: float = 5.0, max_pause_s: float = 60.0):
        self.cfg = cfg
        self._start = start or (lambda: ensure_llama_server(cfg))
        self.pause_s, self.max_pause_s = pause_s, max_pause_s
        self.proc: subprocess.Popen | None = None
        self.restarts = 0
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start (or reuse) llama-server and wait until it is ready; raises LauncherError. Only a process
        started here is watched. The watcher thread lives as long as the app: on Linux a child started by
        it gets SIGTERM when the thread ends (PR_SET_PDEATHSIG is per thread)."""
        self.proc = self._start()
        if self.proc is not None:
            self._thread = threading.Thread(target=self._watch, name="llama-server-watch", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        stop(self.proc)
        if self._thread is not None:
            self._thread.join(15)

    def _watch(self) -> None:
        pause = self.pause_s
        while not self._stopping.is_set():
            proc = self.proc
            code = proc.wait() if proc is not None else None
            if self._stopping.is_set():
                return
            if proc is not None:
                log.error("llama-server exited with %s; restarting it", describe_exit(code))
            try:
                self.proc = self._start()
                if self._stopping.is_set():  # stop() came while it was loading
                    stop(self.proc)
                    return
                self.restarts += 1
                pause = self.pause_s
                if self.proc is None:
                    log.warning("another llama-server answers on the port now; not watching it")
                    return
                log.info("llama-server restarted (restart %d)", self.restarts)
            except Exception as e:  # never let the watcher die: llama-server must come back
                self.proc = None
                log.error("restarting llama-server failed: %s; next try in %g s", e, pause)
                if self._stopping.wait(pause):
                    return
                pause = min(pause * 2, self.max_pause_s)


def describe_exit(code: int | None) -> str:
    """'code 1' or, for a process killed by a signal (negative code on POSIX), 'signal SIGKILL'."""
    if code is not None and code < 0:
        try:
            return f"signal {signal.Signals(-code).name}"
        except ValueError:
            return f"signal {-code}"
    return f"code {code}"


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
