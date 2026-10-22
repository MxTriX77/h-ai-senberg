"""Start (or reuse) the local llama-server from config, wait until it is ready, stop it on exit."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
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
    proc = subprocess.Popen(command, env=build_env(cfg))  # console inherited: load progress shows here
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


def _healthy(http: httpx.Client, url: str) -> bool:
    try:
        return http.get(url).status_code == 200
    except httpx.HTTPError:
        return False
