"""Command-line entry point: `uv run haisenberg` starts llama-server (if enabled) and the app."""

from __future__ import annotations

import logging
import sys

import uvicorn

from .config import ConfigError, load_config
from .launcher import LauncherError, ensure_llama_server, stop
from .server import create_app

log = logging.getLogger("haisenberg")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # no line per health poll
    try:
        cfg = load_config()
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 1

    ls = cfg.llama_server
    proc = None
    try:
        if ls.enabled:
            if f"{ls.host}:{ls.port}" not in cfg.model_backend.api_base:
                log.warning(
                    "model_backend.api_base (%s) does not point at llama_server (%s:%s)",
                    cfg.model_backend.api_base, ls.host, ls.port,
                )
            proc = ensure_llama_server(cfg)
        uvicorn.run(
            create_app(cfg),
            host=cfg.app.host,
            port=cfg.app.port,
            log_level="debug" if cfg.app.debug else "info",
        )
    except LauncherError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        stop(proc)
    return 0
