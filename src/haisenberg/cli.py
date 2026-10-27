"""Command-line entry point: `uv run haisenberg` starts llama-server (if enabled, restarted if it dies)
and the app."""

from __future__ import annotations

import logging
import sys

import uvicorn

from .config import ConfigError, load_config
from .launcher import LauncherError, Supervisor
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
    supervisor = Supervisor(cfg)
    try:
        if ls.enabled:
            if f"{ls.host}:{ls.port}" not in cfg.model_backend.api_base:
                log.warning(
                    "model_backend.api_base (%s) does not point at llama_server (%s:%s)",
                    cfg.model_backend.api_base, ls.host, ls.port,
                )
            supervisor.start()
        uvicorn.run(
            create_app(cfg),
            host=cfg.app.host,
            port=cfg.app.port,
            log_level="debug" if cfg.app.debug else "info",
        )
    except (LauncherError, ConfigError) as e:  # ConfigError: KNOWLEDGE=extended without the literature store
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        supervisor.stop()
    return 0
