"""Build the WebUI in ui/ into ui/dist (served by the app). Needs Node.js.

Usage: uv run python scripts/build_ui.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

UI = Path(__file__).resolve().parents[1] / "ui"


def main() -> int:
    npm = shutil.which("npm")
    if not npm:
        print("npm not found: install Node.js and reopen the terminal", file=sys.stderr)
        return 1
    env = {**os.environ, "LLAMA_BUILD_NUMBER": "b10173-haisenberg"}
    for step in (["ci", "--no-audit", "--no-fund"], ["run", "check"], ["run", "build"]):
        print(f"> npm {' '.join(step)}", flush=True)
        code = subprocess.call([npm, *step], cwd=UI, env=env)
        if code:
            print(f"npm {' '.join(step)} failed with exit code {code}", file=sys.stderr)
            return code
    print(f"WebUI built: {UI / 'dist'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
