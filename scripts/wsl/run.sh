#!/usr/bin/env bash
# Start llama-server and the app natively in WSL; Ctrl+C stops both.
# Run scripts/wsl/setup.sh once first. Then open http://localhost:8000 in a Windows browser.
set -euo pipefail

LLAMA_DIR="${LLAMA_DIR:-$HOME/.local/share/llama.cpp/b11782-cuda-13.4}"
VENV="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/h-ai-senberg}"
MODELS_DIR="${MODELS_DIR:-$HOME/.local/share/h-ai-senberg/models}"
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$VENV"
export LD_LIBRARY_PATH="$LLAMA_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# The real environment wins over .env, so this replaces a Windows path a shared .env may set.
export LLAMA_SERVER_EXE="$LLAMA_DIR/llama-server"

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [ ! -x "$LLAMA_SERVER_EXE" ] || [ ! -x "$VENV/bin/haisenberg" ]; then
  echo "not set up yet: run scripts/wsl/setup.sh first" >&2
  exit 1
fi
uv sync --frozen --quiet  # picks up dependency changes after a git pull

# Use the model copy on the WSL disk when setup made one (much faster to load than /mnt/c).
model="$("$VENV/bin/python" -c 'from haisenberg.config import load_config; print(load_config().llama_server.model_path.name)')"
if [ -f "$MODELS_DIR/$model" ]; then
  export MODEL_PATH="$MODELS_DIR/$model"
fi

# Run the app directly (not via `uv run`), so Ctrl+C reaches it once.
exec "$VENV/bin/haisenberg"
