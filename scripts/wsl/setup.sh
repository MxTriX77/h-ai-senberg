#!/usr/bin/env bash
# One-time setup to run h-ai-senberg natively in WSL (Ubuntu). No sudo needed.
# Installs uv (~/.local/bin), the official llama.cpp CUDA build (~/.local/share/llama.cpp)
# and the Linux Python environment (~/.venvs/h-ai-senberg, separate from the Windows .venv).
# The GPU works through the Windows NVIDIA driver, which WSL exposes as libcuda.
set -euo pipefail

LLAMA_BUILD="${LLAMA_BUILD:-b11782}"
LLAMA_CUDA="${LLAMA_CUDA:-13.4}"
LLAMA_DIR="${LLAMA_DIR:-$HOME/.local/share/llama.cpp/$LLAMA_BUILD-cuda-$LLAMA_CUDA}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/h-ai-senberg}"
MODELS_DIR="${MODELS_DIR:-$HOME/.local/share/h-ai-senberg/models}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

step() { printf '\n== %s\n' "$*"; }

step "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
  echo "warning: nvidia-smi not found; llama-server will fall back to the CPU" >&2
fi

step "uv"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
fi
uv --version

step "llama.cpp $LLAMA_BUILD, CUDA $LLAMA_CUDA -> $LLAMA_DIR"
if [ -x "$LLAMA_DIR/llama-server" ]; then
  echo "already installed"
else
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  base="${LLAMA_BASE_URL:-https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_BUILD}"
  mkdir -p "$LLAMA_DIR"
  # The server archive, then the CUDA runtime libraries it needs (WSL ships only libcuda).
  for pair in "llama-$LLAMA_BUILD-bin-ubuntu-cuda-$LLAMA_CUDA-x64.tar.gz:llama-server" \
              "cudart-llama-$LLAMA_BUILD-bin-ubuntu-cuda-$LLAMA_CUDA-x64.tar.gz:libcudart.so*"; do
    file="${pair%%:*}" marker="${pair#*:}"
    echo "downloading $file"
    curl -fL --retry 3 --progress-bar -o "$tmp/$file" "$base/$file"
    mkdir "$tmp/x"
    tar -xzf "$tmp/$file" -C "$tmp/x"
    # Copy the folder that holds the files, whatever the archive's top folder is called.
    src="$(dirname "$(find "$tmp/x" -name "$marker" | head -n 1)")"
    cp -a "$src"/. "$LLAMA_DIR"/
    rm -rf "$tmp/x" "$tmp/$file"
  done
fi
LD_LIBRARY_PATH="$LLAMA_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$LLAMA_DIR/llama-server" --list-devices

step "Python environment -> $UV_PROJECT_ENVIRONMENT"
cd "$REPO"
uv sync --frozen

step "models -> $MODELS_DIR"
# Reading a GGUF from /mnt/c is slow (~90 s per start); a copy on the WSL disk loads in seconds.
mkdir -p "$MODELS_DIR"
for f in "$REPO"/models/*.gguf; do
  [ -e "$f" ] || { echo "no .gguf in $REPO/models"; break; }
  dst="$MODELS_DIR/$(basename "$f")"
  if [ -f "$dst" ] && [ "$(stat -c %s "$f")" = "$(stat -c %s "$dst")" ]; then
    echo "already copied: $(basename "$f")"
  else
    echo "copying $(basename "$f") ($(du -h "$f" | cut -f1))"
    cp "$f" "$dst.part" && mv "$dst.part" "$dst"
  fi
done

step "WebUI"
if [ -f ui/dist/index.html ]; then
  echo "built"
else
  echo "not built: run 'uv run python scripts/build_ui.py' once (needs Node.js), from Windows or WSL"
fi

step "done: start with  scripts/wsl/run.sh"
