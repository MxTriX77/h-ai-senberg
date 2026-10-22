# h-ai-senberg

A local, verifiable research chat for chemistry and physics.

A local LLM answers questions and checks its claims against offline sources (Wikipedia chemistry/physics subsets, PubChem, CODATA, open textbooks) before showing the answer. Every claim is cited or marked unverified.

Status: early development.

## Setup

Requirements: Windows, Linux or macOS; [uv](https://docs.astral.sh/uv/) on `PATH`; [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server`) for the local model.

```bash
uv sync
uv run pytest
```

Optional: copy `.env.example` to `.env` to change defaults (ports, backend URL, API key, data folder, model path).

Everything that controls behaviour lives in one file, `config/config.yaml`: model backend, sampling, the system prompt, data paths and the llama-server launch settings. Any value can use `${VAR}` or `${VAR:-default}` from the environment or `.env`.

The default local model is `Qwen3.5-9B-Q4_K_M.gguf` from [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF). Put it in `models/`, or point `MODEL_PATH` at it. Any GGUF that llama.cpp supports works; change `MODEL_PATH` and `MODEL_ID`.
