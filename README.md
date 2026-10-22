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

## Run

```bash
uv run haisenberg
```

This starts `llama-server` with the model from `config/config.yaml` (or reuses one already running on port 8080), waits until the model is loaded, then serves the app at http://127.0.0.1:8000. Ctrl+C stops both.

The app speaks the OpenAI chat API at `http://127.0.0.1:8000/v1/chat/completions` (streaming and non-streaming): the model's thinking arrives as `reasoning_content`, the answer as `content`.

The server owns the policy: the system prompt from `config/config.yaml` replaces any system message a client sends, and client sampling settings are ignored.

To use another backend (an already running server or any OpenAI-compatible API), set in `.env`:

```bash
LLAMA_SERVER_ENABLED=false
API_BASE_URL=https://example.com/v1
API_KEY=your-key
MODEL_ID=model-name
```
