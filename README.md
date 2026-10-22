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

## WebUI

`ui/` is the [llama.cpp](https://github.com/ggml-org/llama.cpp) WebUI from tag `b10173` (MIT, see `ui/LICENSE`), stripped down: no attachments, model picker, regenerate/delete, system-prompt or thinking controls. The app serves the built UI at http://127.0.0.1:8000; the model's thinking shows in the collapsible Reasoning block above each answer.

Build it once (and after UI changes); needs [Node.js](https://nodejs.org/):

```bash
uv run python scripts/build_ui.py
```

The build goes to `ui/dist/` (not committed).

## Data

Offline sources are listed in `config/sources.yaml` (URL, destination, checksum, license). One command downloads whatever is missing into `data/raw/` (or `$DATA_DIR/raw`):

```bash
uv run python -m ingest.download            # everything (~22 GB)
uv run python -m ingest.download --layer 1  # Layer 1 only: Wikipedia, PubChem, CODATA (~19 GB)
```

Files that are already present and verified are skipped, and partial downloads resume. Progress goes to `data/raw/download.log`, and a `DONE` or `FAILED` marker is written at the end. PubChem regenerates its files daily; they are checked against PubChem's own `.md5`, and the verified digest is kept next to each file (`*.md5`). To fetch a newer copy, delete the file and its `.md5`.

### Layer 1 store

One command rebuilds the Layer 1 store (`data/encyclopedia_fts.db`, one SQLite file) from `data/raw/` alone, without network:

```bash
uv run python -m ingest.build_l1
```

It holds:

- **Wikipedia articles** (lead and infobox text, with full-text search): everything from the chemistry and physics topic ZIMs except person articles. Gaps are filled from the all-articles mini ZIM: compound and element articles, and chemistry/physics articles that WikiProject Chemistry, Chemicals, Physics or Elements rates Top, High or Mid importance.
- **CODATA 2022 constants**: all 355.
- **PubChem compounds**: every compound with a MeSH heading, plus every compound linked from a stored article. Each has its title, formula and synonyms, and a molar mass computed from the formula with IUPAC standard atomic weights.

Every record carries a citation (source file and version, license, locator, URL). Query it from the command line:

```bash
uv run haisenberg-l1 search "planck constant"
uv run haisenberg-l1 compound caffeine          # also CID 2519, 58-08-2, C8H10N4O2
uv run haisenberg-l1 constant "electron mass" --json
```

With the 2026 sources the store holds about 37,500 articles, 355 constants and 96,700 compounds: about 200 MB, rebuilt in about 6 minutes. Lookups take a few milliseconds. `uv run pytest` runs a latency and facts check on the real store when it exists (set `DATA_DIR` if it is not under `data/`).
