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

### Tools and citations

When the Layer 1 store exists, the model looks facts up before it answers. The tools are generated from the sources in the store: `search_knowledge(query, domain?)` searches every text source (today the Wikipedia articles; `domain` is one of the domains the sources declare, e.g. `chemistry`, `physics`), and there is one lookup per kind of structured data (`lookup_compound` for PubChem, `lookup_constant` for CODATA). Their descriptions, the extra instructions for tool use (`reasoning_policy.tool_prompt`) and the number of tool rounds per answer (`reasoning_policy.max_search_depth`; `0` turns tools off) are in `config/config.yaml`. Tools are sent as standard OpenAI `tools`, so the backend must support tool calls (llama-server does with `--jinja`, which the launcher sets).

The Reasoning block shows the model's thinking, every tool call with the sources it returned (`[S1] Ethanol`, …) and the draft, in which the model cites sources as `[S1]`. The answer follows once the model is done, as a plain chat answer (see Verification).

### Verification

Before the answer is shown, the draft is checked claim by claim (sentences and list items) against the local sources only:

- Numbers and formulas are compared in code with the structured records (PubChem molar masses and formulas, CODATA values), within the claim's own rounding or `verifier.rel_tol`. A wrong number or formula next to a cited record is replaced by the record's value.
- Everything else is checked by one extra model call, the judge (`verifier.judge_prompt`, without thinking), which reads the cited passages and says per claim: supports, contradicts (with a fix), not said, or no claim.
- A claim no source supports gets one more search with its own text and a second judge call.

The checks only make the answer better; they are not shown in it. The answer reads like a normal chat answer: contradicted claims are replaced by the corrected text or dropped when there is no fix, everything else stays as the model wrote it, and the `[S#]` ids are removed. Every check, fix and source is listed in the Reasoning block. For debugging, `verifier.answer_style: annotated` shows the outcome of every claim in the answer instead: supported claims with their source ids, *(unverified)*, *(corrected from [S2])*, a summary line counting the outcomes, and the list of cited sources with license and URL. Turn verification off with `reasoning_policy.enable_multi_hop_verification: false`; settings are in the `verifier` section of `config/config.yaml`.

To use another backend (an already running server or any OpenAI-compatible API), set in `.env`:

```bash
LLAMA_SERVER_ENABLED=false
API_BASE_URL=https://example.com/v1
API_KEY=your-key
MODEL_ID=model-name
```

### Other machines and models

Nothing in the code is tied to this laptop or model. On another machine (for example a cloud GPU box), set environment variables instead of editing files:

- `MODEL_PATH`, `MODEL_ID`: any GGUF llama.cpp supports
- `CTX_SIZE`, `GPU_LAYERS`, `LLAMA_THREADS`, `REASONING_BUDGET`: context size, layers on the GPU, CPU threads, thinking-token cap
- `LLAMA_SERVER_ENABLED=false` with `API_BASE_URL`, `API_KEY`: use a model server that is already running, here or remote
- `APP_HOST`, `APP_PORT`, `DATA_DIR`: address and data folder

All variables and their defaults are in `.env.example`.

**Model profiles.** Each extra model gets a small file `config/<name>.yaml` with only its own settings (model file and ID, sampling, thinking budget, context, GPU layers or MoE offload), merged over `config/config.yaml`. Pick one with `MODEL_PROFILE`, for example:

```bash
MODEL_PROFILE=<name> bash scripts/wsl/run.sh
```

Without `MODEL_PROFILE` the defaults in `config/config.yaml` apply.

## Run from WSL (Ubuntu)

Runs natively in WSL, with the GPU through the Windows NVIDIA driver. No sudo needed. From the repository folder (for example `/mnt/c/.dev/h-ai-senberg`):

```bash
bash scripts/wsl/setup.sh
bash scripts/wsl/run.sh
```

Run `setup.sh` once, and again after adding a model. Then `run.sh` starts llama-server and the app; open http://localhost:8000 in a Windows browser. Ctrl+C stops both.

`setup.sh` installs everything under your WSL home: uv (`~/.local/bin`), the official llama.cpp Ubuntu CUDA build (`~/.local/share/llama.cpp/`), the Python environment (`~/.venvs/h-ai-senberg`, separate from the Windows `.venv`), and a copy of `models/*.gguf` on the WSL disk (`~/.local/share/h-ai-senberg/models/`), which loads in seconds instead of ~90 s over `/mnt/c`. The Windows setup is left as it is.

## WebUI

`ui/` is the [llama.cpp](https://github.com/ggml-org/llama.cpp) WebUI from tag `b10173` (MIT, see `ui/LICENSE`), stripped down: no attachments, model picker, regenerate/delete, system-prompt or thinking controls. The app serves the built UI at http://127.0.0.1:8000; the model's thinking shows in the collapsible Reasoning block above each answer.

Build it once (and after UI changes); needs [Node.js](https://nodejs.org/):

```bash
uv run python scripts/build_ui.py
```

The build goes to `ui/dist/` (not committed).

## Data

Offline sources are listed in `config/sources.yaml` (URL, destination, checksum, license, domains). One command downloads whatever is missing into `data/raw/` (or `$DATA_DIR/raw`):

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

Every record carries its domains and a citation (source file and version, license, locator, URL). Query it from the command line:

```bash
uv run haisenberg-l1 search "planck constant"
uv run haisenberg-l1 compound caffeine          # also CID 2519, 58-08-2, C8H10N4O2
uv run haisenberg-l1 constant "electron mass" --json
```

Adding a source of a known kind is config only: add its entry (with `domains`) to `config/sources.yaml`, download, rebuild; the tools pick it up. Which pages of the all-articles mini ZIM fill gaps (infobox classes, WikiProjects and importance ratings, each mapped to a domain) is the `filter` of its entry.

The store has a schema version; code from a newer version refuses an older store and asks for a rebuild (`uv run python -m ingest.build_l1`).

With the 2026 sources the store holds about 37,500 articles, 355 constants and 96,700 compounds: about 200 MB, rebuilt in about 6 minutes. Lookups take a few milliseconds. `uv run pytest` runs a latency and facts check on the real store when it exists (set `DATA_DIR` if it is not under `data/`).

## Evals

`evals/questions.yaml` holds about 100 chemistry and physics questions with known answers (a number with a tolerance, a formula, or accepted words), written for this project. The known answers come from CODATA 2022 (public domain), PubChem (no restrictions) and facts stated in the Wikipedia articles of the store (CC BY-SA 4.0; facts only, no text copied). A test checks every CODATA and PubChem answer against the store.

The runner sends questions through the same tools, tool loop and verifier as the chat, using the configured backend (the model must be running):

```bash
uv run python -m evals.run --limit 20      # 20 questions spread over every source and domain
uv run python -m evals.run                 # all of them (hours with a local 9B model)
uv run python -m evals.run --ids codata-01,wchem-01
```

It prints one line per question and a summary: % correct answers, % of claims supported, unmarked contradictions (wrong answers whose every claim the verifier rated supported, target 0), unverified/corrected/removed claims and seconds per question. Outcomes come from the verification report, so the score is the same in both answer styles. The full report goes to `data/evals/run-<timestamp>.json` (or `--out FILE`).
