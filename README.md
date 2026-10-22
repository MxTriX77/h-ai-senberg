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

### Slow is fine, crashing is not

- **No time limit on the model.** The app waits as long as the model needs: a request queued behind another one on llama-server's single slot, a long prompt on a CPU-offloaded model, slow generation (`model_backend.timeout_s: null`; set seconds to limit it). If the backend refuses connections or is still loading (HTTP 503), each call retries for up to `model_backend.wait_s` (300 s).
- **Keep-alives.** A streamed answer starts at once, and the app sends an SSE comment (`: keep-alive`) after every `app.keepalive_s` (1 s) without output, so the browser never sees a silent connection (the WebUI gives up on a stream that is silent for 3 s when its tab comes back into view).
- **Failures end the answer, not the connection.** If anything fails during an answer (the backend dies, rejects a request, or the app hits an error), the answer says what went wrong and that the question can be sent again, and the stream ends normally.
- **llama-server is restarted.** When the llama-server the app started exits (a crash, a kill), the app logs its exit code or signal and starts it again; the next question works once it has loaded. Set `LLAMA_LOG_FILE=/path/llama.log` to also keep llama-server's own log in a file.
- **The context cannot overflow.** Each model call fits `prompt + max_tokens` into the context (`llama_server.ctx_size`, or `model_backend.context_tokens` for a remote backend), estimating tokens as characters / `model_backend.chars_per_token` (2.0, on the safe side of the 2.4 measured on dense tool results). Old turns of a long chat are dropped first, tool results are shortened when they would not fit (and the model then answers with what it has), and the judge gets only the passages that fit.

### Tools and citations

When the Layer 1 store exists, the model looks facts up before it answers. The tools are generated from the sources in the store: `search_knowledge(query, domain?)` searches every text source (today the Wikipedia articles; `domain` is one of the domains the sources declare, e.g. `chemistry`, `physics`), and there is one lookup per kind of structured data (`lookup_compound` for PubChem, `lookup_constant` for CODATA). Their descriptions, the extra instructions for tool use (`reasoning_policy.tool_prompt`) and the number of tool rounds per answer (`reasoning_policy.max_search_depth`; `0` turns tools off) are in `config/config.yaml`. Tools are sent as standard OpenAI `tools`, so the backend must support tool calls (llama-server does with `--jinja`, which the launcher sets).

Weak search hits are dropped before the model sees them. A hit is kept only when its title and passage contain at least `retrieval.min_term_share` (half) of the query's content words, or when the store matched its title or alias. Dropped hits show as "weak match, not used" in the Reasoning block. Without the filter, a query whose words no single article holds gets articles that share just one of them. Set `0` to turn the filter off.

The Reasoning block shows the model's thinking, every tool call with the sources it returned (`[S1] Ethanol`, …) and the draft, in which the model cites sources as `[S1]`. The answer follows once the model is done, as a plain chat answer (see Verification).

### Verification

Before the answer is shown, the draft is checked claim by claim (sentences and list items) against the local sources only:

- Numbers and formulas are compared in code with the structured records (PubChem molar masses and formulas, CODATA values), within the claim's own rounding or `verifier.rel_tol`. A wrong number or formula next to a cited record is replaced by the record's value.
- Everything else is checked by one extra model call, the judge (`verifier.judge_prompt`, without thinking), which reads the cited passages and says per claim: supports, contradicts (with a fix), not said, or no claim.
- A claim no source supports gets one more search with its own text and a second judge call.
- Last, a consistency review reads the whole answer with the user's task (one judge call, `verifier.consistency_prompt`). It looks for conclusions, verdicts or table cells that contradict the answer's own calculations, labels that conflict with their reasons, implausible results presented as fine, ignored instructions, and parts that contradict each other. It replies with exact `OLD`/`NEW` replacements. A replacement is applied only when its text occurs exactly once in the answer. Switch: `verifier.consistency_check`.

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

Keep that terminal open while you use the app. WSL shuts the whole Ubuntu instance down about 15 s after its last terminal (or other `wsl.exe` process) closes, and llama-server and the app go with it, even when they were started in the background with `nohup` or `&`. To run unattended, keep a `wsl.exe` session attached.

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

`evals/reasoning.yaml` holds 31 multi-step problems, also written for this project: stoichiometry and yields (including an impossible yield), unit conversions, kinematics, energy, circuits, gas laws, pH, buffers, equilibrium, titration, decay, inconsistent given data, sorting claims into true, false or unconfirmed from the given data, finding the wrong statements in a short report, and following explicit constraints. Some are in Russian. Each problem asks for its results on labelled lines (`A1: …`, `A2: …`), and every part is scored in code on its line: a number within a tolerance (a decimal comma is fine), a verdict word at the start of the line, or the exact set of wrong statement ids. No LLM grades anything. A problem counts as solved when all its parts are right. Five problems are held out: they are for a final check only and never run unless you pass `--holdout`.

The runner sends questions and problems through the same tools, tool loop and verifier as the chat, using the configured backend (the model must be running):

```bash
uv run python -m evals.run --limit 20      # 20 items spread over every source/domain group and the reasoning set
uv run python -m evals.run                 # all fact questions and reasoning problems (hours with a local 9B model)
uv run python -m evals.run --set reasoning # only the reasoning problems (or --set facts)
uv run python -m evals.run --holdout       # only the held-out problems: final check
uv run python -m evals.run --ids codata-01,r-stoich-02
```

It prints one line per item (for problems, `+`/`-` per part) and a summary for all items and per set: % correct answers (solved problems), % of problem parts right, % of claims supported, unmarked contradictions (wrong answers whose every claim the verifier rated supported, target 0), unverified/corrected/removed claims and seconds per item. Outcomes come from the verification report, so the score is the same in both answer styles. The full report goes to `data/evals/run-<timestamp>.json` (or `--out FILE`).
