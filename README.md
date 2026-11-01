# h-ai-senberg

A local, general-purpose chatbot with verifiable reasoning. Chemistry and physics are its first knowledge domains.

A local LLM answers like a normal chat assistant. Behind the answer, facts are looked up in offline sources (Wikipedia chemistry/physics subsets, PubChem, CODATA, open textbooks), numbers are recomputed in code, and claims are checked against the sources before the answer is shown. The checks work silently: a wrong claim is corrected from a source or left unverified, and the lookups, the draft and every check appear in a collapsible Reasoning block, not in the answer.

Status: in development. Quality is measured on eval sets scored in code (see [Evals](#evals)). Today every message goes through the full lookup-and-check pipeline, and the shipped system prompt is written for chemistry and physics, so casual questions also get thorough, calculation-style answers; routing each message to the depth it needs is planned.

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

Next to the knowledge tools there is a built-in `calculate(expressions)` tool. It evaluates plain arithmetic exactly: numbers in any common notation, `+ - * / ^`, parentheses, `sqrt`, `exp`, `ln`, `log`, trigonometry, `pi` and `e`. It parses each expression and runs no code. One call takes a list of expressions and returns one `expression = value` line (or error) per expression; a single string counts as one expression. The tool prompt asks the model to use it for every calculation and to put all the calculations of a step into one call. Its text is `tools.calculate` in the config; remove that entry (and the calculator sentence of the tool prompt) to turn the tool off.

Calculations have their own rounds, so they do not use up the lookups: a round with only `calculate` calls counts against `reasoning_policy.max_calc_rounds` (1: one extra round for calculations after the lookups), not `max_search_depth`. Once the lookup rounds are used, the model is offered the calculator alone until its rounds are used too; then the last call sends `tool_choice: "none"`. With `max_calc_rounds: 0` every round counts as a lookup round.

Safety nets keep the answer an answer:

- **Tool calls written as text.** Some chat templates write a tool call as text (`<tool_call><function=calculate>…`) when tools are not allowed. Such text in the reply is read with the patterns in `reasoning_policy.tool_call_markup` (Qwen3.5 XML and Hermes JSON). When every call names a tool that still has rounds, the calls run and the loop goes on. Otherwise the markup is removed and the model is asked once more for a plain final answer (`reasoning_policy.answer_retry_prompt`, request fields `answer_retry_request`, shipped: thinking off). Switch: `reasoning_policy.text_tool_calls`.
- **Calculations written as text.** The calculator needs no model call, so `calculate` calls written where they cannot run as a round (after the last round, or in that retry) are run anyway and their results given back before the final answer is asked for; calls to other tools are removed. A calculation written in the retry costs one more model call, at most one per answer. Every removed call shows in the reasoning block. Switch: `reasoning_policy.run_written_calculations`.
- **Answers cut off at a removed call.** If the last reply still ends where a call was removed (or is empty), the most complete reply is kept: the one with the most labelled answer lines (`A1: …`) whose labels the question names, then a complete one over a cut one, then the longer one. The reasoning block says which reply was kept and why.
- **Empty answers.** When the model leaves its whole answer in the thinking and returns no content, it is asked once more the same way. Switch: `reasoning_policy.empty_answer_retry`.

If the retry gives nothing, the text before the markup is kept; if there is none at all, the answer says plainly that it could not be completed. The Reasoning block shows what happened ("tool call written as text: run", "empty answer: final answer asked again").

The Reasoning block shows the model's thinking, every tool call with the sources it returned (`[S1] Ethanol`, …) and the draft, in which the model cites sources as `[S1]`. The answer follows once the model is done, as a plain chat answer (see Verification).

### Verification

Before the answer is shown, the draft is checked claim by claim (sentences and list items) against the local sources only:

- Arithmetic first: every `a × b = c` in the draft (also with units after the numbers, `%`, roots, logs, powers and LaTeX such as `\frac{a}{b}`) is recomputed in code. A result that is off by more than its rounding and `verifier.arithmetic_rel_tol` (0.5 %) is replaced by the right value. A result in % is read every plausible way (as written or as a fraction; `a%` as a/100 or as a) and only rewritten when it is wrong under all of them. When a unit in the chain has an SI prefix (on an operand, right after the result or after the next `=`), the result is also read scaled by 10^±3, ±6 or ±9, so `… = 2452 W = 2.45 kW` or `250 mg × 6 = 1.5 g` stays; a result wrong under every reading is rewritten in the scale nearest to what was written (switch: `verifier.arithmetic_prefixes`). Digits grouped in threes with spaces, thin spaces, `\,` or commas (`20 000`, `1 234,5`, `1,234.5`) are one number everywhere; a number or expression that cannot be read one way is left alone. A list marker at the start of a line (`- `, `* `, `1. `) is not a minus sign; `-5 × 2` still is. Switch: `verifier.arithmetic_check`. A labelled answer line (`A1: …`) that states exactly the old wrong result of a fixed computation, once, gets the fixed value too (`A1: 3.80 g` becomes `A1: 2.80 g` after `0.0500 × 56.08 = 3.80` was fixed); nothing else changes answer lines here (switch: `verifier.arithmetic_answer_lines`).
- Only a whole left side is checked. `a × 10ⁿ` (also `a·10^n`, `a \times 10^{n}`) and `10ⁿ` are one number, so `6.4 × 10⁷ ÷ 2.0 × 10⁶ = 32` is right, and a left side that is then one number (`125 × 10⁶ bytes/s = 125 MB/s`) is a conversion, not arithmetic. A currency sign before a number is dropped (`($1,840,000 − $1,232,800) / $1,840,000 × 100`). An expression is skipped when the text right before it belongs to the math (an operator, `(`, `½`, `π`, `half of`, `million`: `½ · 1200 kg · (25 m/s)²` is not read as `1200 kg · (25 m/s)²`), or when a number carries a scale suffix (`1.3B`, `20k`, `2 million`, `1,5 млн`); Markdown `**bold**` and `*italics*` around an equation do not count. A wrong result that is a unit conversion is left as written: its unit is not on the left side, and one side has a scale prefix the prefix readings do not know (`MB`, `Gbit`, `cm`) on the same unit as the other side (`bytes`, `bits`, `m`). Switch: `verifier.arithmetic_strict_parse` (false: as before).
- Only slip-sized fixes are applied: a fix that changes the stated number by `verifier.arithmetic_max_change` (100) times or more, flips its sign, or goes to or from zero at the stated precision (`2.5 hours / 5 = 30` would become `0`) is only listed in the Reasoning block (`not fixed: change too large to be a slip (would be …)`). So is a division whose stated result is the reciprocal of its value (`€1224 / €4200 = 3.43 years`: `result is the reciprocal: division written the other way round?`). `0` applies every fix.
- Domain checkers next: plug-ins in `src/haisenberg/checkers/` that know one subject, listed in `verifier.checkers`. The verifier itself knows no subject. The first one, `element_balance`, counts atoms in reaction equations written with formulas. It rebalances an equation when exactly one balance exists (`H2 + O2 → H2O` becomes `2H2 + O2 → 2H2O`) and only reports it otherwise. It also checks "A → B loses X" statements by subtracting the formulas. Ions, names and LaTeX are left alone. It rewrites only clear chemistry: an equation with a formula that has an explicit count of 1 (`P₁V₁ = P₂V₂`, `S1 → S2`: an indexed symbol or a label), a written number over 20 in front (`1000 IU = 3000 IU`: an amount) or a step or statement label of the task or the answer (`S5 → S6` when lines start with `S4.`, `S5.`, …; one letter with two or more numbers, so a lone `O2: 32 g/mol` line is not a label) is only reported, and the Reasoning block names the fix it would have made (switch: `verifier.checkers_clear_only`). Add a checker module and list it to cover another subject; `[]` turns them off.
- Numbers and formulas are compared in code with the structured records (PubChem molar masses and formulas, CODATA values), within the claim's own rounding or `verifier.rel_tol`. A wrong number or formula next to a cited record is replaced by the record's value. Each record value says what it is: its unit (a constant's CODATA unit, `g/mol` for a molar mass, `Da` for a monoisotopic mass) and, for a compound, the words that name its property. A number counts only as the same quantity (switch: `verifier.unit_checks`; off: numbers alone, as before). A unit written after it must be the record's unit or an equal spelling: units are read in code (`src/haisenberg/l1/units.py`) and compared as SI base units with their scale, so `J s` = `J Hz^-1` = `J·s` and `m/s` = `m s⁻¹` = `metres per second`, while `eV s`, `km/h` or `°C` are other units and leave the claim to the judge. The property words nearest the number must not name another property (`boils at`, `density`, `monoisotopic` next to a molar mass). Support also needs the record cited or named (a constant's value with its unit is enough). A correction needs the record's unit after the number, or the property (for a constant: its name) named with no word after the number, and a number within a factor of ten of the record's; a pure number (a dimensionless constant) is corrected only when the claim names the constant.
- Computations are checked only in code, never against sources: a sentence with an equation, comparison or operation with numbers, or one that restates the answer's own results (next to numbers from the task), is left as written (unverified, not wrong). Labelled answer and verdict lines (`A1: …`, `Claim C: Incorrect`) are not claims at all, like table rows. Switch: `verifier.skip_derived_claims`. A bare label cut off its line (`S3.`, `**C2.`, `A1)`) is never a claim.
- Everything else is checked by one extra model call, the judge (`verifier.judge_prompt`, without thinking), which reads the cited passages and says per claim: supports, contradicts (with a fix), not said, or no claim. A contradiction without a fix only marks the claim unverified; nothing is removed without a passage value to put in its place (switch: `verifier.keep_uncorrected`). A fix that repeats the judge's own instructions (for example its format example, five words in a row) counts as no fix (switch: `verifier.ignore_prompt_echo`). A fix is applied only as a minimal edit: compared term by term with the claim (numbers by value, words, formulas and unit symbols case ignored), each change replaces one to three terms by one to three terms, adds or drops nothing, does not turn a word into one that contains it (`balanced` → `unbalanced`), and every term it puts in is stated in the passages the judge named (numbers within rounding), while no number or content word it takes out is (a passage that states both, like `halite` and `galena` or `1538 °C` and `2862 °C`, does not contradict the claim). Only those terms change; the rest of the claim stays as the model wrote it. A rewrite, an added explanation, a changed verdict or a value no passage states leaves the claim as written and unverified, and the Reasoning block says why (switch: `verifier.judge_minimal_edits`; off: the fix replaces the whole claim).
- A claim no source supports gets one more search with its own text and a second judge call.
- A claim that only literature passages support is not counted as supported (unverified, text unchanged) when a reference record retrieved for the answer gives another value for the same quantity: a number with the same unit, within a factor of two, in a line or sentence of the record that holds at least half of the claim's words. Switch: `verifier.reference_over_literature`.
- Last, a consistency review reads the whole answer with the user's task (one judge call, `verifier.consistency_prompt`). It looks only for places where the answer contradicts itself: conclusions, verdicts, answer lines or table cells that differ from the answer's own calculations, labels that conflict with their reasons, and parts that contradict each other; it does not solve the task again or reword. It replies with exact `OLD`/`NEW` replacements and quotes, as `CONFLICT`, the other statement of the answer that `OLD` contradicts. A replacement is applied only when its text occurs exactly once in the answer and code confirms the conflict: the quote is elsewhere in the answer, every number the replacement adds is stated in the quote (within rounding) or computed there, and, for an answer or verdict line (`A1: …`) or a change without a number, every word it adds is in the quote. Anything else stays as written, and the Reasoning block says why (switch: `verifier.consistency_quotes`; off: any exact quote is replaced). Labelled answer and verdict lines (`A1: …`, `Claim C: …`) are never changed by the review: a model that reconsiders while writing its final lines is more often right there than in the analysis before, so the line is kept and the Reasoning block names the conflict the review saw (switch: `verifier.consistency_keep_answer_lines`; off: such lines follow the quote rule, and a fix that would drop a line's label is skipped). Switch: `verifier.consistency_check`.
- Quotations of the task are never rewritten. An answer that audits a report often quotes the report's equation, claim or figure to call it wrong; fixing the quote would make the answer call the fixed text wrong. A run of the answer of at least 20 characters that also occurs in the user's message (compared with sub- and superscript digits as plain digits, without Markdown emphasis or quote marks, and with spacing collapsed) counts as a quotation; a formula, a name or a short equation alone does not. A domain checker's fix inside a quotation is only reported (the Reasoning block says it was quoted from the task and names the fix), and so is an arithmetic fix of a wrong `a × b = c` that the task writes the same way, with its result inside a quotation; a claim that is a quotation plus at most a label or verdict mark (`✗ WRONG`, no number) is not checked (unverified); a review fix that changes quoted text is skipped. Everything the answer says itself is checked as before. Switch: `verifier.keep_quotes`.

The checks only make the answer better; they are not shown in it. The answer reads like a normal chat answer: contradicted claims are replaced by the corrected text (or dropped when there is no fix and `keep_uncorrected` is off), everything else stays as the model wrote it, and the `[S#]` ids are removed. Every check, fix and source is listed in the Reasoning block. For debugging, `verifier.answer_style: annotated` shows the outcome of every claim in the answer instead: supported claims with their source ids, *(unverified)*, *(corrected from [S2])*, a summary line counting the outcomes, and the list of cited sources with license and URL. Turn verification off with `reasoning_policy.enable_multi_hop_verification: false`; settings are in the `verifier` section of `config/config.yaml`.

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
- `KNOWLEDGE`: which knowledge stores the tools read (`base`: the Layer 1 store; `extended`: also the Layer 2 literature store)

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

## Literature store (Layer 2)

A second store, `data/literature_fts.db`, holds cited passages for mechanisms, derivations and worked material: full Wikipedia articles (chemistry and physics topic sets, mathematics, astronomy, molecular and cell biology, climate, and the chemistry/physics articles of the top-1M set), LibreTexts textbooks (chemistry, physics, mathematics, engineering, biology), Stack Exchange answers voted 3 or more (chemistry, physics, matter modeling, astronomy, engineering, earth science), 60 Wikibooks science books and ChEBI definitions. Every page is cut into passages of about 1,200 characters along its headings (math kept as TeX; references, tables and navigation left out; a worked example kept with its solution), identical passages are stored once, and the passages are searched with SQLite FTS5 (BM25):

```bash
uv run python -m ingest.download --layer 2     # about 27 GB of new files under data/raw/
uv run python -m ingest.build_l2               # builds data/literature_fts.db next to the Layer 1 store
uv run haisenberg-l2 search "coupled equilibria" --domain chemistry
```

Stores are declared in `config/config.yaml` under `retrieval.stores` (name, kind, path under `DATA_DIR`); `retrieval.knowledge_sets` says which stores each knowledge setting enables, and `KNOWLEDGE` picks the setting:

- `KNOWLEDGE=base` (default): the Layer 1 store only, exactly as before.
- `KNOWLEDGE=extended`: `search_knowledge` returns the same Layer 1 records, then up to `retrieval.literature_k` (2) literature passages. Each result's title then says what it is (`retrieval.source_labels`: Layer 1 records are reference data, literature passages explanatory text whose numbers may differ by source or conditions), and the `search_knowledge` description adds `tools.search_knowledge.literature`: take property values and constants from the reference data, use literature for explanations and context, and give the reference value when they disagree. Switch: `retrieval.label_sources` (off: the same tools and texts as `base`). With `extended` and no literature store the app and `evals.run` stop with the rebuild command.

Licences: Wikipedia, Wikibooks and Stack Exchange CC BY-SA; LibreTexts mostly CC BY-NC-SA (local, non-commercial use); ChEBI CC BY 4.0.

## Evals

`evals/questions.yaml` holds about 100 chemistry and physics questions with known answers (a number with a tolerance, a formula, or accepted words), written for this project. The known answers come from CODATA 2022 (public domain), PubChem (no restrictions) and facts stated in the Wikipedia articles of the store (CC BY-SA 4.0; facts only, no text copied). A test checks every CODATA and PubChem answer against the store. The file's `answer_target` is appended to every question and asks the answer to end with the line `A1: <your final answer, with its unit if it has one>`. A fact question is scored on that line only, with the rules of the reasoning parts below (a formula in any order of its atoms); without the line it is wrong. The looser score of earlier runs, a match anywhere in the answer, is kept as `loose_correct`.

`evals/reasoning.yaml` holds 31 multi-step problems, also written for this project: stoichiometry and yields (including an impossible yield), unit conversions, kinematics, energy, circuits, gas laws, pH, buffers, equilibrium, titration, decay, inconsistent given data, sorting claims into true, false or unconfirmed from the given data, finding the wrong statements in a short report, and following explicit constraints. Some are in Russian. Each problem asks for its results on labelled lines (`A1: …`, `A2: …`; a Cyrillic look-alike letter in the label, such as `А1:`, counts as the Latin one), and every part is scored in code on its line: a number within a tolerance, or the known value rounded to the answer's own precision when the answer has at least two significant figures (`0.024` for 0.0237, but not `0.025`, `0.0240` or `0.02`; `3` for 2.61 is wrong; a decimal comma is fine, and so are CODATA digit groups such as `9.109 383 7139`), accepted words as whole words (a count may be written as digits or words: `2 up quarks` for `two up quarks`, `2` for `два`), a verdict word at the start of the line, or the exact set of wrong statement ids. No LLM grades anything. A problem counts as solved when all its parts are right. Five problems are held out: they are for a final check only and never run unless you pass `--holdout`.

The runner sends items through the same tools, tool loop and verifier as the chat, in-process, using the configured backend (the model must be running):

```bash
uv run python -m evals.run --limit 20      # 20 items spread over every source/domain group and the reasoning set
uv run python -m evals.run                 # all fact questions and reasoning problems (hours with a local 9B model)
uv run python -m evals.run --set reasoning # only the reasoning problems (or --set facts)
uv run python -m evals.run --holdout       # only the held-out problems: final check
uv run python -m evals.run --ids codata-01,r-stoich-02
uv run python -m evals.run --authored data/evals/authored/b01 --split dev --sets S3,S4 --mode C
```

The fact questions are set S1 and the reasoning problems set S2 in the logs.

`EVAL_SEED=<n>` (an integer ≥ 0, `model_backend.seed`) sends that sampling seed with every model call and records it in the manifest's `sampling`: with the same seed in both arms of an A/B (llama-server `--parallel 1`), an answer the change does not touch comes out the same. Unset or empty, no seed is sent.

### Authored items

Further items are written as YAML files in batch folders (git-ignored, under `data/`). `--authored DIR` (repeatable) loads every `*.yaml` in the folder; then only the authored items are run unless `--set` is given too. `--split dev|holdout|monitor` keeps the authored items of one split, `--sets S3,S4,...` keeps the items of those sets (S1 and S2 included). One file holds a list of items:

```yaml
items:
  - id: b01-s3-01          # unique across batches
    set: S3                # S3, S4, S5, S6 (conversation) or U (user case)
    split: dev             # dev | holdout | monitor
    domain: general        # free text, also used to spread --limit
    lang: en               # en | ru (decimal commas in Russian)
    format: audit          # free text
    defect_tags: []        # tags of the defects planted in the prompt
    turns: []              # earlier messages [{role: user|assistant, content: ...}], sent before the prompt
    prompt: |              # the last user message
      ... Finish with exactly these lines: A1: <verdict for claim A> ...
    key:                   # never sent to the model; shown to the blind reviewer next to the answer
      parts:               # optional: scored in code on labelled answer lines, as in evals/reasoning.yaml
        - {label: A1, verdict: [incorrect]}
        - {label: A2, number: 4.2, rel_tol: 0.01}
      defects: [{id: D1, tag: F1, what: "...", found_if: "..."}]
      verdicts: {A: {verdict: incorrect, why: "..."}}
      required: []         # items the answer must address
      must_not: []         # statements the answer must not make
      notes: ""
      conversation: {max_chars: 600, max_latency_s: 15}   # S6: checked in code
```

Items without `parts` get no code score; the blind review grades them.

### Verdicts read from the answer body

A verdict part tied to one item the task lists (`C1.`, `У3)`, `S4:` at a line start) is read where the answer discusses that item, as a reader would, not from the closing line: the part's label is the item (`C1: <Correct / Incorrect / Unverified>`) or its placeholder names that item alone (`A4: <verdict for C1>`). The item's section starts at a line that begins, after Markdown markers, with its id (`**C1:**`, `### C1 —`, `- C1:`, a table row `| C1 |`) and ends at the next item, a `---` line, a heading or an answer line; the answer lines at the end are never part of a section. Its verdict is the first labelled verdict line (`Verdict:`, `Status:`, `Conclusion:`, `Вердикт:`, `Статус:`, `Вывод:`), else the verdict word right after the id or in a cell of its row, with quoted text left out (`**S8:** "CO is the oxidising agent" is wrong` gives none: the quote restates the claim). Verdict words are those of the task's verdict parts, matched as on a line. When the item has no section, its sections give no verdict, or they give two different ones, the closing line counts as before; numbers, names, id lists and verdict parts tied to no item are always read from their lines. Each score keeps `line_parts` (every part from its line) and `sources` (`body` or `line` per part). `uv run python -m evals.rescore RUN [RUN ...]` scores recorded runs again from their stored answers, without a model, in place (the first rescore keeps the old scores as `scores_before_rescore`; a log whose prompt changed since stays as it is) and prints, per run, parts before → after, the line-only total and the verdict parts read from the body; `--dry-run` only prints.

### Modes

`--mode A|B|C|D` switches pipeline components off in-process, before the agent is built (`evals/modes.yaml`); prompts and sampling stay the same. Without `--mode` the config applies as it is (mode `config`).

| Mode | Tools (`max_search_depth`) | Claim verification | Arithmetic recheck, checkers | Consistency review |
|---|---|---|---|---|
| A | off (0): no tools, no tool prompt | off | off, none | off |
| B | 3 | off | off, none | off |
| C | 3 | on | on, `element_balance` | off |
| D | 3 | on | on, `element_balance` | on |

### Run folders

Each run writes `${DATA_DIR:-data}/evals/<run-id>/` in the data folder (default id `<yyyymmdd-hhmm>-<mode>`, or `--run-id ID`):

- `manifest.json`: run id, git commit and dirty flag (also for a Windows checkout or worktree read from WSL), date, mode, `MODEL_PROFILE`, model id and backend, sha256 of the config after the mode, the model file the backend serves (llama-server's `/props` when it answers, else `MODEL_PATH` or the configured path, else the WSL copy in `$MODELS_DIR`) and its sha256 (cached by path, size and mtime in `${DATA_DIR:-data}/evals/.sha-cache.json`), the sha256 of the uncommitted changes when the tree is dirty (diff against the commit plus untracked files), the knowledge setting (`KNOWLEDGE`), `stores`: every store of `retrieval.stores` with its kind, whether the setting enables it, path, size and mtime, and every other store path in the `retrieval` settings except `raw_dir` (for a folder, the total size and newest mtime of its files), eval files and their sha256, GPU (`nvidia-smi`; null when unavailable), context size, sampling settings and the command-line arguments.
- `items/<id>.json`, written as soon as the item is done: set, mode, the messages sent, the draft, the final answer, the reasoning trace, the verification report, timings (first reasoning token, first answer token, total), model calls, character counts, error, code scores of the final answer and of the draft, and the protocol artefacts found in the final answer.
- `summary.json`: the metrics for all items and per set.

Run again with the same `--run-id` to resume: items with a log are skipped, and with `--retry-errors` the failed ones run again. A run is resumed only when nothing that defines it changed: mode, config, commit, dirty flag and uncommitted changes, model file sha256, every store's path, size and mtime, and sampling; each eval file it uses must be in the manifest with the same sha256 (fewer files are fine). Anything else is refused with the names of the fields that differ: use a new `--run-id`. A reviewed run keeps its review metrics.

Code metrics (per set): items, errors, empty answers, solved items, fact questions right strictly and loosely, parts right (and right on their lines only), verdict parts right (and how many were read from the body), missing answer lines, false corrections (a part right in the draft and wrong in the final answer) and true corrections, answers with protocol artefacts (`[S#]` ids, the draft marker, verification or consistency headings, outcome labels, raw tool-call markup; tool-call leaks are also counted on their own), the S6 checks (no artefacts, at most `max_chars`, at most `max_latency_s`), latency median and 90th percentile, model calls, and the claim outcomes and fixes of the verifier (supported claims, unmarked contradictions, consistency, arithmetic and checker fixes).

### Report

```bash
uv run python -m evals.report 20261023-1400-A 20261023-1500-B             # one row per run and set
uv run python -m evals.report 20261023-1500-B --base 20261023-1400-A      # plus paired deltas on the same items
```

It prints markdown tables: code metrics per set and mode (with the fact questions' strict and loose scores, `strict / loose of n`, the parts right line-only next to the body-first parts, and the verdict parts read from the body; `–` for runs scored before body reading), review metrics when the run was reviewed, and with `--base` the change on the items both runs have (solved, parts right, items better and worse, artefacts, latency, review score).

### Blind review

```bash
uv run python -m evals.review pack --runs 20261023-1400-A 20261023-1500-B --out data/evals/review-01 [--seed N]
uv run python -m evals.review merge data/evals/review-01
```

`pack` shuffles every answer of the runs (failed items are left out), gives each a blind id (R001, …) and writes `pack.md` (blind id, prompt, key, final answer: nothing about run, mode, model or trace) and `mapping.json` (blind id to run and item, and the seed; not for the reviewer). The reviewer writes `reviews.yaml` next to them, one block per answer:

```yaml
- blind_id: R001
  score: 6                       # 0-10
  defects_found: [D1]            # ids from the key
  defects_missed: []
  verdicts: {A: incorrect}       # the verdict the answer gave per key item
  new_errors: [{tag: F4, what: "..."}]
  omissions: []                  # required items not addressed
  unsupported_additions: 0
  unsupported_as_verified: 0
  final_vs_analysis: 0           # conclusions that contradict the answer's own working
  must_not_violated: []
  reveals: "one line"
```

`merge` checks the file (every blind id reviewed once, score 0-10, counts whole numbers, defect and verdict ids from the key) and lists every problem it finds. It then writes `reviews.json` into each run folder and adds the review metrics to its `summary.json`: mean score, share of planted defects found, verdicts right, omissions, unsupported additions, unsupported statements presented as verified, final-vs-analysis contradictions, must-not violations and new errors by tag.
