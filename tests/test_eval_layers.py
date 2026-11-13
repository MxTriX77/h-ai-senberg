"""Eval arms (eval-runner spec "Bare arm", "Layers", "Layer arms", "Arm and layers in the manifest"): the bare
call, the layer registry, arms built by switching layers, and the proof that every layer off is the bare call."""

import json
import shutil

import httpx
import pytest

import evals.run as run
from conftest import REPO, content, finish, reasoning
from evals import layers
from evals.bare import Bare, bare_request
from evals.run import load_modes, manifest, prepare, read_logs, run_eval, run_one
from general_items import item, ready, write_items
from haisenberg.agent import Agent
from haisenberg.config import load_config
from haisenberg.tools import Knowledge

TOOLS = ["search_knowledge", "lookup_compound", "lookup_constant", "calculate", "compute"]
TOOL_LAYERS = [f"tool.{t}" for t in TOOLS]
NO_TOOLS = "tool,tool_prompt,record_check,judge,research,written_calculations,weak_hit_filter"
TURNS = [{"role": "user", "content": "Aconcagua?"}, {"role": "assistant", "content": "About 6,961 m."}]
ENV = ("MODEL_PROFILE", "KNOWLEDGE", "REASONING_BUDGET", "CTX_SIZE", "EVAL_SEED", "API_BASE_URL", "MODEL_ID", "API_KEY")


@pytest.fixture
def shipped(tmp_path, built_store, monkeypatch):
    """The shipped config as an eval run loads it (no test pins), with the fixture store as its knowledge store."""
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "no-models"))  # never hash a real model file for a manifest
    dest = tmp_path / "config" / "config.yaml"
    dest.parent.mkdir(parents=True)
    shutil.copy(REPO / "config" / "config.yaml", dest)
    cfg = load_config(dest)
    cfg.model_backend.wait_s = 0
    cfg.retrieval.stores["encyclopedia"].path = built_store
    cfg.verifier.checkers = []  # whatever ships: the tests name the checkers they use
    return cfg


def on(cfg):
    return {name for name, state in layers.switches(cfg).items() if state}


def knowledge(cfg):
    return Knowledge.open(cfg)


# --- the registry ------------------------------------------------------------------------------------


def test_every_layer_has_a_name_and_the_shipped_suite_has_them_on(shipped):
    names = [layer.name for layer in layers.registry(shipped)]
    assert names == ["system_prompt", "tool_prompt", *TOOL_LAYERS, "thinking_budget", "text_tool_calls",
                     "answer_retry", "written_calculations", "weak_hit_filter", "arithmetic", "carry_forward",
                     "checker.element_balance", "record_check", "judge", "research", "review", "ledger", "writer",
                     "independent", "cite_stripping"]
    off = {"independent", "checker.element_balance"}
    assert on(shipped) == set(names) - off
    assert layers.effective(shipped) == {n: n not in off for n in names}
    assert layers.effective(shipped, knowledge(shipped)) == layers.effective(shipped)  # the store offers every tool


def test_mode_a_is_not_the_bare_model(shipped):
    run.apply_mode(shipped, load_modes()["A"])
    acts = layers.effective(shipped)
    assert not any(acts[n] for n in (*TOOL_LAYERS, "tool_prompt", "arithmetic", "judge", "research", "review"))
    assert all(acts[n] for n in ("system_prompt", "thinking_budget", "ledger", "writer", "answer_retry"))
    assert not acts["weak_hit_filter"] and layers.switches(shipped)["weak_hit_filter"]  # on, but nothing to filter


def test_a_layer_counts_only_with_what_it_needs_and_with_the_store(shipped, tmp_path):
    shipped.reasoning_policy.tools_off = list(TOOLS)
    acts = layers.effective(shipped)
    assert not acts["tool_prompt"] and not acts["record_check"] and not acts["written_calculations"]
    assert acts["judge"] and acts["research"] and acts["weak_hit_filter"]  # the re-search holds the judge up
    shipped.reasoning_policy.tools_off = []
    shipped.retrieval.stores["encyclopedia"].path = tmp_path / "no-store.db"
    acts = layers.effective(shipped, knowledge(shipped))
    assert not any(acts[n] for n in (*TOOL_LAYERS, "tool_prompt", "arithmetic", "judge", "review"))
    assert acts["system_prompt"] and acts["ledger"] and acts["cite_stripping"]


def test_list_layers(shipped):
    lines = layers.table(shipped).splitlines()
    assert lines[0].split() == ["layer", "state", "switch", "needs", "adds"] and len(lines) == 1 + len(layers.registry(shipped))
    row = {line.split()[0]: line for line in lines}
    assert "on " in row["judge"] and "verifier.claim_judge" in row["judge"] and "or research" in row["judge"]
    assert " off " in row["independent"] and "ledger" in row["independent"]


# --- arms --------------------------------------------------------------------------------------------


def test_the_suite_minus_one_layer(shipped):
    before, was = on(shipped), shipped.model_dump()
    name, arm = layers.apply_arm(shipped, layers.CONFIG, minus=["review"])
    assert (name, arm) == ("config-minus-review", {"base": "config", "plus": [], "minus": ["review"]})
    assert on(shipped) == before - {"review"} and shipped.verifier.consistency_check is False
    now = shipped.model_dump()
    now["verifier"]["consistency_check"] = True
    assert now == was  # nothing else moved


def test_writing_the_configured_layers_back_changes_nothing(shipped):
    was = shipped.model_dump_json()
    assert layers.apply_arm(shipped) == ("config", {"base": "config", "plus": [], "minus": []})
    assert shipped.model_dump_json() == was


def test_needs_are_errors_not_repairs(shipped):
    with pytest.raises(ValueError, match="research needs judge"):
        layers.apply_arm(shipped, layers.CONFIG, minus=["judge"])
    assert on(shipped) >= {"judge", "research"}  # a refused arm leaves the config alone
    assert layers.apply_arm(shipped, layers.CONFIG, minus=["judge", "research"])[0] == "config-minus-judge+research"
    with pytest.raises(ValueError, match="carry_forward needs arithmetic"):
        layers.apply_arm(shipped, layers.BARE, plus=["carry_forward"])
    with pytest.raises(ValueError, match="writer needs ledger"):
        layers.apply_arm(shipped, layers.BARE, plus=["writer"])
    with pytest.raises(ValueError, match="tool_prompt needs tool.search_knowledge or tool.lookup_compound or"):
        layers.apply_arm(shipped, layers.BARE, plus=["tool_prompt"])
    with pytest.raises(ValueError, match="judge needs tool.search_knowledge or .* or research"):
        layers.apply_arm(shipped, layers.BARE, plus=["judge"])


@pytest.mark.parametrize("base, plus, minus, error", [
    ("config", [], ["reviews"], "unknown layer 'reviews'; known: system_prompt, tool_prompt, "),
    ("config", [], ["independent"], "layer independent is already off"),
    ("config", ["review"], [], "layer review is already on"),
    ("config", ["independent"], ["independent"], "layer independent is already off"),
    ("bare", [], ["review"], "--minus needs the configured suite"),
    ("config", ["checker"], ["checker"], "every checker layer is already off"),
])
def test_arms_that_make_no_sense_are_refused(shipped, base, plus, minus, error):
    with pytest.raises(ValueError, match=error):
        layers.apply_arm(shipped, base, plus, minus)


def test_a_layer_without_a_value_cannot_be_switched_on(shipped):
    shipped.verifier.max_research = 0
    with pytest.raises(ValueError, match="verifier.max_research is 0 in the config"):
        layers.apply_arm(shipped, layers.CONFIG, plus=["research"])
    shipped.retrieval.min_term_share = 0
    with pytest.raises(ValueError, match="retrieval.min_term_share is 0 in the config"):
        layers.apply_arm(shipped, layers.CONFIG, plus=["weak_hit_filter"])
    shipped.reasoning_policy.thinking_budget.field = ""
    with pytest.raises(ValueError, match="thinking_budget.field is empty"):
        layers.apply_arm(shipped, layers.BARE, plus=["thinking_budget"])
    shipped.reasoning_policy.max_search_depth = 0
    with pytest.raises(ValueError, match="max_search_depth is 0"):
        layers.apply_arm(shipped, layers.BARE, plus=["tool.calculate"])


def test_a_group_name_stands_for_its_layers(shipped):
    name, arm = layers.apply_arm(shipped, layers.CONFIG, minus=NO_TOOLS.split(","))
    assert arm["minus"] == ["tool_prompt", *TOOL_LAYERS, "written_calculations", "weak_hit_filter", "record_check",
                            "judge", "research"]
    assert name == "config-minus-" + "+".join(arm["minus"])
    p = shipped.reasoning_policy
    assert p.tools_off == TOOLS[:-1] and p.compute_tool is False and p.max_search_depth == 3
    k = knowledge(shipped)
    assert k.tools == [] and k.available and layers.runtime_problems(shipped, k) == []
    assert on(shipped) >= {"arithmetic", "carry_forward", "review", "system_prompt", "ledger"}


def test_a_plus_on_the_configured_suite_and_a_checker(shipped):
    name, arm = layers.apply_arm(shipped, layers.CONFIG, plus=["independent", "checker"], minus=["writer"])
    assert name == "config-plus-checker.element_balance+independent-minus-writer"
    assert shipped.verifier.checkers == ["element_balance"] and shipped.evidence.independent
    assert shipped.evidence.enabled and not shipped.evidence.writer


def test_bare_plus_one_layer_switches_everything_else_off(shipped):
    name, arm = layers.apply_arm(shipped, layers.BARE, plus=["arithmetic"])
    assert (name, arm) == ("bare-plus-arithmetic", {"base": "bare", "plus": ["arithmetic"], "minus": []})
    assert on(shipped) == {"arithmetic"}
    p, v, e = shipped.reasoning_policy, shipped.verifier, shipped.evidence
    assert p.enable_multi_hop_verification and v.arithmetic_check and not v.carry_fixes_forward
    assert not (p.use_system_prompt or p.use_tool_prompt or p.thinking_budget.enabled or p.empty_answer_retry
                or p.text_tool_calls or v.claim_judge or v.record_check or v.consistency_check or v.max_research
                or v.strip_source_ids or e.enabled or shipped.retrieval.min_term_share)
    assert p.tools_off == TOOLS[:-1] and not p.compute_tool


def test_a_layer_arm_needs_its_store(shipped, tmp_path):
    layers.apply_arm(shipped, layers.BARE, plus=["arithmetic", "tool.calculate"])
    shipped.retrieval.stores["encyclopedia"].path = tmp_path / "no-store.db"
    assert layers.runtime_problems(shipped, knowledge(shipped)) == [
        "tool.calculate is on but the knowledge store does not offer that tool",
        "arithmetic is on but no knowledge store is open (the agent runs the verifier only with it)"]


# --- the bare call -----------------------------------------------------------------------------------


async def bare_body(cfg, fake, q, script=None):
    fake.scripts = [script or [content("fine")]]
    log = await run_one(Bare(cfg, fake.client(cfg.model_backend)), q, "bare", cfg.evals)
    return fake.last_body, log


@pytest.mark.anyio
async def test_bare_is_one_plain_request(shipped, fake):
    shipped.model_backend.seed = 7
    prepare(shipped, load_modes(), "bare")
    q = ready(item("facts-dev-001", turns=TURNS, prompt="And Kilimanjaro?"))
    body, log = await bare_body(shipped, fake, q, [reasoning("hm"), content("  5,895 m.\n"), finish("stop")])
    b = shipped.model_backend
    assert body == {"model": b.model_id, "messages": [*TURNS, {"role": "user", "content": "And Kilimanjaro?"}],
                    "stream": True, "temperature": b.temperature, "max_tokens": 7168, "seed": 7,
                    "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0,
                    "stream_options": {"include_usage": True}}
    assert len(fake.requests) == 1
    assert (log["answer"], log["draft"], log["verification"], log["model_calls"]) == ("5,895 m.", "5,895 m.", {}, 1)
    assert (log["mode"], log["finish_reason"], log["error"]) == ("bare", "stop", None)
    assert [(s["stage"], s["kind"], s["fields"]) for s in log["stages"]] == [("bare", "model", {"max_tokens": 7168})]
    assert set(log["timings"]) == {"first_reasoning_s", "first_answer_s", "total_s"}


@pytest.mark.anyio
async def test_bare_system_line_and_backends_without_a_budget(shipped, fake):
    shipped.reasoning_policy.bare_system_prompt = "Answer in the user's language."
    assert bare_request(shipped) == {"system_prompt": "Answer in the user's language.", "max_tokens": 7168}
    body, _ = await bare_body(shipped, fake, ready(item()))
    assert body["messages"][0] == {"role": "system", "content": "Answer in the user's language."}
    assert len(body["messages"]) == 2 and "tools" not in body and "thinking_budget_tokens" not in body
    shipped.reasoning_policy.thinking_budget.field = ""
    body, _ = await bare_body(shipped, fake, ready(item()))
    assert body["max_tokens"] == shipped.model_backend.max_tokens == 4096


@pytest.mark.anyio
async def test_bare_makes_no_second_call_for_an_empty_reply(shipped, fake, tmp_path):
    fake.scripts = [[reasoning("only thinking"), finish("length")]]
    summary = await run_eval(Bare(shipped, fake.client(shipped.model_backend)), [ready(item())], tmp_path / "run",
                             "bare",
                             settings=shipped.evals)
    log = read_logs(tmp_path / "run")[0]
    assert len(fake.requests) == 1 and (log["answer"], log["draft"], log["finish_reason"]) == ("", "", "length")
    assert (summary["all"]["no_answer"], summary["all"]["truncated"]) == (1, 1)


# --- every layer off is the bare call ----------------------------------------------------------------


def all_off(cfg):
    """The agent's config with every layer off: the base of every `--mode bare --plus` arm."""
    assert layers.apply_arm(cfg, layers.BARE)[0] == "bare" and not on(cfg)
    return cfg


@pytest.mark.anyio
@pytest.mark.parametrize("line, turns", [("", []), ("Be brief.", []), ("", TURNS), ("Be brief.\n", TURNS)])
async def test_every_layer_off_sends_the_bare_request(shipped, fake, line, turns):
    shipped.reasoning_policy.bare_system_prompt = line
    prepare(shipped, load_modes(), "bare")
    q = ready(item("coding-dev-001", turns=turns, prompt="Why does table[S1] fail?\n1. first\n2. second\n3. third"))
    reply = [reasoning("hm"), content(" Use table[S2] instead: 2 × 3 = 7.\n"), finish("stop")]
    bare, bare_log = await bare_body(shipped, fake, q, reply)
    fake.scripts = [reply]
    agent = Agent(all_off(shipped), fake.client(shipped.model_backend), knowledge(shipped))
    log = await run_one(agent, q, "bare-plus-nothing", shipped.evals)
    assert fake.bodies[1] == bare and len(fake.requests) == 2  # the same request, key for key, and one call each
    for field in ("answer", "draft", "reasoning", "verification", "model_calls", "finish_reason", "messages"):
        assert log[field] == bare_log[field], field
    assert log["answer"] == "Use table[S2] instead: 2 × 3 = 7."  # no marker removed, no number fixed, no ledger


@pytest.mark.anyio
async def test_bare_plus_arithmetic_is_the_bare_request_and_the_fix(shipped, fake):
    prepare(shipped, load_modes(), "bare")
    q = ready(item("math-dev-001", prompt="How many apples?"))
    bare, _ = await bare_body(shipped, fake, q, [content("So 2 × 3 = 7 apples.")])
    mode, arm = prepare(shipped, load_modes(), "bare", plus=["arithmetic"])
    k = knowledge(shipped)
    assert mode == "bare-plus-arithmetic" and layers.runtime_problems(shipped, k) == []
    fake.scripts = [[content("So 2 × 3 = 7 apples.")]]
    log = await run_one(Agent(shipped, fake.client(shipped.model_backend), k), q, mode, shipped.evals)
    assert fake.bodies[1] == bare
    assert (log["draft"], log["answer"]) == ("So 2 × 3 = 7 apples.", "So 2 × 3 = 6 apples.")
    assert [s["stage"] for s in log["stages"]] == ["reasoner", "arithmetic"]


@pytest.mark.anyio
async def test_the_suite_without_its_tools_offers_none(shipped, fake):
    mode, _ = prepare(shipped, load_modes(), None, minus=NO_TOOLS.split(","))
    fake.scripts = [[content("So 2 × 3 = 7 apples.")], [content("OK")]]
    agent = Agent(shipped, fake.client(shipped.model_backend), knowledge(shipped))
    log = await run_one(agent, ready(item()), mode, shipped.evals)
    first = fake.bodies[0]
    assert "tools" not in first and "tool_choice" not in first and first["thinking_budget_tokens"] == 1024
    assert first["messages"][0]["content"] == shipped.reasoning_policy.system_prompt  # no tool prompt without tools
    assert log["answer"] == "So 2 × 3 = 6 apples." and [s["stage"] for s in log["stages"]] == [
        "reasoner", "arithmetic", "carry_forward", "review"]


# --- prepare and the manifest ------------------------------------------------------------------------


def test_prepare_leaves_the_configured_suite_alone_but_for_the_eval_request(shipped):
    was = shipped.model_dump()
    assert prepare(shipped, load_modes()) == ("config", {"base": "config", "plus": [], "minus": []})
    now = shipped.model_dump()
    assert now["model_backend"]["extra_body"].pop("stream_options") == {"include_usage": True}
    assert now == was
    assert prepare(shipped, load_modes(), "bare") == ("bare", {"base": "bare", "plus": [], "minus": []})
    assert prepare(shipped, load_modes(), "D")[1] is None
    with pytest.raises(ValueError, match="--plus and --minus do not combine with --mode A"):
        prepare(shipped, load_modes(), "A", minus=["review"])


def no_backend(url, timeout):
    raise httpx.ConnectError("refused")


def test_manifest_records_the_arm_and_every_layer(shipped, tmp_path, monkeypatch):
    monkeypatch.setattr(run, "NVIDIA_SMI", ["no-such-nvidia-smi-here"])
    monkeypatch.setattr(httpx, "get", no_backend)
    cache = tmp_path / "cache.json"
    mode, arm = prepare(shipped, load_modes(), None, minus=["review"])
    man = manifest(shipped, "r1", mode, [], [], cache, arm, knowledge(shipped))
    assert (man["mode"], man["arm"]) == ("config-minus-review", {"base": "config", "plus": [], "minus": ["review"]})
    assert man["layers"]["review"] is False and man["layers"]["judge"] is True and man["bare"] is None
    assert man["tools"] == TOOLS and man["sampling"]["stream_options"] == {"include_usage": True}
    assert man["thinking"] == {"field": "thinking_budget_tokens", "enabled": True, "min_tokens": 1024,
                               "max_tokens": 4096, "answer_tokens": 3072, "server_cap": "4096"}
    assert man["evals"]["budgets_s"] == {"chat": 15, "factual": 120, "audit": 600}


def test_bare_manifest_has_every_layer_off(shipped, tmp_path, monkeypatch):
    monkeypatch.setattr(run, "NVIDIA_SMI", ["no-such-nvidia-smi-here"])
    monkeypatch.setattr(httpx, "get", no_backend)
    mode, arm = prepare(shipped, load_modes(), "bare")
    man = manifest(shipped, "r1", mode, [], [], tmp_path / "cache.json", arm)
    assert man["mode"] == "bare" and man["arm"]["base"] == "bare" and man["tools"] == []
    assert set(man["layers"]) == {layer.name for layer in layers.registry(shipped)} and not any(man["layers"].values())
    assert man["bare"] == {"system_prompt": "", "max_tokens": 7168}
    old = manifest(shipped, "r1", "A", [], [], tmp_path / "cache.json")  # the modes A to D have no arm record
    assert old["arm"] is None and old["layers"]["system_prompt"] is True


# --- the command line --------------------------------------------------------------------------------


@pytest.fixture
def cli(tmp_path, built_store, fake, monkeypatch):
    """`evals.run.main` against the fake upstream: a temp config and data folder with the fixture store."""
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "no-models"))  # never hash a real model file for a manifest
    config = tmp_path / "project" / "config" / "config.yaml"
    config.parent.mkdir(parents=True)
    shutil.copy(REPO / "config" / "config.yaml", config)
    data = tmp_path / "data"
    data.mkdir()
    shutil.copy(built_store, data / "encyclopedia_fts.db")
    monkeypatch.setenv("HAISENBERG_CONFIG", str(config))
    monkeypatch.setenv("DATA_DIR", str(data))
    monkeypatch.setattr(run, "NVIDIA_SMI", ["no-such-nvidia-smi-here"])
    monkeypatch.setattr(httpx, "get", no_backend)

    def client(backend):
        backend.wait_s = 0
        return fake.client(backend)

    monkeypatch.setattr(run, "LLMClient", client)
    items = write_items(tmp_path / "items" / "facts.yaml", item("facts-dev-001", turns=TURNS, prompt="And Kilimanjaro?"),
                        item("facts-dev-002", prompt="How many apples?"))
    return data / "evals", items


def read(folder, name):
    return json.loads((folder / name).read_text(encoding="utf-8"))


def test_main_runs_the_bare_arm(cli, fake, monkeypatch, capsys):
    runs, items = cli
    monkeypatch.setenv("EVAL_SEED", "7")
    fake.scripts = [[content("5,895 m.")], [content("So 2 × 3 = 7 apples.")]]
    assert run.main(["--items", str(items), "--mode", "bare", "--run-id", "b1"]) == 0
    logs = read_logs(runs / "b1")
    assert [(r["id"], r["mode"], r["answer"]) for r in logs] == [
        ("facts-dev-001", "bare", "5,895 m."), ("facts-dev-002", "bare", "So 2 × 3 = 7 apples.")]
    assert [b["messages"] for b in fake.bodies] == [[*TURNS, {"role": "user", "content": "And Kilimanjaro?"}],
                                                    [{"role": "user", "content": "How many apples?"}]]
    assert all(b["seed"] == 7 and b["max_tokens"] == 7168 and "tools" not in b for b in fake.bodies)
    man = read(runs / "b1", "manifest.json")
    assert man["mode"] == "bare" and not any(man["layers"].values()) and man["sampling"]["seed"] == 7
    out = capsys.readouterr().out
    assert "all: 2 items, 0 errors, 0 without an answer" in out and "analytic: median" in out and "no budget" in out


def test_main_runs_a_layer_arm_and_names_it(cli, fake, capsys):
    runs, items = cli
    fake.scripts = [[content("5,895 m.")], [content("So 2 × 3 = 7 apples.")]]
    assert run.main(["--items", str(items), "--mode", "bare", "--plus", "arithmetic", "--run-id", "p1"]) == 0
    logs = read_logs(runs / "p1")
    assert [r["answer"] for r in logs] == ["5,895 m.", "So 2 × 3 = 6 apples."] and logs[0]["mode"] == "bare-plus-arithmetic"
    man = read(runs / "p1", "manifest.json")
    assert man["arm"] == {"base": "bare", "plus": ["arithmetic"], "minus": []} and man["tools"] == []
    assert [n for n, state in man["layers"].items() if state] == ["arithmetic"]
    assert man["bare"] is None  # the agent ran it, with a layer on: not the bare call
    assert all("tools" not in b and "thinking_budget_tokens" not in b for b in fake.bodies)


def test_main_refuses_a_layer_arm_without_its_store(cli, fake, capsys):
    runs, items = cli
    (runs.parent / "encyclopedia_fts.db").unlink()
    assert run.main(["--items", str(items), "--mode", "bare", "--plus", "arithmetic", "--run-id", "x1"]) == 1
    assert "arithmetic is on but no knowledge store is open" in capsys.readouterr().err
    assert not (runs / "x1").exists() and not fake.requests
    assert run.main(["--items", str(items), "--minus", "judge", "--run-id", "x2"]) == 1
    assert "research needs judge" in capsys.readouterr().err and not (runs / "x2").exists()


def test_main_says_that_mode_a_is_not_bare(cli, fake, capsys):
    runs, items = cli
    fake.chunks = [content("fine")]
    assert run.main(["--items", str(items), "--mode", "A", "--run-id", "a1", "--ids", "facts-dev-002"]) == 0
    err = capsys.readouterr().err
    assert "mode A is not the bare model" in err and "--mode bare" in err
    assert fake.bodies[0]["messages"][0]["role"] == "system" and "thinking_budget_tokens" in fake.bodies[0]
    assert read(runs / "a1", "manifest.json")["layers"]["system_prompt"] is True


def test_main_lists_the_layers_of_an_arm(cli, fake, capsys):
    def states():
        lines = capsys.readouterr().out.splitlines()
        return lines[0], {line.split()[0]: line.split()[1] for line in lines[2:]}

    assert run.main(["--list-layers"]) == 0
    arm, state = states()
    assert arm == "arm: config" and state["judge"] == "on" and state["independent"] == "off"
    assert run.main(["--list-layers", "--minus", "judge,research"]) == 0
    arm, state = states()
    assert arm == "arm: config-minus-judge+research" and (state["judge"], state["review"]) == ("off", "on")
    assert run.main(["--list-layers", "--mode", "bare", "--plus", "arithmetic"]) == 0
    arm, state = states()
    assert arm == "arm: bare-plus-arithmetic" and [n for n, s in state.items() if s != "off"] == ["arithmetic"]
    assert run.main(["--list-layers", "--mode", "bare"]) == 0
    arm, state = states()
    assert arm == "arm: bare" and set(state.values()) == {"off"}  # the bare call uses no layer
    assert not fake.requests
