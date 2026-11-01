"""Eval infrastructure (eval-runner spec): authored items, modes, item logs, resume, manifest, report and
blind review. Synthetic items only, the fake upstream, no GPU."""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import httpx
import pytest
import yaml
from pydantic import ValidationError

import evals.run as run
from conftest import CONNECT_ERROR, content
from evals.report import report
from evals.rescore import main as rescore_main
from evals.unsourced import main as unsourced_main
from evals.review import check_reviews, merge, pack
from evals.run import (
    QUESTIONS,
    apply_mode,
    artefacts,
    choose,
    file_sha256,
    load_authored,
    load_modes,
    manifest,
    open_run,
    run_dir,
    runs_dir,
    served_model,
    wsl_path,
    read_logs,
    run_eval,
    run_one,
    summarize,
    write_summary,
)
from haisenberg.agent import DRAFT_MARK, Agent
from haisenberg.l1.store import L1Store
from haisenberg.tools import Knowledge
from haisenberg.verifier import SUPPORTED, UNVERIFIED, render, split_claims, strip_cites

ITEMS = {"items": [
    {"id": "b1-s3-01", "set": "S3", "split": "dev", "domain": "general", "lang": "en", "format": "audit",
     "defect_tags": ["F1"], "prompt": "Check the claims. Finish with A1: <verdict> and A2: <number>.",
     "key": {"parts": [{"label": "A1", "verdict": ["incorrect"]}, {"label": "A2", "number": 4.2, "rel_tol": 0.01}],
             "defects": [{"id": "D1", "tag": "F1", "what": "w", "found_if": "f"}],
             "verdicts": {"A": {"verdict": "incorrect", "why": "y"}}}},
    {"id": "b1-s6-01", "set": "S6", "split": "holdout", "prompt": "And now?",
     "turns": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
     "key": {"conversation": {"max_chars": 20, "max_latency_s": 60}}},
]}
MORE = {"items": [{"id": "b1-u-01", "set": "U", "split": "monitor", "prompt": "Plan my week."}]}


def batch(tmp_path, *files):
    folder = tmp_path / "b1"
    folder.mkdir(exist_ok=True)
    for n, data in enumerate(files):
        (folder / f"f{n}.yaml").write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return folder


def no_store(cfg):
    return Knowledge(None, cfg)


# --- authored items ---------------------------------------------------------------------------------


def test_authored_items_load_and_filter(tmp_path):
    items = load_authored(batch(tmp_path, ITEMS, MORE))
    assert [i["id"] for i in items] == ["b1-s3-01", "b1-s6-01", "b1-u-01"]
    s3 = items[0]
    assert s3["question"] == s3["prompt"] and [p["label"] for p in s3["parts"]] == ["A1", "A2"] and s3["turns"] == []
    assert items[2]["parts"] == [] and items[2]["lang"] == "en"
    facts = [{"id": "f1", "set": "S1"}]
    assert [i["id"] for i in choose(facts + items, split="dev")] == ["f1", "b1-s3-01"]
    assert [i["id"] for i in choose(facts + items, sets=["S6", "U"])] == ["b1-s6-01", "b1-u-01"]
    assert [i["id"] for i in choose(facts + items, split="monitor", sets=["S1", "U"])] == ["f1", "b1-u-01"]


@pytest.mark.parametrize("item, error", [
    ({"id": "x", "set": "S3", "split": "dev"}, "needs set"),
    ({"id": "x", "set": "S9", "split": "dev", "prompt": "p"}, "needs set"),
    ({"id": "x", "set": "S3", "split": "later", "prompt": "p"}, "needs set"),
    ({"id": "x", "set": "S3", "split": "dev", "prompt": "p", "turns": [{"role": "system", "content": "c"}]}, "turns"),
    ({"id": "x", "set": "S3", "split": "dev", "prompt": "p", "key": {"parts": [{"label": "A1", "number": 1}]}},
     "part A1: a number needs abs_tol or rel_tol"),
    ({"id": "x", "set": "S6", "split": "dev", "prompt": "p", "key": {"conversation": {"max_chars": 9}}},
     "max_latency_s"),
])
def test_bad_authored_item(tmp_path, item, error):
    with pytest.raises(ValueError, match=f"item x: .*{error}"):
        load_authored(batch(tmp_path, {"items": [item]}))


def test_duplicate_ids_across_files(tmp_path):
    with pytest.raises(ValueError, match="duplicate ids"):
        load_authored(batch(tmp_path, ITEMS, ITEMS))


# --- modes ------------------------------------------------------------------------------------------


def test_modes(cfg):
    modes = load_modes()
    assert set(modes) == {"A", "B", "C", "D"}
    expected = {"A": (0, False, False, [], False), "B": (3, False, False, [], False),
                "C": (3, True, True, ["element_balance"], False), "D": (3, True, True, ["element_balance"], True)}
    for mode, want in expected.items():
        apply_mode(cfg, modes[mode])
        p, v = cfg.reasoning_policy, cfg.verifier
        assert (p.max_search_depth, p.enable_multi_hop_verification, v.arithmetic_check, v.checkers,
                v.consistency_check) == want, mode
    with pytest.raises(ValidationError):
        apply_mode(cfg, {"verifier": {"checkers": ["no_such_checker"]}})


@pytest.mark.anyio
async def test_mode_a_sends_no_tools_and_same_sampling(cfg, fake, built_store, tmp_path):
    item = {"id": "i1", "set": "S5", "question": "q", "parts": [], "lang": "en"}
    bodies = {}
    for mode in ("A", "B"):
        apply_mode(cfg, load_modes()[mode])
        knowledge = Knowledge(L1Store(built_store), cfg)
        fake.scripts = [[content("fine")]]
        await run_eval(Agent(cfg, fake.client(cfg.model_backend), knowledge), [item], tmp_path / mode, mode)
        bodies[mode] = fake.last_body
    a, b = bodies["A"], bodies["B"]
    assert "tools" not in a and "tool_choice" not in a and a["messages"][0]["content"] == cfg.reasoning_policy.system_prompt
    assert b["tools"] and cfg.reasoning_policy.tool_prompt in b["messages"][0]["content"]
    sampling = ("model", "temperature", "max_tokens", *cfg.model_backend.extra_body)
    assert {k: a[k] for k in sampling} == {k: b[k] for k in sampling}
    assert read_logs(tmp_path / "A")[0]["mode"] == "A"


# --- item log: draft, scores, corrections, artefacts ----------------------------------------------------


@pytest.mark.anyio
async def test_draft_scores_and_corrections(cfg, fake, built_store, tmp_path):
    """Mode D: the consistency review turns a right part wrong (false correction) and a wrong one right."""
    apply_mode(cfg, load_modes()["D"])
    cfg.verifier.max_research = 0  # no re-search: the review is the only call after the draft
    cfg.verifier.consistency_quotes = False  # the review's unquoted edits stand in for any false/true correction
    cfg.verifier.consistency_keep_answer_lines = False  # ... and they change answer lines
    item = {"id": "s4-1", "set": "S4", "question": "Q?", "lang": "en",
            "turns": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
            "parts": [{"label": "A1", "number": 6, "abs_tol": 0}, {"label": "A2", "verdict": ["true"]},
                      {"label": "A3", "text": ["x"]}]}
    fake.scripts = [[content("A1: 6\nA2: false")],
                    [content("OLD: A1: 6\nNEW: A1: 7\nWHY: w\nOLD: A2: false\nNEW: A2: true\nWHY: w")]]
    agent = Agent(cfg, fake.client(cfg.model_backend), Knowledge(L1Store(built_store), cfg))
    summary = await run_eval(agent, [item], tmp_path / "run", "D")
    r = read_logs(tmp_path / "run")[0]
    assert r["messages"] == [*item["turns"], {"role": "user", "content": "Q?"}]
    assert [m["role"] for m in fake.bodies[0]["messages"]] == ["system", "user", "assistant", "user"]
    assert (r["draft"], r["answer"], r["mode"], r["model_calls"], r["error"]) == ("A1: 6\nA2: false", "A1: 7\nA2: true",
                                                                                  "D", 2, None)
    assert r["scores"]["draft"]["parts"] == {"A1": True, "A2": False, "A3": False}
    parts = {"A1": False, "A2": True, "A3": False}
    assert r["scores"]["final"] == {"correct": False, "parts": parts, "line_parts": parts,
                                    "sources": dict.fromkeys(parts, "line"), "missing": ["A3"], "verdicts": {"A2": True}}
    assert set(r["timings"]) == {"first_reasoning_s", "first_answer_s", "total_s"}
    assert r["chars"] == {"reasoning": len(r["reasoning"]), "draft": len(r["draft"]), "answer": len(r["answer"])}
    assert len(r["verification"]["consistency"]) == 2
    assert r["artefacts"] == [] and {"draft_marker", "heading"} <= set(artefacts(r["reasoning"]))
    s = summary["sets"]["S4"]
    assert (s["false_corrections"], s["true_corrections"], s["draft_parts_right"], s["parts_right"]) == (1, 1, 1, 1)
    assert (s["missing_lines"], s["verdict_parts_right"], s["verdict_parts_total"], s["model_calls"]) == (1, 1, 1, 2)


def test_artefacts_follow_the_code():
    draft = "Water boils at 100 °C [S1]. Ethanol boils at 90 °C."
    claims = split_claims(draft)
    claims[0].outcome, claims[0].ids = SUPPORTED, ["S1"]
    claims[1].outcome = UNVERIFIED
    assert set(artefacts(render(draft, claims, annotated=True))) == {"source_id", "outcome_label", "heading"}
    assert artefacts(render(draft, claims, annotated=False)) == []
    assert artefacts(f"x{DRAFT_MARK}y") == ["draft_marker"]
    for text in ("a [S1].", "a [S1, S2] b", "a [S1; 3]", "S1 is fine", "[s1]"):
        assert ("source_id" in artefacts(text)) == (strip_cites(text) != text), text
    leak = "The result follows.\n<tool_call>\n<function=calculate>\n<parameter=expression>\n2*3\n</parameter>"
    assert artefacts(leak) == ["tool_call"]
    s = summarize([{"id": "x", "set": "S5", "answer": leak, "error": None, "verification": {}, "timings": {"total_s": 1},
                    "model_calls": 1, "scores": None, "artefacts": artefacts(leak), "unmarked_contradiction": False}])
    assert (s["with_artefacts"], s["tool_call_leaks"], s["artefacts"]) == (1, 1, {"tool_call": 1})


@pytest.mark.anyio
async def test_s6_conversation_checks(cfg, fake, tmp_path):
    conv = {"key": {"conversation": {"max_chars": 20, "max_latency_s": 60}}, "set": "S6", "parts": []}
    items = [{"id": "c1", "question": "a", **conv}, {"id": "c2", "question": "b", **conv},
             {"id": "c3", "question": "c", **conv, "key": {"conversation": {"max_chars": 20, "max_latency_s": 0.01}}}]
    fake.scripts = [[content("Hello there.")], [content("A much longer answer than allowed [S1]")], [content("Hi.")]]
    fake.delay_s = 0.02  # every answer takes longer than c3's limit
    summary = await run_eval(Agent(cfg, fake.client(cfg.model_backend), no_store(cfg)), items, tmp_path / "run")
    logs = {r["id"]: r for r in read_logs(tmp_path / "run")}
    assert logs["c1"]["conversation"] == {"artefact_free": True, "max_chars": True, "max_latency": True}
    assert logs["c2"]["conversation"] == {"artefact_free": True, "max_chars": False, "max_latency": True}  # plain: no ids
    assert logs["c3"]["conversation"]["max_latency"] is False and logs["c1"]["scores"] is None
    assert summary["sets"]["S6"]["conversation"] == {"items": 3, "artefact_free": 3, "max_chars": 2, "max_latency": 2}


# --- resume, manifest ---------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_resume_and_retry_errors(cfg, fake, tmp_path):
    items = [{"id": f"q{n}", "set": "S1", "question": "bp?", "answer": {"number": 80.1, "abs_tol": 0.5}} for n in (1, 2)]
    agent = Agent(cfg, fake.client(cfg.model_backend), no_store(cfg))
    folder = tmp_path / "run"
    fake.scripts = [[content("A1: 80.1 °C")], CONNECT_ERROR]
    s = await run_eval(agent, items, folder)
    assert s["all"]["errors"] == 1 and "unreachable" in read_logs(folder)[1]["error"]
    s = await run_eval(agent, items, folder)  # resume: nothing left to do
    assert len(fake.requests) == 2 and s["all"]["errors"] == 1
    fake.scripts = [[content("A1: 80.2 °C")]]
    s = await run_eval(agent, items, folder, retry_errors=True)
    assert len(fake.requests) == 3 and s["all"]["errors"] == 0 and s["all"]["correct"] == 2
    assert read_logs(folder)[0]["answer"] == "A1: 80.1 °C"  # the done item was not run again


RUN = {"mode": "A", "config_sha256": "c1", "git": {"sha": "g1", "dirty": False, "diff_sha256": None}, "date": "d1",
       "model_file": {"path": "m.gguf", "sha256": "m1"}, "gpu": [{"power_draw": "10 W"}],
       "stores": {"layer1_encyclopedia_db": {"path": "s.db", "size": 5, "mtime": "2026-10-25T10:00:00"},
                  "layer2_literature_db": {"path": "l.db", "size": 7, "mtime": "2026-10-25T10:00:00"}},
       "eval_files": [{"path": "evals/questions.yaml", "sha256": "q1"}, {"path": "evals/reasoning.yaml", "sha256": "p1"}],
       "sampling": {"temperature": 0.6, "max_tokens": 100}}


def test_resumed_run_must_match(tmp_path):
    open_run(tmp_path / "r", RUN)
    open_run(tmp_path / "r", {**RUN, "date": "d2", "gpu": [{"power_draw": "60 W"}]})  # nothing that defines it changed
    open_run(tmp_path / "r", {**RUN, "eval_files": RUN["eval_files"][:1]})  # fewer eval files (--ids, --set)
    assert json.loads((tmp_path / "r" / "manifest.json").read_text(encoding="utf-8"))["date"] == "d1"
    store = RUN["stores"]["layer2_literature_db"]
    for change, field in [
        ({"mode": "B"}, "mode"), ({"config_sha256": "c2"}, "config_sha256"),
        ({"git": {**RUN["git"], "sha": "g2"}}, "commit"),
        ({"git": {"sha": "g1", "dirty": True, "diff_sha256": "d1"}}, "dirty, diff_sha256"),
        ({"model_file": {"path": "m.gguf", "sha256": "m2"}}, "model_sha256"),
        ({"stores": {**RUN["stores"], "layer2_literature_db": {**store, "mtime": "2026-10-25T11:00:00"}}}, "stores"),
        ({"stores": {**RUN["stores"], "layer3": store}}, "stores"),
        ({"eval_files": [{"path": "evals/questions.yaml", "sha256": "q2"}]}, "eval_files"),
        ({"eval_files": [{"path": "data/evals/b1/s3.yaml", "sha256": "x"}]}, "eval_files"),
        ({"sampling": {"temperature": 0.7, "max_tokens": 100}}, "sampling"),
    ]:
        with pytest.raises(ValueError, match=f"differs from its manifest in {field}; use a new --run-id"):
            open_run(tmp_path / "r", {**RUN, **change})
    dirty = tmp_path / "dirty"
    open_run(dirty, {**RUN, "git": {"sha": "g1", "dirty": True, "diff_sha256": "d1"}})
    with pytest.raises(ValueError, match="in diff_sha256;"):  # other uncommitted changes on the same commit
        open_run(dirty, {**RUN, "git": {"sha": "g1", "dirty": True, "diff_sha256": "d2"}})
    old = tmp_path / "old"  # made before the stores and the diff hash were recorded
    open_run(old, {k: v for k, v in RUN.items() if k != "stores"} | {"git": {"sha": "g1", "dirty": False}})
    with pytest.raises(ValueError, match="in stores;"):
        open_run(old, RUN)


def test_manifest(cfg, tmp_path, monkeypatch):
    model, store, cache = tmp_path / "m.gguf", tmp_path / "s.db", tmp_path / "evals" / ".sha-cache.json"
    model.write_bytes(b"weights")
    store.write_bytes(b"store")
    index = tmp_path / "index"  # a store folder: the total size and the newest mtime of its files
    (index / "sub").mkdir(parents=True)
    (index / "a.bin").write_bytes(b"12")
    (index / "sub" / "b.bin").write_bytes(b"345")
    os.utime(index / "a.bin", (1e9, 1e9))
    os.utime(index / "sub" / "b.bin", (2e9, 2e9))
    cfg.llama_server.model_path, cfg.retrieval.stores["encyclopedia"].path = model, store
    cfg.retrieval.layer2_literature_index = index
    monkeypatch.setattr(run, "NVIDIA_SMI", ["no-such-nvidia-smi-here"])
    monkeypatch.setattr(httpx, "get", no_backend)
    monkeypatch.setenv("MODEL_PROFILE", "p1")
    man = manifest(cfg, "r1", "A", [QUESTIONS], ["--mode", "A"], cache)
    assert man["model_file"]["sha256"] == hashlib.sha256(b"weights").hexdigest() and man["gpu"] is None
    assert (man["run_id"], man["mode"], man["model_profile"], man["args"], man["seed"]) == ("r1", "A", "p1", ["--mode", "A"], None)
    stores = man["stores"]
    enc = stores["encyclopedia"]  # registry stores carry their kind and whether the knowledge setting enables them
    assert (enc["kind"], enc["enabled"], enc["path"], enc["size"]) == ("encyclopedia", True, str(store), 5)
    assert enc["mtime"] and "raw_dir" not in stores
    assert (stores["literature"]["kind"], stores["literature"]["enabled"]) == ("literature", False)
    assert stores["layer2_literature_index"] == {"path": str(index), "size": 5,
                                                 "mtime": datetime.fromtimestamp(2e9).isoformat(timespec="seconds")}
    cfg.retrieval.layer2_literature_index = tmp_path / "missing"
    assert manifest(cfg, "r1", "A", [], [], cache)["stores"]["layer2_literature_index"]["size"] is None
    assert man["eval_files"] == [{"path": "evals/questions.yaml", "sha256": file_sha256(QUESTIONS)}]
    assert man["context_tokens"] == cfg.context_tokens and man["sampling"]["temperature"] == cfg.model_backend.temperature
    assert set(man["sampling"]) >= {"max_tokens", *cfg.model_backend.extra_body} and "git" in man and man["date"]
    assert man["sampling"]["seed"] is None  # EVAL_SEED unset
    cfg.model_backend.seed = 7
    assert manifest(cfg, "r1", "A", [], [], cache)["sampling"]["seed"] == 7
    entry = json.loads(cache.read_text(encoding="utf-8"))[str(model.resolve())]
    assert entry["sha256"] == man["model_file"]["sha256"]
    # cached by path, size and mtime: a stored digest is reused, a changed file is hashed again
    cache.write_text(json.dumps({str(model.resolve()): {**entry, "sha256": "from-cache"}}), encoding="utf-8")
    assert file_sha256(model, cache) == "from-cache"
    model.write_bytes(b"other weights")
    assert file_sha256(model, cache) == hashlib.sha256(b"other weights").hexdigest()
    before = man["config_sha256"]
    apply_mode(cfg, load_modes()["B"])
    assert manifest(cfg, "r1", "B", [], [], cache)["config_sha256"] != before


def no_backend(url, timeout):
    raise httpx.ConnectError("refused")


def test_gpu_is_a_json_list(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(run, "NVIDIA_SMI", [sys.executable, "-c", "print('GPU X, 80.00 W, 10.50 W, 6141 MiB')"])
    monkeypatch.setattr(httpx, "get", no_backend)
    open_run(tmp_path / "r", manifest(cfg, "r", "A", [], [], tmp_path / "cache.json"))
    written = json.loads((tmp_path / "r" / "manifest.json").read_text(encoding="utf-8"))
    assert written["gpu"] == [{"name": "GPU X", "power_limit": "80.00 W", "power_draw": "10.50 W", "memory_total": "6141 MiB"}]


def test_served_model(cfg, tmp_path, monkeypatch):
    asked = []

    def props(url, timeout):
        asked.append(url)
        return httpx.Response(200, json={"model_path": "/wsl/models/m.gguf"})

    monkeypatch.setattr(httpx, "get", props)
    cfg.model_backend.api_base = "http://gpu-box:1234/v1/"
    assert served_model(cfg) == Path("/wsl/models/m.gguf") and asked == ["http://gpu-box:1234/props"]
    monkeypatch.setattr(httpx, "get", no_backend)  # no llama-server: the configured file, else the WSL copy
    cfg.llama_server.model_path = tmp_path / "models" / "m.gguf"
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "copies"))
    assert served_model(cfg) == tmp_path / "models" / "m.gguf"  # neither exists: the configured path
    (tmp_path / "copies").mkdir()
    (tmp_path / "copies" / "m.gguf").write_bytes(b"w")
    assert served_model(cfg) == tmp_path / "copies" / "m.gguf"
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "m.gguf").write_bytes(b"w")
    assert served_model(cfg) == tmp_path / "models" / "m.gguf"
    monkeypatch.setattr(httpx, "get", lambda url, timeout: httpx.Response(200, json={"model_path": ""}))  # the app's /props
    assert served_model(cfg) == tmp_path / "models" / "m.gguf"


def git(*args, cwd):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
                          cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def test_git_state_and_windows_worktree(tmp_path, monkeypatch):
    repo, tree = tmp_path / "repo", tmp_path / "tree"
    repo.mkdir()
    git("init", "-q", cwd=repo)
    (repo / "a.txt").write_bytes(b"a\r\n")  # checked out by Windows git (core.autocrlf): CRLF on disk, LF stored
    git("-c", "core.autocrlf=true", "add", "a.txt", cwd=repo)
    git("commit", "-q", "-m", "c", cwd=repo)
    sha = git("rev-parse", "HEAD", cwd=repo)
    os.utime(repo / "a.txt", (1e9, 1e9))  # read from WSL: the index's stat data no longer match, git rereads files
    assert git("-c", "core.autocrlf=false", "--no-optional-locks", "status", "--porcelain", cwd=repo)  # plain: modified
    assert run._git(repo) == {"sha": sha, "dirty": False, "diff_sha256": None}
    # every other uncommitted change, untracked files included, has another hash
    hashes = []
    for name, data in [("a.txt", b"bb\r\n"), ("a.txt", b"cc\r\n"), ("new.txt", b"x"), ("new.txt", b"y")]:
        (repo / name).write_bytes(data)
        state = run._git(repo)
        assert state["sha"] == sha and state["dirty"] is True
        hashes.append(state["diff_sha256"])
    assert len(set(hashes)) == 4 and run._git(repo)["diff_sha256"] == hashes[-1]
    # a worktree made by Windows git: its .git file names C:/..., which git here cannot follow
    tree.mkdir()
    (tree / "a.txt").write_bytes(b"a\r\n")
    (tree / ".git").write_text("gitdir: C:/dev/repo/.git\n", encoding="utf-8")
    converted = []
    monkeypatch.setattr(run, "wsl_path", lambda p: converted.append(p) or str(repo / ".git"))
    assert run._git(tree) == {"sha": sha, "dirty": False, "diff_sha256": None} and converted == ["C:/dev/repo/.git"]
    assert run._git(tmp_path / "nothing") is None


def test_wsl_path():
    assert wsl_path("C:/.dev/h-ai-senberg/.git/worktrees/-hai-wt-eng") == "/mnt/c/.dev/h-ai-senberg/.git/worktrees/-hai-wt-eng"
    assert wsl_path("D:\\x\\y") == "/mnt/d/x/y" and wsl_path("/home/u/.git") == "/home/u/.git"


def test_runs_live_in_the_data_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    assert runs_dir() == tmp_path / "data" / "evals" and run_dir("r1") == tmp_path / "data" / "evals" / "r1"
    assert run_dir(str(tmp_path)) == tmp_path  # a folder is taken as it is
    monkeypatch.setenv("DATA_DIR", "")
    assert runs_dir() == run.ROOT / "data" / "evals"


# --- report and review --------------------------------------------------------------------------------


def write_run(folder, mode, logs, eval_file=None):
    (folder / "items").mkdir(parents=True)
    for r in logs:
        (folder / "items" / f"{r['id']}.json").write_text(json.dumps(r), encoding="utf-8")
    man = {"run_id": folder.name, "mode": mode, "model_id": "secret-model",
           "eval_files": [{"path": str(eval_file), "sha256": None}] if eval_file else []}
    (folder / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    write_summary(folder, mode)
    return folder


def item_log(iid, parts, mode="A", answer="answer", error=None, total=2.0, found=(), messages=None):
    final = {"correct": all(parts.values()), "parts": parts, "missing": [], "verdicts": {}}
    return {"id": iid, "set": "S3", "mode": mode, "messages": messages or [{"role": "user", "content": "prompt"}],
            "draft": answer, "answer": answer, "reasoning": "SECRET-TRACE", "verification": {}, "timings": {"total_s": total},
            "model_calls": 1, "chars": {}, "error": error, "correct": final["correct"], "scores": {"final": final, "draft": final},
            "artefacts": list(found), "unmarked_contradiction": False}


def test_report_with_paired_deltas(tmp_path):
    base = write_run(tmp_path / "run-a", "A", [item_log("i1", {"A1": True, "A2": False}),
                                               item_log("i2", {"A1": False, "A2": False}, found=["tool_call"])])
    new = write_run(tmp_path / "run-b", "B", [item_log("i1", {"A1": True, "A2": True}, total=4.0),
                                              item_log("i2", {"A1": False, "A2": False}, total=4.0),
                                              item_log("i3", {"A1": True, "A2": True})])
    text = report([base, new], base)
    assert "| run-a | S3 | A | 2 | 0 | 0 | 0/2 (0 %) | – | 1/4 (25 %) |" in text
    assert "| run-b | all | B | 3 |" in text and "mean score" not in text
    assert "| run-b | S3 | 2 | 0 → 1 | +1 | 1 / 0 | 1 → 0 | +2 | – |" in text  # i3 is not in the base: not paired


def test_report_shows_strict_and_loose_fact_scores(tmp_path):
    logs = [item_log(f"f{n}", {"answer": ok}) for n, ok in enumerate((True, True, False))]
    for r in logs:
        r["set"], r["scores"]["final"]["loose_correct"] = "S1", True
    text = report([write_run(tmp_path / "run-f", "D", logs)])
    for name in ("S1", "all"):
        assert f"| run-f | {name} | D | 3 | 0 | 0 | 2/3 (66.7 %) | 2 / 3 of 3 |" in text


def test_report_shows_unsourced_numbers_per_answer(tmp_path):
    logs = [item_log("i1", {"A1": True}), item_log("i2", {"A1": True})]
    logs[0]["unsourced"], logs[1]["unsourced"] = 3, 0
    lines = report([write_run(tmp_path / "run-u", "D", logs),
                    write_run(tmp_path / "run-old", "D", [item_log("i1", {"A1": True})])]).splitlines()
    assert lines[0].endswith(" | unsourced numbers per answer |")
    assert next(x for x in lines if x.startswith("| run-u | all |")).endswith(" | 1.5 |")
    assert next(x for x in lines if x.startswith("| run-old | all |")).endswith(" | – |")  # logged before the flag


def test_offline_unsourced_counts_and_sample(tmp_path, capsys):
    prompt = "The report claims a yield of 81.7 % from 20.40 g."
    trace = ('thinking: 99.9 % sounds right\n\n> calculate(expressions=["20.40 / 24.97"])\n- 20.40 / 24.97 = 0.8169803764\n\n'
             '> search_knowledge(query="aspirin purity")  [re-search C1]\n- [S1] Aspirin\n- C1 judge: supports S1, 99.9 %\n')
    answer = ("The ratio is 0.817 and the yield 81.7 % [S1]; aspirin melts at 135 °C. Purity 99.9 %. Students get 60–70 %."
              "\n\n*Verification: 12 claims: 11 supported, 1 unverified, 0 corrected, 0 removed.*\n\nSources:\n- [S1] Aspirin")
    logs = [item_log("i1", {"A1": True}, answer=answer, messages=[{"role": "user", "content": prompt}]),
            item_log("i2", {"A1": True}, answer=" ")]  # no answer: not counted
    logs[0]["reasoning"] = trace  # the calculator result is a source; the model's thinking and the judge are not
    logs[0]["verification"] = {"claims": [{"text": "t", "outcome": "supported", "ids": ["S1"], "note": "code: 135 ≈ 135 (S1)"}]}
    folder = write_run(tmp_path / "run-x", "D", logs)
    held = write_run(tmp_path / "run-h", "D", logs, tmp_path / "holdout-v1" / "s3.yaml")
    assert unsourced_main([str(folder), str(held), "--sample", "5"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[:2] == ["run-x: 1 answers, 2 unsourced numbers, 2.00 per answer", "run-h: skipped (holdout)"]
    assert sorted(x.split("  | ")[0] for x in out[3:]) == ["run-x i1: 60–70 %", "run-x i1: 99.9 %"]
    assert "Purity 99.9 %. Students" in out[3] + out[4]


def test_rescore_reads_the_body_and_keeps_the_first_scores(tmp_path, capsys):
    prompt = "C1. The sum is 12.\nC2. The mass is 4 g.\nFinish with:\nA1: <verdict for C1>\nA2: <verdict for C2>\n"
    key = {"parts": [{"label": "A1", "verdict": ["incorrect"]}, {"label": "A2", "verdict": ["correct"]}]}
    items = tmp_path / "items.yaml"
    items.write_text(yaml.safe_dump({"items": [{"id": i, "set": "S3", "split": "dev", "prompt": prompt, "key": key}
                                               for i in ("b1-s3-01", "b1-s3-02")]}), encoding="utf-8")
    answer = "**C1:** Incorrect — 11.\n**C2:** Correct.\n\nA1: Correct\nA2: Incorrect"
    logs = [item_log("b1-s3-01", {"A1": False, "A2": False}, answer=answer,
                     messages=[{"role": "user", "content": prompt}]),
            item_log("b1-s3-02", {"A1": True}, answer=answer)]  # answered another prompt: left as it is
    folder = write_run(tmp_path / "run-r", "D", logs, items)
    assert rescore_main([str(folder), "--dry-run"]) == 0
    assert capsys.readouterr().out.startswith("run-r: parts 0 → 2 of 2 (line-only 0), 2 verdict parts read from"
                                              " the body, 1 items, 1 skipped")
    assert read_logs(folder)[0]["scores"]["final"]["parts"] == {"A1": False, "A2": False}  # nothing written
    assert rescore_main([str(folder)]) == 0 and rescore_main([str(folder)]) == 0
    r, kept = read_logs(folder)
    assert r["scores"]["final"]["parts"] == {"A1": True, "A2": True} and r["correct"]
    assert r["scores"]["final"]["sources"] == {"A1": "body", "A2": "body"}
    assert r["scores_before_rescore"]["final"]["parts"] == {"A1": False, "A2": False}  # the first scores, kept
    assert "scores_before_rescore" not in kept and kept["scores"]["final"]["parts"] == {"A1": True}
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))["all"]
    assert (summary["parts_right"], summary["line_parts_right"], summary["body_verdict_parts"]) == (3, 1, 2)
    assert "| run-r | all | D | 2 | 0 | 0 | 2/2 (100 %) | – | 3/3 (100 %) | 1/3 | 2/2 | 2 |" in report([folder])


@pytest.fixture
def two_runs(tmp_path):
    items = tmp_path / "items.yaml"
    items.write_text(yaml.safe_dump(ITEMS), encoding="utf-8")
    sent = {i["id"]: [*i.get("turns", []), {"role": "user", "content": i["prompt"]}] for i in ITEMS["items"]}
    return [write_run(tmp_path / f"20261023-1200-{m}", m,
                      [item_log(i, {"A1": True}, mode=m, messages=msgs) for i, msgs in sent.items()]
                      + [item_log("b1-s3-01x", {"A1": False}, mode=m, error="down")], items) for m in ("A", "B")]


def test_pack_is_blind_and_reproducible(two_runs, tmp_path):
    assert pack(two_runs, tmp_path / "pack", seed=7) == 4  # the failed item is left out
    text = (tmp_path / "pack" / "pack.md").read_text(encoding="utf-8")
    mapping = json.loads((tmp_path / "pack" / "mapping.json").read_text(encoding="utf-8"))
    assert sorted(mapping["answers"]) == ["R001", "R002", "R003", "R004"] and mapping["seed"] == 7
    assert {(a["run"], a["item"]) for a in mapping["answers"].values()} == \
           {(r.name, i) for r in two_runs for i in ("b1-s3-01", "b1-s6-01")}
    for leak in ("20261023", "mode", "secret-model", "SECRET-TRACE", "b1-s3-01", "S3"):
        assert leak not in text, leak
    assert "## R001" in text and "D1" in text and "**user:** Hi\n\n**assistant:** Hello\n\nAnd now?" in text
    assert "Finish with A1: <verdict>" in text
    pack(two_runs, tmp_path / "again", seed=7)
    assert (tmp_path / "again" / "pack.md").read_text(encoding="utf-8") == text


def review(blind, **over):
    return {"blind_id": blind, "score": 6, "defects_found": [], "defects_missed": [], "verdicts": {}, "new_errors": [],
            "omissions": [], "unsupported_additions": 0, "unsupported_as_verified": 0, "final_vs_analysis": 0,
            "must_not_violated": [], "reveals": "r", **over}


def test_merge_writes_reviews_and_metrics(two_runs, tmp_path):
    pack(two_runs, tmp_path / "pack", seed=1)
    answers = json.loads((tmp_path / "pack" / "mapping.json").read_text(encoding="utf-8"))["answers"]
    blocks = []
    for blind, a in answers.items():
        if a["item"] == "b1-s3-01":
            good = a["run"].endswith("B")
            blocks.append(review(blind, score=8 if good else 4, defects_found=["D1"] if good else [],
                                 defects_missed=[] if good else ["D1"], verdicts={"A": "incorrect" if good else "correct"},
                                 new_errors=[] if good else [{"tag": "F4", "what": "w"}]))
        else:
            blocks.append(review(blind, score=7, omissions=["E"], unsupported_additions=1))
    (tmp_path / "pack" / "reviews.yaml").write_text(yaml.safe_dump(blocks), encoding="utf-8")
    assert merge(tmp_path / "pack") == [r.resolve() for r in two_runs]
    reviewed = json.loads((two_runs[0] / "summary.json").read_text(encoding="utf-8"))
    assert write_summary(two_runs[0], "A") == reviewed  # a resumed run rebuilds its summary with the reviews
    a, b = (json.loads((r / "summary.json").read_text(encoding="utf-8"))["review"] for r in two_runs)
    assert (a["all"]["mean_score"], a["all"]["defects_found_pct"], a["all"]["verdict_pct"]) == (5.5, 0.0, 0.0)
    assert (b["all"]["mean_score"], b["all"]["defects_found_pct"], b["all"]["verdict_pct"]) == (7.5, 100.0, 100.0)
    assert a["all"]["new_errors"] == {"F4": 1} and (a["all"]["omissions"], a["all"]["unsupported_additions"]) == (1, 1)
    assert set(a["sets"]) == {"S3"} and a["sets"]["S3"]["reviewed"] == 2
    reviews = json.loads((two_runs[1] / "reviews.json").read_text(encoding="utf-8"))
    assert set(reviews) == {"b1-s3-01", "b1-s6-01"} and reviews["b1-s3-01"]["score"] == 8
    text = report(two_runs, two_runs[0])
    assert "| 20261023-1200-B | all | B | 2 | 7.5 | 100 % | 100 % | 1 | 1 | 0 | 0 | 0 | – |" in text
    assert text.rstrip().endswith("| +0 | +2 |")  # paired review score: (8 - 4 + 7 - 7) / 2


def test_merge_rejects_bad_reviews_clearly(two_runs, tmp_path):
    pack(two_runs, tmp_path / "pack", seed=1)
    answers = json.loads((tmp_path / "pack" / "mapping.json").read_text(encoding="utf-8"))["answers"]
    s3 = next(b for b, a in answers.items() if a["item"] == "b1-s3-01")
    s6 = next(b for b, a in answers.items() if a["item"] == "b1-s6-01" and b != s3)
    blocks = [review(s3, score=11, defects_found=["D9"], verdicts={"Z": "x"}), review(s6, unsupported_additions=-1),
              review(s6), review("R999"), review(s3, new_errors=[{"what": "no tag"}])]
    (tmp_path / "pack" / "reviews.yaml").write_text(yaml.safe_dump(blocks), encoding="utf-8")
    with pytest.raises(ValueError) as e:
        merge(tmp_path / "pack")
    message = str(e.value)
    for expected in (f"{s3}: score must be a number from 0 to 10, got 11", f"{s3}: defects not in the key: ['D9']",
                     f"{s3}: verdicts for items not in the key: ['Z']", f"{s6}: unsupported_additions must be a whole number",
                     f"{s6}: reviewed twice", "block 4: unknown or missing blind_id 'R999'", f"{s3}: each new_errors entry needs a tag",
                     "no review for"):
        assert expected in message, expected
    assert not (two_runs[0] / "reviews.json").exists()
    assert check_reviews({"blind_id": "R001"}, {}) == ["reviews.yaml must be a list of review blocks (- blind_id: R001 ...)"]


@pytest.mark.anyio
async def test_run_one_on_error_scores_nothing_right(cfg, fake):
    fake.scripts = [CONNECT_ERROR]
    item = {"id": "e1", "set": "S2", "question": "q", "lang": "en", "parts": [{"label": "A1", "verdict": ["yes"]}]}
    r = await run_one(Agent(cfg, fake.client(cfg.model_backend), no_store(cfg)), item, "C")
    assert r["error"] and r["correct"] is False and r["draft"] == "" and r["mode"] == "C"
    assert summarize([r])["missing_lines"] == 0  # a failed item is an error, not an omission
