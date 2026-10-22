"""Eval infrastructure (eval-runner spec): authored items, modes, item logs, resume, manifest, report and
blind review. Synthetic items only, the fake upstream, no GPU."""

import hashlib
import json

import pytest
import yaml
from pydantic import ValidationError

import evals.run as run
from conftest import CONNECT_ERROR, content
from evals.report import report
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
    assert r["scores"]["final"] == {"correct": False, "parts": {"A1": False, "A2": True, "A3": False},
                                    "missing": ["A3"], "verdicts": {"A2": True}}
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
    fake.scripts = [[content("80.1 °C")], CONNECT_ERROR]
    s = await run_eval(agent, items, folder)
    assert s["all"]["errors"] == 1 and "unreachable" in read_logs(folder)[1]["error"]
    s = await run_eval(agent, items, folder)  # resume: nothing left to do
    assert len(fake.requests) == 2 and s["all"]["errors"] == 1
    fake.scripts = [[content("80.2 °C")]]
    s = await run_eval(agent, items, folder, retry_errors=True)
    assert len(fake.requests) == 3 and s["all"]["errors"] == 0 and s["all"]["correct"] == 2
    assert read_logs(folder)[0]["answer"] == "80.1 °C"  # the done item was not run again


def test_resumed_run_must_match(tmp_path):
    man = {"mode": "A", "config_sha256": "c1", "git": {"sha": "g1", "dirty": False}, "date": "d1"}
    open_run(tmp_path / "r", man)
    open_run(tmp_path / "r", {**man, "date": "d2", "git": {"sha": "g1", "dirty": True}})
    assert json.loads((tmp_path / "r" / "manifest.json").read_text(encoding="utf-8"))["date"] == "d1"
    for change in ({"mode": "B"}, {"config_sha256": "c2"}, {"git": {"sha": "g2", "dirty": False}}):
        with pytest.raises(ValueError, match="new --run-id"):
            open_run(tmp_path / "r", {**man, **change})


def test_manifest(cfg, tmp_path, monkeypatch):
    model, store, cache = tmp_path / "m.gguf", tmp_path / "s.db", tmp_path / "evals" / ".sha-cache.json"
    model.write_bytes(b"weights")
    store.write_bytes(b"store")
    cfg.llama_server.model_path, cfg.retrieval.layer1_encyclopedia_db = model, store
    monkeypatch.setattr(run, "NVIDIA_SMI", ["no-such-nvidia-smi-here"])
    monkeypatch.setenv("MODEL_PROFILE", "p1")
    man = manifest(cfg, "r1", "A", [QUESTIONS], ["--mode", "A"], cache)
    assert man["model_file"]["sha256"] == hashlib.sha256(b"weights").hexdigest() and man["gpu"] is None
    assert (man["run_id"], man["mode"], man["model_profile"], man["args"], man["seed"]) == ("r1", "A", "p1", ["--mode", "A"], None)
    assert man["store"]["path"] == str(store) and man["store"]["size"] == 5 and man["store"]["mtime"]
    assert man["eval_files"] == [{"path": "evals/questions.yaml", "sha256": file_sha256(QUESTIONS)}]
    assert man["context_tokens"] == cfg.context_tokens and man["sampling"]["temperature"] == cfg.model_backend.temperature
    assert set(man["sampling"]) >= {"max_tokens", *cfg.model_backend.extra_body} and "git" in man and man["date"]
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
    assert "| run-a | S3 | A | 2 | 0 | 0 | 0/2 (0 %) | 1/4 (25 %) |" in text
    assert "| run-b | all | B | 3 |" in text and "mean score" not in text
    assert "| run-b | S3 | 2 | 0 → 1 | +1 | 1 / 0 | 1 → 0 | +2 | – |" in text  # i3 is not in the base: not paired


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
