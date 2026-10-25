"""The knowledge switch: base (Layer 1 only) vs extended (+ literature passages in search_knowledge)."""

import json

import httpx
import pytest

import evals.run as run
from haisenberg.config import ConfigError
from haisenberg.l1.store import L1Store
from haisenberg.l2.store import L2Store
from haisenberg.tools import SEARCH, Knowledge, Sources


def search(k: Knowledge, query: str, sources: Sources | None = None, **args):
    sources = sources or Sources()
    text, lines, ids = k.run(SEARCH, json.dumps({"query": query, **args}), sources)
    return text, ids, sources


@pytest.fixture
def base(cfg, built_store):
    return Knowledge(L1Store(built_store), cfg)


@pytest.fixture
def extended(cfg, built_store, l2_store):
    return Knowledge(L1Store(built_store), cfg, L2Store(l2_store))


def test_extended_appends_literature_after_layer1(base, extended, cfg):
    text_a, ids_a, _ = search(base, "electron")
    text_b, ids_b, sources = search(extended, "electron")
    assert ids_a and text_b.startswith(text_a)  # the encyclopedia records first, exactly as in base
    assert ids_b[: len(ids_a)] == ids_a
    added = [sources.get(i) for i in ids_b[len(ids_a):]]
    assert 0 < len(added) <= cfg.retrieval.literature_k
    assert all(s.kind == "text" and s.record.label().endswith(")") for s in added)  # '... (Wikipedia Physics)'
    assert all(f"[{s.id}] {s.record.label()}" in text_b for s in added)


def test_same_tools_in_both(base, extended):
    assert base.tools == extended.tools


def test_literature_only_hit(base, extended):
    assert search(base, "trimethylxanthine")[1] == []
    text, ids, sources = search(extended, "trimethylxanthine")
    assert ids and sources.get(ids[0]).record.title == "caffeine"


def test_domain_filters_literature(extended):
    _, ids, sources = search(extended, "electron", domain="chemistry")
    assert ids and all("chemistry" in sources.get(i).record.domains for i in ids)


def test_two_passages_of_one_page_are_two_sources(cfg, built_store, l2_store):
    cfg.retrieval.literature_k = 10
    k = Knowledge(L1Store(built_store), cfg, L2Store(l2_store))
    _, ids, sources = search(k, "photoelectric")
    pages = [sources.get(i).record for i in ids if sources.get(i).record.citation.locator.get("passage")]
    same_page = [r for r in pages if r.path == "Photoelectric_effect"]
    assert len(same_page) >= 2 and len({sources.get(i).id for i in ids}) == len(ids)


def test_open_follows_the_setting(cfg, built_store, l2_store):
    cfg.retrieval.stores["encyclopedia"].path, cfg.retrieval.stores["literature"].path = built_store, l2_store
    assert Knowledge.open(cfg).literature is None  # base ignores an existing literature store
    cfg.retrieval.knowledge = "extended"
    assert Knowledge.open(cfg).literature is not None


@pytest.mark.parametrize("wrong", ["missing", "layer1"])
def test_extended_without_literature_store_fails(cfg, built_store, tmp_path, wrong):
    cfg.retrieval.knowledge = "extended"
    cfg.retrieval.stores["literature"].path = tmp_path / "none.db" if wrong == "missing" else built_store
    with pytest.raises(ConfigError, match="KNOWLEDGE=extended.*ingest.build_l2"):
        Knowledge.open(cfg)


def no_backend(url, timeout):
    raise httpx.ConnectError("refused")


def test_manifest_tells_a_and_b_apart(cfg, l2_store, tmp_path, monkeypatch):
    monkeypatch.setattr(run, "NVIDIA_SMI", ["no-such-nvidia-smi-here"])
    monkeypatch.setattr(httpx, "get", no_backend)
    cfg.retrieval.stores["literature"].path = l2_store
    a = run.manifest(cfg, "a", "D", [], [], tmp_path / "cache.json")
    cfg.retrieval.knowledge = "extended"
    b = run.manifest(cfg, "b", "D", [], [], tmp_path / "cache.json")
    assert (a["knowledge"], b["knowledge"]) == ("base", "extended")
    lit_a, lit_b = a["stores"]["literature"], b["stores"]["literature"]
    assert (lit_a["kind"], lit_a["enabled"], lit_b["enabled"]) == ("literature", False, True)
    assert lit_b["path"] == str(l2_store) and lit_b["size"] == l2_store.stat().st_size and lit_b["mtime"]
    assert a["stores"]["encyclopedia"] == b["stores"]["encyclopedia"] and a["stores"]["encyclopedia"]["enabled"]
    assert a["config_sha256"] != b["config_sha256"]
    assert (a["git"], a["mode"], a["model_id"], a["sampling"]) == (b["git"], b["mode"], b["model_id"], b["sampling"])


def test_eval_run_with_bad_b_makes_no_run_folder(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("KNOWLEDGE", "extended")
    assert run.main(["--set", "facts", "--limit", "1", "--run-id", "b-run"]) == 1
    err = capsys.readouterr().err
    assert "KNOWLEDGE=extended" in err and "ingest.build_l2" in err
    assert not (tmp_path / "data" / "evals" / "b-run").exists()
