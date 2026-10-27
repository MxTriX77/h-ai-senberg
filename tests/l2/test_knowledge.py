"""The knowledge switch: base (Layer 1 only) vs extended (+ literature passages in search_knowledge) vs hybrid
(the literature passages ranked by BM25 and embeddings together)."""

import json
import shutil

import httpx
import pytest

import evals.run as run
import haisenberg.tools as tools
from haisenberg.config import ConfigError
from haisenberg.l1.store import L1Store
from haisenberg.l2.dense import HybridSearch
from haisenberg.l2.store import L2Store
from haisenberg.llm import UpstreamError
from haisenberg.tools import SEARCH, Knowledge, Sources

from .test_dense import DIM, MODEL, fake_embed
from .test_dense import build as build_dense


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
    unlabelled = text_b.replace(f"; {cfg.retrieval.source_labels.reference})\n", ")\n")
    assert ids_a and unlabelled.startswith(text_a)  # the encyclopedia records first, as in base but labelled
    assert ids_b[: len(ids_a)] == ids_a
    added = [sources.get(i) for i in ids_b[len(ids_a):]]
    assert 0 < len(added) <= cfg.retrieval.literature_k
    assert all(s.kind == "text" and s.record.label().endswith(")") for s in added)  # '... (Wikipedia Physics)'
    assert all(f"[{s.id}] {s.record.label()}" in text_b for s in added)


def test_same_tools_in_both_without_labels(cfg, built_store, l2_store):
    cfg.retrieval.label_sources = False
    assert Knowledge(L1Store(built_store), cfg).tools == Knowledge(L1Store(built_store), cfg, L2Store(l2_store)).tools


def test_extended_labels_reference_and_literature(base, extended, cfg):
    labels = cfg.retrieval.source_labels
    text, ids, sources = search(extended, "electron")
    kinds = {sources.get(i).literature for i in ids}
    assert kinds == {False, True}  # fixture: encyclopedia records and literature passages
    for i in ids:
        s = sources.get(i)
        label = labels.literature if s.literature else labels.reference
        assert f"[{s.id}] {s.record.label()} ({s.kind}; {', '.join(s.record.domains)}; {label})\n" in text
    note = cfg.tools[SEARCH].literature
    for a, b in zip(base.tools, extended.tools):
        name, desc = a["function"]["name"], a["function"]["description"]
        assert b["function"]["description"] == (f"{desc} {note}" if name == SEARCH else desc)
        assert {**b["function"], "description": desc} == a["function"]


@pytest.mark.parametrize("query", ["electron", "ethanol"])
def test_base_unchanged_by_the_label_switch(cfg, built_store, query):
    runs = []
    for on in (True, False):
        cfg.retrieval.label_sources = on
        k = Knowledge(L1Store(built_store), cfg)
        sources = Sources()
        runs.append((k.tools, k.run(SEARCH, json.dumps({"query": query}), sources),
                     k.run("lookup_compound", json.dumps({"query": "caffeine"}), sources)))
    assert runs[0] == runs[1]
    text, sources = runs[0][1][0], Sources()
    Knowledge(L1Store(built_store), cfg).run(SEARCH, json.dumps({"query": query}), sources)
    assert sources.items and not any(s.literature for s in sources.items)
    assert text == "\n\n".join(f"[{s.id}] {s.record.label()} ({s.kind}; {', '.join(s.record.domains)})\n{s.passage}"
                               for s in sources.items)  # the format before labels


def test_extended_without_labels_when_switched_off(cfg, built_store, l2_store):
    cfg.retrieval.label_sources = False
    text, ids, sources = search(Knowledge(L1Store(built_store), cfg, L2Store(l2_store)), "electron")
    assert ids and all(f"[{i}] {sources.get(i).record.label()} ({sources.get(i).kind}; "
                       f"{', '.join(sources.get(i).record.domains)})\n" in text for i in ids)


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


# --- hybrid ---

class FakeClient:
    """Stands in for EmbeddingClient in Knowledge.open: the fake embedder, or an unreachable backend."""

    fail = False

    def __init__(self, cfg):
        pass

    def embed(self, texts):
        if FakeClient.fail:
            raise UpstreamError("embeddings backend unreachable at http://127.0.0.1:8092/v1")
        return fake_embed(texts)


@pytest.fixture
def hybrid_cfg(cfg, built_store, l2_store, tmp_path, monkeypatch):
    """cfg with every store on the fixtures, KNOWLEDGE=hybrid, the dense index built by the fake embedder and no
    embedding server (FakeClient answers)."""
    r = cfg.retrieval
    r.stores["encyclopedia"].path, r.stores["literature"].path = built_store, l2_store
    r.stores["literature_dense"].path = tmp_path / "literature_dense"
    build_dense(l2_store, r.stores["literature_dense"].path, model={**MODEL, "model_id": cfg.embedding.model_id})
    r.knowledge = "hybrid"
    cfg.embedding.server.enabled = False
    FakeClient.fail = False
    monkeypatch.setattr(tools, "EmbeddingClient", FakeClient)
    return cfg


def test_hybrid_results_look_like_extended(hybrid_cfg, built_store, l2_store):
    cfg = hybrid_cfg
    labels = cfg.retrieval.source_labels
    ext_k = Knowledge(L1Store(built_store), cfg, L2Store(l2_store))
    hyb_k = Knowledge.open(cfg)
    assert isinstance(hyb_k.literature, HybridSearch) and hyb_k.tools == ext_k.tools  # same tools and texts (#46 note too)
    text_e, ids_e, src_e = search(ext_k, "electron")
    text_h, ids_h, src_h = search(hyb_k, "electron")
    layer1_e = [src_e.get(i) for i in ids_e if not src_e.get(i).literature]
    layer1_h = [src_h.get(i) for i in ids_h if not src_h.get(i).literature]
    assert layer1_e and [(s.id, s.record.label(), s.passage) for s in layer1_h] == \
        [(s.id, s.record.label(), s.passage) for s in layer1_e]  # the same Layer 1 records, first
    assert ids_h[: len(layer1_h)] == [s.id for s in layer1_h]
    added = [src_h.get(i) for i in ids_h[len(layer1_h):]]
    assert 0 < len(added) <= cfg.retrieval.literature_k and all(s.literature for s in added)
    for s in added:  # labelled as literature, as in extended
        assert f"[{s.id}] {s.record.label()} ({s.kind}; {', '.join(s.record.domains)}; {labels.literature})\n" in text_h


def test_hybrid_finds_a_paraphrase_extended_misses(hybrid_cfg, built_store, l2_store):
    ext_k = Knowledge(L1Store(built_store), hybrid_cfg, L2Store(l2_store))
    assert search(ext_k, "guaranine")[1] == []
    _, ids, sources = search(Knowledge.open(hybrid_cfg), "guaranine")
    assert ids and {sources.get(i).record.title.lower() for i in ids} == {"caffeine"}


def test_embedding_error_in_a_search_is_an_error_line(hybrid_cfg):
    k = Knowledge.open(hybrid_cfg)
    FakeClient.fail = True
    text, lines, ids = k.run(SEARCH, json.dumps({"query": "electron"}), Sources())
    assert text.startswith("Error: lookup failed") and "unreachable" in text and ids == []


def test_open_starts_the_embedding_server(hybrid_cfg, monkeypatch):
    started = []

    class StubSupervisor:
        def __init__(self, view):
            self.view = view

        def start(self):
            started.append(self.view)

    monkeypatch.setattr(tools, "Supervisor", StubSupervisor)
    hybrid_cfg.embedding.server.enabled = True
    Knowledge.open(hybrid_cfg)
    (view,) = started
    assert view.llama_server is hybrid_cfg.embedding.server and view.model_backend.model_id == hybrid_cfg.embedding.model_id


def test_hybrid_open_failures(hybrid_cfg, l2_store, tmp_path, monkeypatch):
    dense = hybrid_cfg.retrieval.stores["literature_dense"]
    FakeClient.fail = True  # the server does not answer
    with pytest.raises(ConfigError, match="KNOWLEDGE=hybrid.*embedding server.*unreachable"):
        Knowledge.open(hybrid_cfg)
    FakeClient.fail = False
    monkeypatch.setattr(FakeClient, "embed", lambda self, texts: fake_embed(texts)[:, : DIM // 2])
    with pytest.raises(ConfigError, match="dimensions"):
        Knowledge.open(hybrid_cfg)
    stale = tmp_path / "stale"
    shutil.copytree(dense.path, stale)
    meta = json.loads((stale / "meta.json").read_text(encoding="utf-8"))
    (stale / "meta.json").write_text(json.dumps({**meta, "store_built_at": "2000-01-01T00:00:00Z"}), encoding="utf-8")
    dense.path = stale
    with pytest.raises(ConfigError, match="KNOWLEDGE=hybrid.*ingest.embed_l2 --restart"):
        Knowledge.open(hybrid_cfg)
    dense.path = tmp_path / "missing"
    with pytest.raises(ConfigError, match="KNOWLEDGE=hybrid.*missing.*ingest.embed_l2"):
        Knowledge.open(hybrid_cfg)


def test_base_and_extended_never_open_the_dense_store(hybrid_cfg, monkeypatch):
    monkeypatch.setattr(tools, "DenseIndex", None)  # would fail if used
    for knowledge in ("base", "extended"):
        hybrid_cfg.retrieval.knowledge = knowledge
        k = Knowledge.open(hybrid_cfg)
        assert k.literature is None or isinstance(k.literature, L2Store)


def test_eval_run_with_bad_hybrid_makes_no_run_folder(l2_store, tmp_path, monkeypatch, capsys):
    data = tmp_path / "data"
    data.mkdir()
    shutil.copy(l2_store, data / "literature_fts.db")  # the literature store is there, its dense index is not
    monkeypatch.setenv("DATA_DIR", str(data))
    monkeypatch.setenv("KNOWLEDGE", "hybrid")
    assert run.main(["--set", "facts", "--limit", "1", "--run-id", "h-run"]) == 1
    err = capsys.readouterr().err
    assert "KNOWLEDGE=hybrid" in err and "ingest.embed_l2" in err
    assert not (data / "evals" / "h-run").exists()


@pytest.mark.parametrize("floor, kept", [(0.1, True), (0.99, False)])
def test_similar_dense_hits_pass_the_term_share_filter(hybrid_cfg, floor, kept):
    hybrid_cfg.retrieval.min_term_share = 0.5  # as shipped: a paraphrase shares none of the query's words
    hybrid_cfg.retrieval.dense.min_similarity = floor
    text, lines, ids = Knowledge.open(hybrid_cfg).run(SEARCH, json.dumps({"query": "guaranine"}), Sources())
    if kept:
        assert ids and "caffeine" in text.lower()
    else:
        assert ids == [] and text.startswith("Nothing relevant found") and any("weak match" in x for x in lines)
