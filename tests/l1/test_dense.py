"""The dense encyclopedia index (ingest.embed_l1, haisenberg.l1.dense) and KNOWLEDGE=crosslingual, with a fake
embedder that understands a few Russian words: no GPU, no network."""

import dataclasses
import json
import shutil
import sqlite3
import zlib

import numpy as np
import pytest

import evals.run as run
import haisenberg.tools as tools
import ingest.embed_l2 as embed_l2
from haisenberg.config import ConfigError
from haisenberg.l1.dense import (REBUILD_COMMAND, TEXT_CHARS, DenseArticleRecord, DenseEncyclopedia, article_text,
                                 non_latin)
from haisenberg.l1.store import ArticleRecord, L1Store
from haisenberg.l2.dense import META, VECTORS, DenseIndex
from haisenberg.l2.store import content_words
from haisenberg.llm import UpstreamError
from haisenberg.tools import SEARCH, Knowledge, Sources
from ingest import embed_l1
from ingest.l1base import BuildError

DIM = 256
RU = {"бензол": "benzene", "этанол": "ethanol", "электрон": "electron", "железо": "iron", "спирт": "ethanol"}
MODEL = {"model_id": "fake-embedder", "query_prefix": "", "model_sha256": None}


def fake_embed(texts: list[str]) -> np.ndarray:
    """Bag of words hashed into DIM buckets, L2-normalised, the Russian words of RU read as their English ones."""
    out = np.zeros((len(texts), DIM), np.float32)
    for i, text in enumerate(texts):
        for w in content_words(text) or ["empty"]:
            out[i, zlib.crc32(RU.get(w, w).encode()) % DIM] += 1
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def no_embed(texts):
    raise AssertionError("no embedding may be requested")


def build(db, out, embed=fake_embed, model=MODEL, restart=False, batch=2, workers=1):
    return embed_l1.build(db, out, embed, model, batch, workers, restart, log=lambda line: None)


def texts_of(store: L1Store) -> list[str]:
    return [article_text(*r) for r in store.con.execute("SELECT title, lead, infobox FROM article ORDER BY id")]


class StubIndex:
    """A dense ranking given by the test: (article id, similarity) pairs, or ids with similarities falling from 0.9."""

    def __init__(self, ranking):
        self.ranking = [r if isinstance(r, tuple) else (r, 0.9 - i / 100) for i, r in enumerate(ranking)]

    def search(self, vector, rerank):
        return self.ranking[:rerank]


@pytest.fixture(scope="module")
def dense_dir(built_store, tmp_path_factory):
    out = tmp_path_factory.mktemp("dense") / "encyclopedia_dense"
    build(built_store, out)
    return out


@pytest.fixture
def store(built_store):
    return L1Store(built_store)


@pytest.fixture
def dense_cfg(cfg):
    return cfg.retrieval.dense.model_copy(update={"rerank": 12, "candidates": 50})


def encyclopedia(store, index, cfg, embed=fake_embed):
    return DenseEncyclopedia(store, index, embed, "", cfg)


def nearest(store, query):
    """Title of the article whose fake vector is nearest to the query's."""
    sims = fake_embed(texts_of(store)) @ fake_embed([query])[0]
    return store.articles([int(np.argmax(sims)) + 1])[0].title


# --- the gate ---

@pytest.mark.parametrize("query", ["теплота", "диметиловый эфир температура кипения", "水素",
                                   "Молярная масса CH₃COOH — 60,05 г/моль.", "Метил orange", "S1 — Верно."])
def test_non_latin_words(query):
    assert non_latin(query)


@pytest.mark.parametrize("query", ["ΔH of formation of H2O", "λ", "pH", "C₂H₆O", "S1", "5 μm", "café crème",
                                   "Schrödinger equation", "", "42"])
def test_latin_words_only(query):
    assert not non_latin(query)


def test_article_text():
    assert article_text("Benzene", "Benzene is a ring.", "Formula: C6H6") == "Benzene\nBenzene is a ring."
    assert article_text("Iron", "", "Symbol: Fe") == "Iron\nSymbol: Fe"
    assert len(article_text("T", "x" * 5000, "")) == TEXT_CHARS


# --- the Layer 1 store ---

def test_store_identity_and_records_by_id(store):
    meta = dict(store.con.execute("SELECT key, value FROM meta"))
    assert store.identity() == {"store_built_at": meta["built_at"], "articles": int(meta["articles"])}
    ranked = store.ranked_articles("benzene", 5)
    assert [r for _, r in ranked] == store.search_encyclopedia("benzene", 5)
    by_id = store.articles([aid for aid, _ in ranked][::-1])
    assert [r.title for r in by_id] == [r.title for _, r in ranked][::-1]
    fields = [f.name for f in dataclasses.fields(ArticleRecord) if f.name not in ("score", "snippet")]
    for record, (_, keyword) in zip(by_id[::-1], ranked):
        assert {f: getattr(record, f) for f in fields} == {f: getattr(keyword, f) for f in fields}
    assert store.articles([]) == [] and store.in_domain([], "physics") == []


def test_in_domain(store):
    ids = [aid for aid, _ in store.ranked_articles("electron constant iron benzene", 10)]
    physics = store.in_domain(ids, "physics")
    assert physics and set(physics) < set(ids)
    assert all("physics" in r.domains for r in store.articles(physics))


# --- build ---

def test_build_writes_one_vector_per_article(dense_dir, store):
    meta = json.loads((dense_dir / META).read_text(encoding="utf-8"))
    n = store.identity()["articles"]
    assert meta["complete"] and (meta["dim"], meta["done_rows"], meta["shortened"]) == (DIM, n, 0)
    assert {k: meta[k] for k in ("store_built_at", "articles")} == store.identity()
    assert (meta["model_id"], meta["index"]) == ("fake-embedder", "Flat")
    vectors = np.load(dense_dir / VECTORS)
    assert vectors.dtype == np.float16 and vectors.shape == (n, DIM)
    assert np.allclose(vectors, fake_embed(texts_of(store)), atol=1e-3)  # row = article id - 1


def test_resume_after_a_stop_embeds_only_the_rest(built_store, dense_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(embed_l2, "SAVE_S", 0)  # save after every window (2 x 1 x 4 = 8 articles)
    calls = []

    def stops(texts):
        calls.append(texts)
        if len(calls) == 5:  # the first request of the second window
            raise KeyboardInterrupt
        return fake_embed(texts)

    out = tmp_path / "dense"
    with pytest.raises(KeyboardInterrupt):
        build(built_store, out, stops)
    assert json.loads((out / META).read_text(encoding="utf-8"))["done_rows"] == 8
    seen = []
    build(built_store, out, lambda texts: seen.extend(texts) or fake_embed(texts))
    assert seen == texts_of(L1Store(built_store))[8:]
    assert np.array_equal(np.load(out / VECTORS), np.load(dense_dir / VECTORS))


def test_other_layer1_build_needs_restart(built_store, tmp_path):
    out = tmp_path / "dense"
    build(built_store, out)
    meta = json.loads((out / META).read_text(encoding="utf-8"))
    (out / META).write_text(json.dumps({**meta, "store_built_at": "2000-01-01T00:00:00Z"}), encoding="utf-8")
    before = (out / META).read_bytes()
    with pytest.raises(BuildError, match="--restart"):
        build(built_store, out)
    assert (out / META).read_bytes() == before
    assert build(built_store, out, restart=True)["complete"]


def test_gap_in_article_ids_is_refused(built_store, tmp_path):
    db = tmp_path / "gap.db"
    shutil.copy(built_store, db)
    with sqlite3.connect(db) as con:
        con.execute("DELETE FROM article WHERE id = 5")
    with pytest.raises(BuildError, match="without gaps"):
        build(db, tmp_path / "dense")


# --- search ---

def test_russian_query_finds_the_english_article(store, dense_dir, dense_cfg):
    index = DenseIndex(dense_dir, MODEL["model_id"], store, 1, "encyclopedia", REBUILD_COMMAND)
    got = encyclopedia(store, index, dense_cfg).search_encyclopedia("бензол", 3)
    assert store.search_encyclopedia("бензол", 3) == []
    assert got and got[0].title == nearest(store, "бензол") and "enzene" in got[0].title
    assert isinstance(got[0], DenseArticleRecord) and 0 < got[0].similarity <= 1
    assert got[0].score == round(1 - got[0].similarity, 4)  # found by meaning only: never a title match's 0


def test_english_query_is_the_keyword_search(store, dense_cfg):
    dense = encyclopedia(store, StubIndex([3, 4]), dense_cfg, embed=no_embed)
    for query in ("benzene", "ΔH electron", "C6H6", "the of"):
        assert dense.search_encyclopedia(query, 5) == store.search_encyclopedia(query, 5)


def test_mixed_query_keeps_its_alias_match(store, dense_cfg):
    keyword = store.search_encyclopedia("C6H6 растворитель", 5)
    assert keyword[0].title == "Benzene" and keyword[0].score == 0  # 'c6h6' is an alias of Benzene
    got = encyclopedia(store, StubIndex([8, 9]), dense_cfg).search_encyclopedia("C6H6 растворитель", 5)
    assert got[0] == keyword[0] and type(got[0]) is ArticleRecord  # 1 / 61 each: the lower id first
    both = encyclopedia(store, StubIndex([7]), dense_cfg).search_encyclopedia("C6H6 растворитель", 5)
    assert both[0].title == "Benzene" and both[0].score == 0 and both[0].similarity == 0.9


def test_agreement_wins_and_ties_go_to_the_lower_id(store, dense_cfg, monkeypatch):
    keyword = store.ranked_articles("electron iron", 2)
    assert len(keyword) == 2
    (first, _), (second, _) = keyword
    other = next(i for i in (10, 9, 8) if i not in (first, second))
    monkeypatch.setattr(store, "ranked_articles", lambda query, k, domain=None: keyword)
    got = encyclopedia(store, StubIndex([other, second]), dense_cfg).search_encyclopedia("электрон", 3)
    titles = {aid: r.title for aid, r in zip((first, second, other), store.articles([first, second, other]))}
    assert got[0].title == titles[second]  # second in both rankings beats first in one
    assert [r.title for r in got[1:]] == [titles[i] for i in sorted([first, other])]  # 1 / 61 each
    assert isinstance(got[0], DenseArticleRecord) and got[0].similarity == 0.89
    assert type(next(r for r in got if r.title == titles[first])) is ArticleRecord


def test_domain_filter(store, dense_cfg):
    dense = encyclopedia(store, StubIndex(list(range(1, 11))), dense_cfg)
    got = dense.search_encyclopedia("электрон", 5, "physics")
    assert got and all("physics" in r.domains for r in got)


def test_query_prefix_is_sent(store, dense_cfg):
    sent = []
    dense = DenseEncyclopedia(store, StubIndex([1]), lambda texts: sent.extend(texts) or fake_embed(texts), "Q: ",
                              dense_cfg)
    dense.search_encyclopedia("этанол", 1)
    assert sent == ["Q: этанол"]


# --- KNOWLEDGE=crosslingual ---

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
def cross_cfg(cfg, built_store, tmp_path, monkeypatch):
    """cfg with the fixture Layer 1 store and its dense index (fake embedder), KNOWLEDGE=crosslingual, no server."""
    r = cfg.retrieval
    r.stores["encyclopedia"].path = built_store
    r.stores["encyclopedia_dense"].path = tmp_path / "encyclopedia_dense"
    build(built_store, r.stores["encyclopedia_dense"].path, model={**MODEL, "model_id": cfg.embedding.model_id})
    r.knowledge = "crosslingual"
    cfg.embedding.server.enabled = False
    FakeClient.fail = False
    monkeypatch.setattr(tools, "EmbeddingClient", FakeClient)
    return cfg


def search(k: Knowledge, query: str):
    sources = Sources()
    text, _, ids = k.run(SEARCH, json.dumps({"query": query}), sources)
    return text, [sources.get(i).record.title for i in ids]


def test_crosslingual_finds_what_base_misses(cross_cfg, built_store):
    base = Knowledge(L1Store(built_store), cross_cfg)
    cross = Knowledge.open(cross_cfg)
    assert isinstance(cross.encyclopedia, DenseEncyclopedia) and cross.tools == base.tools
    assert search(base, "этанол")[1] == []
    text, titles = search(cross, "этанол")
    assert titles and titles[0] == "Ethanol" and text.startswith("[S1] Ethanol (text; ")
    labels = cross_cfg.retrieval.source_labels  # no literature store: unlabelled, as with base (#46)
    assert cross.labels is None and labels.reference not in text and labels.literature not in text
    sources = Sources()
    cross.run(SEARCH, json.dumps({"query": "этанол"}), sources)
    assert isinstance(sources.items[0].record, DenseArticleRecord) and not sources.items[0].literature


def test_crosslingual_english_query_is_base(cross_cfg, built_store):
    base, cross = Knowledge(L1Store(built_store), cross_cfg), Knowledge.open(cross_cfg)
    FakeClient.fail = True  # an English query must not need the embedding server
    for query in ("ethanol boiling point", "Planck constant", "ΔH formation"):
        assert search(cross, query) == search(base, query)


def test_embedding_error_in_a_search_is_an_error_line(cross_cfg):
    k = Knowledge.open(cross_cfg)
    FakeClient.fail = True
    text, lines, ids = k.run(SEARCH, json.dumps({"query": "этанол"}), Sources())
    assert text.startswith("Error: lookup failed") and "unreachable" in text and ids == []


@pytest.mark.parametrize("floor, kept", [(0.5, True), (0.95, False)])
def test_similar_dense_articles_pass_the_term_share_filter(cfg, built_store, floor, kept):
    cfg.retrieval.min_term_share = 0.5  # as shipped: a Russian query shares no word with an English article
    cfg.retrieval.dense.min_similarity = floor
    store = L1Store(built_store)
    dense = DenseEncyclopedia(store, StubIndex([(1, 0.6)]), fake_embed, "", cfg.retrieval.dense)
    k = Knowledge(store, cfg, None, dense)
    text, lines, ids = k.run(SEARCH, json.dumps({"query": "этиловый спирт"}), Sources())
    if kept:
        assert ids == ["S1"] and text.startswith("[S1] Ethanol")
    else:
        assert ids == [] and text.startswith("Nothing relevant found") and "weak match, not used: Ethanol" in lines


def test_crosslingual_open_failures(cross_cfg, tmp_path, monkeypatch):
    dense = cross_cfg.retrieval.stores["encyclopedia_dense"]
    FakeClient.fail = True  # the server does not answer
    with pytest.raises(ConfigError, match="KNOWLEDGE=crosslingual.*encyclopedia index.*embedding server.*unreachable"):
        Knowledge.open(cross_cfg)
    FakeClient.fail = False
    monkeypatch.setattr(FakeClient, "embed", lambda self, texts: fake_embed(texts)[:, : DIM // 2])
    with pytest.raises(ConfigError, match="dimensions"):
        Knowledge.open(cross_cfg)
    monkeypatch.undo()
    monkeypatch.setattr(tools, "EmbeddingClient", FakeClient)
    for key, value in (("store_built_at", "2000-01-01T00:00:00Z"), ("model_id", "another")):
        stale = tmp_path / f"stale-{key}"
        shutil.copytree(dense.path, stale)
        meta = json.loads((stale / META).read_text(encoding="utf-8"))
        (stale / META).write_text(json.dumps({**meta, key: value}), encoding="utf-8")
        dense.path = stale
        with pytest.raises(ConfigError, match="KNOWLEDGE=crosslingual.*ingest.embed_l1 --restart"):
            Knowledge.open(cross_cfg)
    dense.path = tmp_path / "missing"
    with pytest.raises(ConfigError, match="KNOWLEDGE=crosslingual.*missing.*ingest.embed_l1"):
        Knowledge.open(cross_cfg)


def test_other_sets_never_open_the_dense_encyclopedia(cross_cfg, monkeypatch):
    monkeypatch.setattr(tools, "_dense_encyclopedia", None)  # would fail if used
    for knowledge in ("base", "extended", "hybrid"):
        cross_cfg.retrieval.knowledge = knowledge
        assert cross_cfg.retrieval.enabled("encyclopedia_dense") is None
    cross_cfg.retrieval.knowledge = "base"
    k = Knowledge.open(cross_cfg)
    assert k.encyclopedia is k.store


def test_eval_run_with_bad_crosslingual_makes_no_run_folder(built_store, tmp_path, monkeypatch, capsys):
    data = tmp_path / "data"
    data.mkdir()
    shutil.copy(built_store, data / "encyclopedia_fts.db")  # the Layer 1 store is there, its dense index is not
    monkeypatch.setenv("DATA_DIR", str(data))
    monkeypatch.setenv("KNOWLEDGE", "crosslingual")
    assert run.main(["--set", "facts", "--limit", "1", "--run-id", "x-run"]) == 1
    err = capsys.readouterr().err
    assert "KNOWLEDGE=crosslingual" in err and "ingest.embed_l1" in err
    assert not (data / "evals" / "x-run").exists()
