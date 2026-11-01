"""The dense literature index (ingest.embed_l2, haisenberg.l2.dense) with a fake embedder: no GPU, no network."""

import dataclasses
import json
import shutil
import sqlite3
import zlib

import numpy as np
import pytest

import ingest.embed_l2 as embed_l2
from haisenberg.config import DenseConfig
from haisenberg.l2.dense import INDEX, META, VECTORS, DenseIndex, DensePassageRecord, HybridSearch, passage_text
from haisenberg.l2.store import L2Error, L2Store, PassageRecord, content_words
from haisenberg.llm import UpstreamError
from ingest.l1base import BuildError

DIM = 256
SAME = {"guaranine": "caffeine"}  # the fake embedder's one paraphrase: another name of caffeine
MODEL = {"model_id": "fake-embedder", "query_prefix": "", "model_sha256": None}
DENSE = DenseConfig(index="Flat", nprobe=1, rerank=12, candidates=50, rrf_k=60, min_similarity=0.5)


def fake_embed(texts: list[str]) -> np.ndarray:
    """Bag of words hashed into DIM buckets, L2-normalised: texts sharing words are near."""
    out = np.zeros((len(texts), DIM), np.float32)
    for i, text in enumerate(texts):
        for w in content_words(text) or ["empty"]:
            out[i, zlib.crc32(SAME.get(w, w).encode()) % DIM] += 1
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def build(db, out, embed=fake_embed, model=MODEL, factory="Flat", restart=False, batch=2, workers=1):
    return embed_l2.build(db, out, embed, model, factory, batch, workers, restart, log=lambda line: None)


def texts_of(store: L2Store) -> list[str]:
    return [passage_text(*r) for r in store.con.execute("SELECT title, heading, text FROM passage ORDER BY id")]


@pytest.fixture(scope="module")
def dense_dir(l2_store, tmp_path_factory):
    out = tmp_path_factory.mktemp("dense") / "literature_dense"
    build(l2_store, out)
    return out


@pytest.fixture
def store(l2_store):
    return L2Store(l2_store)


@pytest.fixture
def hybrid(store, dense_dir):
    return HybridSearch(store, DenseIndex(dense_dir, MODEL["model_id"], store, DENSE.nprobe), fake_embed, "", DENSE)


class StubIndex:
    """A dense ranking given by the test (similarities falling from 0.9)."""

    def __init__(self, ids):
        self.ids = ids

    def search(self, vector, rerank):
        return [(pid, 0.9 - i / 100) for i, pid in enumerate(self.ids[:rerank])]


# --- build ---

def test_build_writes_one_vector_per_passage(dense_dir, store):
    meta = json.loads((dense_dir / META).read_text(encoding="utf-8"))
    assert meta["complete"] and (meta["dim"], meta["done_rows"], meta["shortened"]) == (DIM, 12, 0)
    assert {k: meta[k] for k in ("store_built_at", "passages")} == store.identity()
    assert (meta["model_id"], meta["index"]) == ("fake-embedder", "Flat") and (dense_dir / INDEX).is_file()
    vectors = np.load(dense_dir / VECTORS)
    assert vectors.dtype == np.float16 and vectors.shape == (12, DIM)
    assert np.allclose(vectors, fake_embed(texts_of(store)), atol=1e-3)  # row = passage id - 1


def test_passage_text():
    assert passage_text("Benzene", "", "Ring.") == "Benzene\nRing."
    assert passage_text("Photoelectric effect", "Threshold", "x") == "Photoelectric effect › Threshold\nx"


def test_resume_after_a_stop_embeds_only_the_rest(l2_store, dense_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(embed_l2, "SAVE_S", 0)  # save after every window (2 x 1 x 4 = 8 passages)
    calls = []

    def stops(texts):
        calls.append(texts)
        if len(calls) == 5:  # the first request of the second window
            raise KeyboardInterrupt
        return fake_embed(texts)

    out = tmp_path / "dense"
    with pytest.raises(KeyboardInterrupt):
        build(l2_store, out, stops)
    meta = json.loads((out / META).read_text(encoding="utf-8"))
    assert (meta["done_rows"], meta["complete"]) == (8, False)
    seen = []
    build(l2_store, out, lambda texts: seen.extend(texts) or fake_embed(texts))
    assert seen == texts_of(L2Store(l2_store))[8:]
    assert np.array_equal(np.load(out / VECTORS), np.load(dense_dir / VECTORS))
    assert json.loads((out / META).read_text(encoding="utf-8"))["complete"]


def test_complete_build_is_left_alone(l2_store, tmp_path):
    out = tmp_path / "dense"
    build(l2_store, out)
    calls = []
    build(l2_store, out, lambda texts: calls.append(texts) or fake_embed(texts))
    assert calls == []


@pytest.mark.parametrize("other", ["model", "store"])
def test_other_model_or_store_needs_restart(l2_store, tmp_path, other):
    out = tmp_path / "dense"
    build(l2_store, out)
    meta = json.loads((out / META).read_text(encoding="utf-8"))
    if other == "store":  # as if the literature store had been rebuilt since
        meta["store_built_at"] = "2000-01-01T00:00:00Z"
        (out / META).write_text(json.dumps(meta), encoding="utf-8")
    model = {**MODEL, "model_id": "another"} if other == "model" else MODEL
    before = (out / META).read_bytes()
    with pytest.raises(BuildError, match="--restart"):
        build(l2_store, out, model=model)
    assert (out / META).read_bytes() == before
    assert build(l2_store, out, model=model, restart=True)["complete"]


def test_gap_in_passage_ids_is_refused(l2_store, tmp_path):
    db = tmp_path / "gap.db"
    shutil.copy(l2_store, db)
    with sqlite3.connect(db) as con:
        con.execute("DELETE FROM passage WHERE id = 5")
    with pytest.raises(BuildError, match="without gaps"):
        build(db, tmp_path / "dense")


def test_too_long_passages_are_shortened(l2_store, tmp_path):
    def short_context(texts):
        if any(len(t) > 600 for t in texts):
            raise UpstreamError("input (700 tokens) is larger than the max context size (512 tokens)", 400)
        return fake_embed(texts)

    report = build(l2_store, tmp_path / "dense", short_context)
    long = sum(len(t) > 600 for t in texts_of(L2Store(l2_store)))
    assert long and report["shortened"] == long and report["complete"]


def test_other_backend_errors_stop_the_build(l2_store, tmp_path):
    def down(texts):
        raise UpstreamError("embeddings backend unreachable")

    with pytest.raises(UpstreamError):
        build(l2_store, tmp_path / "dense", down)


@pytest.mark.parametrize("factory", ["Flat", "IVF2,Flat"])
def test_index_finds_a_passage_from_its_own_vector(l2_store, store, tmp_path, factory):
    out = tmp_path / factory.replace(",", "_")
    build(l2_store, out, factory=factory)
    index = DenseIndex(out, MODEL["model_id"], store, nprobe=2)
    vector = fake_embed([texts_of(store)[6]])[0]
    hits = index.search(vector, rerank=5)
    assert hits[0][0] == 7 and hits[0][1] == pytest.approx(1, abs=1e-3)
    assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)


# --- opening ---

def test_open_refuses_a_missing_or_unfinished_folder(store, tmp_path, l2_store):
    with pytest.raises(L2Error, match="ingest.embed_l2"):
        DenseIndex(tmp_path / "none", MODEL["model_id"], store, 1)
    out = tmp_path / "dense"
    build(l2_store, out)
    meta = json.loads((out / META).read_text(encoding="utf-8"))
    (out / META).write_text(json.dumps({**meta, "complete": False}), encoding="utf-8")
    with pytest.raises(L2Error, match="unfinished"):
        DenseIndex(out, MODEL["model_id"], store, 1)


def test_open_refuses_another_model_or_store_build(store, dense_dir, tmp_path):
    with pytest.raises(L2Error, match="'other'.*--restart"):
        DenseIndex(dense_dir, "other", store, 1)
    stale = tmp_path / "stale"
    shutil.copytree(dense_dir, stale)
    meta = json.loads((stale / META).read_text(encoding="utf-8"))
    (stale / META).write_text(json.dumps({**meta, "passages": 99}), encoding="utf-8")
    with pytest.raises(L2Error, match="built for the literature store.*--restart"):
        DenseIndex(stale, MODEL["model_id"], store, 1)


# --- hybrid search ---

def test_paraphrase_found(store, hybrid):
    assert store.search("guaranine", 5) == []
    titles = [r.title for r in hybrid.search("guaranine", 2)]
    assert titles and set(titles) <= {"Caffeine", "caffeine"}


def test_agreement_wins_and_ties_go_to_the_lower_id(store):
    bm25 = store.bm25_ids("photoelectric threshold", 50)
    other = next(i for i in (12, 11, 10) if i not in bm25)
    assert len(bm25) >= 2
    h = HybridSearch(store, StubIndex([other, bm25[1]]), fake_embed, "", DENSE)
    got = h.search("photoelectric threshold", 3)
    ids = [r.citation.locator["passage"] for r in got]
    assert ids[0] == bm25[1]  # second in both rankings beats first in one
    assert ids[1:] == sorted([bm25[0], other])  # 1 / 61 each
    assert got[0].score == round(2 / 62, 6) and got[1].score == round(1 / 61, 6)
    by_id = dict(zip(ids, got))  # passages of the dense ranking carry their similarity, the others do not
    assert isinstance(by_id[bm25[1]], DensePassageRecord) and by_id[bm25[1]].similarity == 0.89
    assert isinstance(by_id[other], DensePassageRecord) and by_id[other].similarity == 0.9
    assert type(by_id[bm25[0]]) is PassageRecord


def test_records_match_the_keyword_search(store, hybrid):
    keyword = {r.citation.locator["passage"]: r for r in store.search("photoelectric effect", 50)}
    base_fields = [f.name for f in dataclasses.fields(PassageRecord) if f.name != "score"]
    for r in hybrid.search("photoelectric effect", 5):
        pid = r.citation.locator["passage"]
        if pid in keyword:
            assert {f: getattr(r, f) for f in base_fields} == {f: getattr(keyword[pid], f) for f in base_fields}


def test_domain_filter(store):
    chemistry_only = [6, 7, 8, 9, 10, 11, 12]
    h = HybridSearch(store, StubIndex(chemistry_only), fake_embed, "", DENSE)
    got = h.search("electron", 5, domain="physics")
    assert got and all("physics" in r.domains for r in got)


def test_no_content_word_finds_nothing(hybrid):
    assert hybrid.search("the of and", 5) == []


def test_k_limits_the_passages(hybrid):
    assert len(hybrid.search("photoelectric effect electron", 2)) == 2


def test_query_prefix_is_sent(store):
    sent = []
    h = HybridSearch(store, StubIndex([1]), lambda texts: sent.extend(texts) or fake_embed(texts), "Q: ", DENSE)
    h.search("electron", 1)
    assert sent == ["Q: electron"]
