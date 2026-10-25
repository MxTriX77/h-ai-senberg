import json
import sqlite3

import pytest

from haisenberg.l2.cli import main as cli
from haisenberg.l2.store import L2Error, L2Store
from ingest.build_l2 import build
from ingest.l1base import BuildError

from .conftest import LIMIT, build_store
from .fixtures import make_raw, write_manifest


def rows(db, sql, *args):
    con = sqlite3.connect(db)
    try:
        return con.execute(sql, args).fetchall()
    finally:
        con.close()


def docs(db) -> dict[str, list[tuple]]:
    """doc path -> its passages (title, heading, text, domains)."""
    out: dict[str, list[tuple]] = {}
    for path, *rest in rows(db, "SELECT d.path, p.title, p.heading, p.text, p.domains FROM passage p"
                                " JOIN doc d ON d.id = p.doc_id ORDER BY p.id"):
        out.setdefault(path, []).append(tuple(rest))
    return out


def test_report(l2_built):
    db, report = l2_built
    assert report["passages"] == rows(db, "SELECT count(*) FROM passage")[0][0] > 0
    assert report["skipped_person_articles"] == 1  # Marie Curie
    assert report["skipped_duplicate_passages"] >= 2  # the LibreTexts course copy
    assert report["chebi_fixture: docs"] == 1


def test_wikipedia_articles(l2_store):
    found = docs(l2_store)
    assert "Marie_Curie" not in found and "Electrons" not in found and "Some_Film" not in found
    assert {d for *_, d in found["Electron"]} == {"chemistry physics"}  # stored once, both topic ZIMs' domains
    assert rows(l2_store, "SELECT count(*) FROM doc WHERE path = 'Electron'")[0][0] == 1
    assert {d for *_, d in found["Benzene"]} == {"chemistry"}  # from the filtered ZIM, by its chembox
    sub = "Emission mechanism › Experimental observation of photoelectric emission"
    full = [f"{h} › {t}" if h else t for _, h, t, _ in found["Photoelectric_effect"]]  # heading, then line headings
    assert any(f"{sub}: The emitted electrons" in x or x.startswith(f"{sub} › The emitted electrons") for x in full)
    text = " ".join(t for _, _, t, _ in found["Photoelectric_effect"])
    assert "E=h\\nu" in text and "Lenard" in text and "REFERENCE LIST TEXT" not in text


def test_libretexts_pages(l2_store):
    found = docs(l2_store)
    assert "Bookshelves/General_Chemistry/00:_Front_Matter/01:_TitlePage" not in found
    assert "Courses/X/15.04:_Coupled_Equilibria" not in found  # every passage a duplicate of page 2
    page = found["Bookshelves/General_Chemistry/15.04:_Coupled_Equilibria"]
    example = [t for _, h, t, _ in page if h == "Example 1: Increased Solubility in Acidic Solutions"]
    assert example and "Solution: \\[K = K_{sp} / K_{a2} = 180" in example[0] and "Significance:" in example[0]


def test_stackexchange_answers(l2_store):
    found = docs(l2_store)
    assert "a/12" not in found  # score 1 < min_answer_score 3
    ((title, heading, text, domains),) = found["a/11"]
    assert (title, heading, domains) == ('Why is benzene "aromatic"?', "Answer (score 13)", "chemistry")
    assert text.startswith("Benzene is aromatic because")
    everything = json.dumps(found)
    assert "QUESTION BODY TEXT" not in everything and "COMMENT TEXT" not in everything and "USER PAGE" not in everything


def test_wikibooks_only_listed_books(l2_store):
    found = docs(l2_store)
    assert "General_Chemistry/Acids" in found and "Cookbook/Bread" not in found


def test_chebi(l2_store):
    found = docs(l2_store)
    assert "CHEBI:1" not in found and "CHEBI:2" not in found  # no definition / obsolete
    ((title, heading, text, _),) = found["CHEBI:27732"]
    assert (title, heading) == ("caffeine", "")
    assert text.startswith("caffeine (CHEBI:27732): A trimethylxanthine")
    assert "Formula C8H10N4O2; mass 194.194; charge 0." in text
    assert "Is a: trimethylxanthine." in text and "Roles: central nervous system stimulant." in text
    assert "Synonyms: 1,3,7-trimethylxanthine; theine." in text  # 'Caffeine' is the name itself


def test_provenance(l2_store):
    src = {r[0]: r[1:] for r in rows(l2_store, "SELECT name, sha256, version, title, license FROM source")}
    assert src["libretexts_fixture"][1:] == ("2026-10-01", "LibreTexts Chemistry", "test licence")
    assert src["chebi_fixture"][1:3] == ("255", "ChEBI")
    assert all(len(sha) == 64 for sha, *_ in src.values())


def test_missing_input(tmp_path):
    raw = make_raw(tmp_path / "raw")
    (raw / "chebi/chebi.obo.gz").unlink()
    with pytest.raises(BuildError, match="chebi_fixture"):
        build(write_manifest(tmp_path / "sources.yaml"), raw, tmp_path / "l2.db", LIMIT)


def test_failed_build_keeps_store_and_layer1(tmp_path):
    db, _ = build_store(tmp_path)
    layer1 = tmp_path / "encyclopedia_fts.db"
    layer1.write_bytes(b"layer 1 bytes")
    before = (db.read_bytes(), layer1.read_bytes(), layer1.stat().st_mtime_ns)
    (tmp_path / "raw/chebi/chebi.obo.gz").write_bytes(b"not gzip")
    with pytest.raises(OSError):
        build(tmp_path / "sources.yaml", tmp_path / "raw", db, LIMIT)
    assert (db.read_bytes(), layer1.read_bytes(), layer1.stat().st_mtime_ns) == before
    assert not list(tmp_path.glob("*.building"))


# --- search ---------------------------------------------------------------------------------------------------


def test_search_ranks_page(l2_store):
    hits = L2Store(l2_store).search("coupled equilibria solubility", 3)
    assert hits[0].title == "15.4: Coupled Equilibria"
    assert hits[0].collection == "LibreTexts Chemistry"
    assert hits[0].citation.url.startswith("https://chem.libretexts.org/Bookshelves/General_Chemistry/")
    assert hits[0].label().endswith("(LibreTexts Chemistry)")


def test_search_domain_filter(l2_store):
    store = L2Store(l2_store)
    assert store.search("photoelectric", 5)
    assert store.search("photoelectric", 5, "chemistry") == []
    assert all("physics" in r.domains for r in store.search("electron", 5, "physics"))


def test_search_any_word_fallback_and_stopwords(l2_store):
    store = L2Store(l2_store)
    assert store.search("the of and", 5) == []
    assert [r.title for r in store.search("benzene zzzunknownword", 5)]  # no passage has both: any word


def test_distinct_passages_have_distinct_citations(l2_store):
    hits = L2Store(l2_store).search("photoelectric", 10)
    assert len({(r.citation.url, r.citation.locator["passage"]) for r in hits}) == len(hits)


def test_wrong_or_missing_file(built_store, tmp_path):
    with pytest.raises(L2Error, match="ingest.build_l2"):
        L2Store(built_store)  # the Layer 1 store
    with pytest.raises(L2Error, match="ingest.build_l2"):
        L2Store(tmp_path / "nothing.db")


def test_cli_json(l2_store, capsys):
    assert cli(["search", "benzene aromatic", "--db", str(l2_store), "--json"]) == 0
    records = json.loads(capsys.readouterr().out)
    assert records and all(r["citation"]["url"] for r in records)
    assert records[0]["citation"]["url"] == "https://chemistry.stackexchange.com/a/11"
