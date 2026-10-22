import sqlite3

import pytest

from ingest.wikipedia import first_section, stub_target
from ingest.zim import Zim

from .wiki_fixtures import CHEM, PHYS


@pytest.fixture(scope="module")
def con(built_store):
    return sqlite3.connect(built_store)


def article(con, path):
    row = con.execute(
        "SELECT a.title, a.lead, a.infobox, a.revision, a.zims, s.name FROM article a JOIN source s ON s.id = a.source_id"
        " WHERE a.path = ?",
        (path,),
    ).fetchone()
    return None if row is None else dict(zip(("title", "lead", "infobox", "revision", "zims", "source"), row))


def alias_target(con, alias):
    return [r[0] for r in con.execute(
        "SELECT a.path FROM article_alias x JOIN article a ON a.id = x.article_id WHERE x.alias = ?", (alias,)
    )]


def test_exponents_survive(con):
    box = article(con, "Planck_constant")["infobox"]
    assert "10^−34" in box and "Hz^−1" in box
    assert "Value: 6.62607015×10^−34 J⋅Hz^−1" in box


def test_formulas_references_and_math(con):
    lead = article(con, "Ethanol")["lead"]
    assert "CH3CH2OH" in lead and "[1]" not in lead
    planck = article(con, "Planck_constant")["lead"]
    assert planck.count(r"{\displaystyle h}") == 1 and "[2]" not in planck


def test_only_first_section_is_kept(con):
    assert con.execute("SELECT count(*) FROM article WHERE lead LIKE '%LATER SECTION%'").fetchone()[0] == 0


def test_nested_infobox_tables_left_out(con):
    box = article(con, "Iron")["infobox"]
    assert "Hydrogen" not in box and "Standard atomic weight: 55.845±0.002" in box
    assert "Skeletal formula" not in article(con, "Ethanol")["infobox"]


def test_article_in_both_topic_zims_kept_once_with_higher_revision(con):
    e = article(con, "Electron")
    assert e["revision"] == 300 and e["lead"].startswith("PHYSICS COPY") and "e^−" in e["lead"]
    assert e["zims"] == "wikipedia_en_chemistry_fixture wikipedia_en_physics_fixture"
    assert con.execute("SELECT count(*) FROM article WHERE path = 'Electron'").fetchone()[0] == 1


def test_redirect_pages_are_not_articles(con):
    assert article(con, "Percent_yield") is None and article(con, "%_yield") is None


def test_redirects_and_stubs_become_aliases(con):
    assert alias_target(con, "% yield") == ["Yield_(chemistry)"]
    assert alias_target(con, "percent yield") == ["Yield_(chemistry)"]  # meta-refresh stub
    assert alias_target(con, "heisenberg uncertainty principle") == ["Uncertainty_principle"]
    assert alias_target(con, "c6h6") == ["Benzene"]  # mini ZIM redirect to a mini article
    assert alias_target(con, "electron (particle)") == ["Electron"]  # mini redirect to a topic article
    assert alias_target(con, "planck constant") == ["Planck_constant"]  # normalized title


def test_person_articles_excluded(con, built_store):
    for path in ("Marie_Curie", "Max_Planck", "Albert_Einstein"):  # project, infobox, category rule
        assert article(con, path) is None
    meta = dict(con.execute("SELECT key, value FROM meta"))
    assert meta["skipped_person_articles"] == "3"


def test_mini_fills_gaps_only(con):
    assert article(con, "Benzene")["zims"] == "wikipedia_en_all_mini_fixture"
    assert article(con, "Sodium_chloride") is not None  # chembox without a chem/phys project
    assert article(con, "Thermal_conductance") is not None  # Physics, Mid importance
    assert article(con, "Obscure_low_topic") is None  # Physics, Low importance, no infobox
    assert article(con, "Football") is None
    assert "MINI COPY" not in article(con, "Electron")["lead"]


def test_linked_compounds_point_to_their_article(con):
    rows = dict(con.execute(
        "SELECT c.cid, a.path FROM compound c JOIN article a ON a.id = c.article_id WHERE c.cid IN (241, 702, 5234)"
    ))
    assert rows == {241: "Benzene", 702: "Ethanol", 5234: "Sodium_chloride"}


def test_fts_covers_every_article(con):
    n = con.execute("SELECT count(*) FROM article").fetchone()[0]
    assert n == 10  # chemistry 3 + physics 3 new (Electron shared) + mini 4; persons and stubs excluded
    hits = con.execute("SELECT rowid FROM article_fts WHERE article_fts MATCH 'subatomic'").fetchall()
    assert len(hits) == 1


def test_cluster_order_covers_every_html_entry(fixture_raw):
    with Zim(fixture_raw / "kiwix/chem_fixture.zim") as z:
        by_cluster = {z.strings(i)[0] for i in z.html_by_cluster()}
        assert by_cluster == set(CHEM["pages"])
        assert z.find("Ethanol") is not None and z.find("Nope") is None
        assert {z.strings(i)[0]: z.strings(t)[0] for i, t in z.redirects()} == CHEM["redirects"]
    with Zim(fixture_raw / "kiwix/phys_fixture.zim") as z:
        assert {z.strings(i)[0] for i in z.html_by_cluster()} == set(PHYS["pages"])


def test_stub_target_and_missing_section():
    assert stub_target(CHEM["pages"]["Percent_yield"].encode()) == "Yield_(chemistry)"
    assert stub_target(CHEM["pages"]["Ethanol"].encode()) is None
    assert first_section(b"<html><body><p>no sections</p></body></html>") is None


def domains(con, path):
    sql = "SELECT d.domain FROM article_domain d JOIN article a ON a.id = d.article_id WHERE a.path = ? ORDER BY 1"
    return [r[0] for r in con.execute(sql, (path,))]


def test_article_domains(con):
    assert domains(con, "Electron") == ["chemistry", "physics"]  # in both topic ZIMs
    assert domains(con, "Ethanol") == ["chemistry"]
    assert domains(con, "Planck_constant") == ["physics"]
    assert domains(con, "Thermal_conductance") == ["physics"]  # mini: only WikiProject Physics (Mid) matched
    assert domains(con, "Sodium_chloride") == ["chemistry"]  # mini: chembox only
    assert con.execute("SELECT count(*) FROM article a WHERE NOT EXISTS"
                       " (SELECT 1 FROM article_domain d WHERE d.article_id = a.id)").fetchone()[0] == 0


def test_source_domains(con):
    rows = dict(con.execute("SELECT name, domains FROM source"))
    assert rows["codata_fixture"] == "physics chemistry"
    assert rows["wikipedia_en_all_mini_fixture"] == "chemistry physics"


def test_mini_filter_changed_in_config_only(fixture_raw, tmp_path):
    from ingest.build_l1 import build

    from .l1_helpers import FIXTURE_SOURCES, MINI_FILTER, write_manifest

    sources = [
        {**s, "filter": {**MINI_FILTER, "projects": {**MINI_FILTER["projects"], "Sports": "physics"}}}
        if s["use"] == "wikipedia-mini-zim" else s
        for s in FIXTURE_SOURCES
    ]
    db = tmp_path / "l1.db"
    build(write_manifest(tmp_path / "sources.yaml", sources), fixture_raw, db)
    con = sqlite3.connect(db)
    assert article(con, "Football") is not None and domains(con, "Football") == ["physics"]
    assert article(con, "Obscure_low_topic") is None
