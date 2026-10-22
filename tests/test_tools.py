"""Agent tools generated from the fixture store (agent-tools spec)."""

import json
import logging

import pytest

from haisenberg.config import ConfigError
from haisenberg.l1.store import L1Store
from haisenberg.tools import Knowledge, Sources


@pytest.fixture
def knowledge(cfg, built_store):
    return Knowledge(L1Store(built_store), cfg)


def names(k):
    return [t["function"]["name"] for t in k.tools]


def test_tools_from_the_shipped_kinds(knowledge, cfg):
    assert names(knowledge) == ["search_knowledge", "lookup_compound", "lookup_constant", "calculate"]
    search = knowledge.tools[0]
    assert search["type"] == "function"
    params = search["function"]["parameters"]
    assert params["required"] == ["query"]
    assert params["properties"]["domain"]["enum"] == ["chemistry", "physics"]
    assert params["properties"]["query"]["description"] == cfg.tools["search_knowledge"].parameters["query"]
    lookup = knowledge.tools[1]["function"]
    assert lookup["description"] == cfg.tools["lookup_compound"].description
    assert list(lookup["parameters"]["properties"]) == ["query"]


def test_new_source_of_a_known_kind_adds_its_domain(cfg, tmp_path):
    from ingest.build_l1 import build
    from l1.conftest import make_raw
    from l1.l1_helpers import FIXTURE_SOURCES, write_manifest

    extra = {**FIXTURE_SOURCES[0], "name": "wikipedia_en_biology_fixture", "domains": ["biology"]}
    db = tmp_path / "l1.db"
    build(write_manifest(tmp_path / "sources.yaml", [*FIXTURE_SOURCES, extra]), make_raw(tmp_path), db)
    k = Knowledge(L1Store(db), cfg)
    assert k.tools[0]["function"]["parameters"]["properties"]["domain"]["enum"] == ["biology", "chemistry", "physics"]


def test_no_store_no_tools(cfg, caplog):
    with caplog.at_level(logging.WARNING):
        k = Knowledge.open(cfg)
    assert k.tools == []
    assert str(cfg.retrieval.layer1_encyclopedia_db) in caplog.text and "ingest.build_l1" in caplog.text


def test_depth_zero_no_tools(cfg, built_store):
    cfg.reasoning_policy.max_search_depth = 0
    assert Knowledge(L1Store(built_store), cfg).tools == []


def test_description_from_config(cfg, built_store):
    cfg.tools["lookup_constant"].description = "New text."
    k = Knowledge(L1Store(built_store), cfg)
    assert k.tools[2]["function"]["description"] == "New text."


def test_missing_texts_fail(cfg, built_store):
    store = L1Store(built_store)
    del cfg.tools["lookup_compound"]
    with pytest.raises(ConfigError, match=r"tools\.lookup_compound"):
        Knowledge(store, cfg)


def test_missing_parameter_text_fails(cfg, built_store):
    del cfg.tools["search_knowledge"].parameters["domain"]
    with pytest.raises(ConfigError, match=r"tools\.search_knowledge\.parameters\.domain"):
        Knowledge(L1Store(built_store), cfg)


def run(k, name, args, sources=None, n=None):
    text, lines, _ = k.run(name, args if isinstance(args, str) else json.dumps(args), sources or Sources(), n)
    return text, lines


def test_search_restricted_to_a_domain(knowledge):
    sources = Sources()
    run(knowledge, "search_knowledge", {"query": "electron", "domain": "physics"}, sources)
    assert sources.items and all("physics" in s.record.domains for s in sources.items)
    text, lines = run(knowledge, "search_knowledge", {"query": "ethanol", "domain": "physics"})
    assert text.startswith("Nothing found") and lines == ["nothing found"]


@pytest.mark.parametrize("name, args, why", [
    ("lookup_compound", "not json", "not valid JSON"),
    ("lookup_compound", "[1, 2]", "JSON object"),
    ("lookup_compound", {"query": " "}, "'query'"),
    ("lookup_compound", {}, "'query'"),
    ("search_knowledge", {"query": "x", "domain": "biology"}, "unknown domain"),
    ("lookup_everything", {"query": "x"}, "unknown tool"),
])
def test_bad_calls_return_error_lines(knowledge, name, args, why):
    text, lines = run(knowledge, name, args)
    assert text.startswith("Error:") and why in text and lines[0].startswith("error:")


def test_numbering_across_calls(knowledge):
    sources = Sources()
    first, _ = run(knowledge, "lookup_constant", {"query": "electron mass energy equivalent"}, sources)
    assert first.startswith("[S1] electron mass energy equivalent (constant; physics, chemistry)\n")
    n = len(sources.items)
    second, lines = run(knowledge, "lookup_constant", {"query": "electron mass"}, sources)
    assert "[S1] electron mass energy equivalent (shown above)" in second
    assert "[S1] electron mass energy equivalent (shown above)" in lines
    assert sources.items[0].passage not in second  # passage not repeated
    assert len(sources.items) > n and f"[S{n + 1}] electron mass (constant" in second  # numbering continues


def test_relevant_infobox_line_kept(knowledge):
    knowledge.passage_chars = 80
    sources = Sources()
    run(knowledge, "search_knowledge", {"query": "ethanol chemical formula"}, sources)
    ethanol = next(s for s in sources.items if s.record.label() == "Ethanol")
    assert "Chemical formula: C2H6O" in ethanol.passage and len(ethanol.passage) <= 80


def test_constant_and_compound_passages(knowledge):
    sources = Sources()
    run(knowledge, "lookup_constant", {"query": "Planck constant"}, sources)
    p = sources.items[0].passage
    assert "6.626 070 15 e-34" in p and "exact" in p and "J Hz^-1" in p and "codata_fixture, version 2022" in p
    run(knowledge, "lookup_compound", {"query": "caffeine"}, sources)
    caffeine = next(s for s in sources.items if s.kind == "compound")
    assert "CID 2519" in caffeine.passage and "194.19" in caffeine.passage and "IUPAC" in caffeine.passage


def test_source_list_only_cited_and_issued(knowledge):
    sources = Sources()
    run(knowledge, "lookup_compound", {"query": "caffeine"}, sources)
    run(knowledge, "lookup_constant", {"query": "Planck constant"}, sources)
    planck = next(s for s in sources.items if s.kind == "constant")
    listing = sources.source_list(f"Caffeine is C8H10N4O2 [S1]. See [{planck.id}] and [S99].")
    assert listing.startswith("\n\nSources:\n- [S1] Caffeine (CID 2519): pubchem_mass_fixture")
    assert f"[{planck.id}] Planck constant" in listing and "S99" not in listing
    assert "https://pubchem.ncbi.nlm.nih.gov/compound/2519" in listing
    assert sources.source_list("No citations here.") == ""


def test_record_count_and_ids(knowledge):
    sources = Sources()
    text, lines, ids = knowledge.run("lookup_constant", json.dumps({"query": "electron mass"}), sources, 2)
    assert ids == ["S1", "S2"] and len(lines) == 2
    assert sources.get("S2") is sources.items[1] and sources.get("S3") is None and sources.get("x") is None


def test_calculator_next_to_the_knowledge_tools(knowledge, cfg):
    calc_tool = knowledge.tools[-1]["function"]
    assert calc_tool["description"] == cfg.tools["calculate"].description
    assert calc_tool["parameters"]["required"] == ["expression"]
    sources = Sources()
    text, lines, ids = knowledge.run("calculate", json.dumps({"expression": "0.0500 * 56.08"}), sources)
    assert text == "0.0500 * 56.08 = 2.804" and lines == [text] and ids == [] and not sources.items
    assert knowledge.run("calculate", '{"expression": "__import__(1)"}', sources)[0].startswith("Error: not allowed")


def test_no_calculator_without_the_store(cfg):
    assert Knowledge(None, cfg).tools == []
