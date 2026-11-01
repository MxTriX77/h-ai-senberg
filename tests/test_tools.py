"""Agent tools generated from the fixture store (agent-tools spec)."""

import dataclasses
import json
import logging

import pytest

from haisenberg.config import ConfigError
from haisenberg.l1.store import L1Store
from haisenberg.tools import Knowledge, Sources, term_share


@pytest.fixture
def knowledge(cfg, built_store):
    return Knowledge(L1Store(built_store), cfg)


def names(k):
    return [t["function"]["name"] for t in k.tools]


def test_tools_from_the_shipped_kinds(knowledge, cfg):
    assert names(knowledge) == ["search_knowledge", "lookup_compound", "lookup_constant", "calculate", "compute"]
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
    assert str(cfg.retrieval.path("encyclopedia")) in caplog.text and "ingest.build_l1" in caplog.text


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


# --- relevance of search hits --------------------------------------------------------------------------


def hits(knowledge, *records):
    """Make the store's search return these (title, lead, score) records, built on a real fixture record."""
    template = knowledge.store.search_encyclopedia("ethanol", 1)[0]
    made = [dataclasses.replace(template, title=t, lead=lead, infobox="", snippet="", score=score,
                                citation=dataclasses.replace(template.citation, url=f"https://x/{t}"))
            for t, lead, score in records]
    knowledge.store.search_encyclopedia = lambda query, k=None, domain=None: made


@pytest.fixture
def relevant(cfg, built_store):
    cfg.retrieval.min_term_share = 0.5
    return Knowledge(L1Store(built_store), cfg)


def test_one_shared_word_is_not_enough(relevant):
    hits(relevant, ("Journal of Optics", "Journal of Optics is a journal about light.", -3.2))
    text, lines, ids = relevant.run("search_knowledge", json.dumps({"query": "journal publisher not given"}), Sources())
    assert text == "Nothing relevant found for 'journal publisher not given'." and ids == []
    assert lines == ["weak match, not used: Journal of Optics", "nothing relevant found"]


def test_weak_hit_dropped_strong_hit_kept(relevant):
    hits(relevant, ("Ethanol", "Ethanol boils at 78 °C at normal pressure.", -5.0),
         ("Boiling", "Boiling is a phase change.", -2.0))
    text, lines, ids = relevant.run("search_knowledge", json.dumps({"query": "ethanol boiling point pressure"}), Sources())
    assert ids == ["S1"] and "Ethanol" in text and "Boiling is a phase" not in text
    assert lines == ["[S1] Ethanol", "weak match, not used: Boiling"]


def test_named_article_kept(relevant):
    hits(relevant, ("Gold", "A precious metal.", 0.0))
    assert relevant.run("search_knowledge", json.dumps({"query": "gold standard economics"}), Sources())[2] == ["S1"]


def test_filter_off(knowledge):
    hits(knowledge, ("Journal of Optics", "Journal of Optics is a journal about light.", -3.2))
    assert knowledge.run("search_knowledge", json.dumps({"query": "journal publisher not given"}), Sources())[2] == ["S1"]


def test_term_share():
    assert term_share("boiling point of ethanol", "Ethanol boils at 78 °C") == pytest.approx(2 / 3)
    assert term_share("half-life of carbon-14", "Carbon-14 has a half-life of 5730 years") == 1.0
    assert term_share("the of", "anything") == 1.0


def test_real_store_keeps_the_ranked_first_hit(cfg):
    from haisenberg.config import load_config
    from l1.l1_helpers import real_path

    db = real_path("encyclopedia")
    real = load_config()
    real.retrieval.min_term_share = 0.5
    knowledge = Knowledge(L1Store(db), real)
    for query, title in [("boiling point of ethanol", "Ethanol"), ("melting point of iron", "Iron"),
                         ("molar mass of caffeine", "Caffeine"), ("half-life of carbon-14", "Carbon-14"),
                         ("ionization energy of hydrogen", "Hydrogen")]:
        _, lines, _ = knowledge.run("search_knowledge", json.dumps({"query": query}), Sources())
        assert lines[0] == f"[S1] {title}", (query, lines)


def test_calculator_next_to_the_knowledge_tools(knowledge, cfg):
    calc_tool = knowledge.tools[names(knowledge).index("calculate")]["function"]
    assert calc_tool["description"] == cfg.tools["calculate"].description
    assert calc_tool["parameters"]["required"] == ["expressions"]
    assert calc_tool["parameters"]["properties"]["expressions"]["type"] == "array"
    sources = Sources()
    text, lines, ids = knowledge.run("calculate", json.dumps({"expressions": ["0.0500 * 56.08"]}), sources)
    assert text == "0.0500 * 56.08 = 2.804" and lines == [text] and ids == [] and not sources.items
    assert knowledge.run("calculate", '{"expressions": ["__import__(1)"]}', sources)[0].startswith("Error: not allowed")


def test_several_calculations_in_one_call(knowledge):
    text, lines, _ = knowledge.run("calculate", json.dumps({"expressions": ["2 * 3", "1 / 0", "sqrt(16)"]}), Sources())
    assert lines[0] == "2 * 3 = 6" and lines[1].startswith("Error") and lines[2] == "sqrt(16) = 4"
    assert text == "\n".join(lines)
    # written as text, the list arrives as a string; a plain string is one expression
    assert knowledge.run("calculate", json.dumps({"expressions": '["2 * 3", "4 / 2"]'}), Sources())[1] == ["2 * 3 = 6", "4 / 2 = 2"]
    assert knowledge.run("calculate", json.dumps({"expressions": "2 * 3"}), Sources())[1] == ["2 * 3 = 6"]
    for bad in ['{"expressions": []}', '{"expression": "2 * 3"}', "not json"]:
        assert knowledge.run("calculate", bad, Sources())[0].startswith("Error:"), bad


def test_no_calculator_without_the_store(cfg):
    assert Knowledge(None, cfg).tools == []


def test_compute_right_after_the_calculator(knowledge, cfg):
    tool = knowledge.tools[-1]["function"]
    assert names(knowledge)[-2:] == ["calculate", "compute"]
    assert tool["description"] == cfg.tools["compute"].description
    assert tool["parameters"]["required"] == ["operations"]
    ops = tool["parameters"]["properties"]["operations"]
    assert ops["type"] == "array" and ops["description"] == cfg.tools["compute"].parameters["operations"]


def test_compute_lines_and_no_source(knowledge):
    sources = Sources()
    ops = ["weekday(2026-10-29)", "weekday(2026-02-30)", "sort(3, 1, 2)"]
    text, lines, ids = knowledge.run("compute", json.dumps({"operations": ops}), sources)
    assert lines == ["weekday(2026-10-29) = Thursday",
                     "weekday(2026-02-30): error: unknown date 2026-02-30 (February 2026 has 28 days)",
                     "sort(3, 1, 2) = 1, 2, 3 (as numbers)"]
    assert text == "\n".join(lines) and ids == [] and not sources.items
    # written as text, the list arrives as a string; a plain string is one operation
    assert run(knowledge, "compute", {"operations": '["count(a, b)", "count(a)"]'})[1] == [
        "count(a, b) = 2 items (2 distinct)", "count(a) = 1 item (1 distinct)"]
    assert run(knowledge, "compute", {"operations": "count(a, b)"})[1] == ["count(a, b) = 2 items (2 distinct)"]
    for bad in ['{"operations": []}', '{"operation": "count(a)"}', "not json"]:
        assert knowledge.run("compute", bad, Sources())[0].startswith("Error:"), bad
    assert run(knowledge, "compute", {"operations": [3]})[1] == [
        "error: an operation must be a non-empty string, e.g. weekday(2026-10-29)"]


def test_compute_switched_off(cfg, built_store):
    cfg.reasoning_policy.compute_tool = False
    k = Knowledge(L1Store(built_store), cfg)
    assert names(k)[-1] == "calculate" and not k.computer
    assert run(k, "compute", {"operations": ["count(a)"]})[0].startswith("Error: unknown tool 'compute'")


def test_compute_needs_its_texts(cfg, built_store):
    del cfg.tools["compute"]
    with pytest.raises(ConfigError, match=r"tools\.compute"):
        Knowledge(L1Store(built_store), cfg)


def test_no_compute_without_the_calculator(cfg, built_store):
    del cfg.tools["calculate"]
    k = Knowledge(L1Store(built_store), cfg)
    assert "compute" not in names(k) and not k.computer
