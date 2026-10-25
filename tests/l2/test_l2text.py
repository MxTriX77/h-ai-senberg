from ingest.l2text import Block, blocks, chunks

from .fixtures import LONG, PHOTO


def test_heading_path_and_dropped_parts():
    found = blocks(PHOTO)
    paths = {b.path for b in found}
    assert ("Emission mechanism", "Experimental observation of photoelectric emission") in paths
    text = " ".join(b.text for b in found)
    for gone in ("REFERENCE LIST TEXT", "TABLE CELL TEXT", "NAVBOX TEXT", "[1]"):
        assert gone not in text
    # the <title> and the h1 page title are not text; the lead is
    assert found[0] == Block((), "Photoelectric effect is a fixture lead paragraph about photoelectric effect.")
    assert not any("References" in b.path for b in found)


def test_math_kept_as_tex():
    assert any("E=h\\nu" in b.text for b in blocks(PHOTO))


def test_long_paragraph_split_at_sentences():
    parts = chunks([Block(("Threshold",), LONG)], 1200)
    assert len(LONG) > 3000 and len(parts) >= 3
    assert all(len(t) <= 1200 and t.endswith(".") for _, t in parts)
    assert {h for h, _ in parts} == {"Threshold"}


def test_worked_example_kept_together():
    ex = ("Example 1",)
    task = "Compute the solubility of calcium carbonate at pH 7. " * 12
    found = chunks([
        Block(ex, task.strip()),
        Block(ex + ("Strategy",), "Combine the two equilibria."),
        Block(ex + ("Solution",), "K = Ksp / Ka2 = 180."),
        Block(ex + ("Significance",), "Acid dissolves limestone."),
        Block((), " ".join(["Coupled equilibria also govern complex-ion formation with ammonia and silver ions."] * 2)),
    ], 1200)
    heading, text = found[0]
    assert heading == "Example 1"
    lines = text.splitlines()
    assert lines[1:] == ["Strategy: Combine the two equilibria.", "Solution: K = Ksp / Ka2 = 180.",
                         "Significance: Acid dissolves limestone."]
    tail = "Coupled equilibria also govern complex-ion formation with ammonia and silver ions."
    assert found[1] == ("", f"{tail} {tail}")


def test_short_sections_merge_under_shared_headings():
    first = "First block of text under B, long enough to stay in."
    second = "Second block of text under C, kept here as well."
    found = chunks([Block(("A", "B"), first), Block(("A", "C"), second)], 1200)
    assert found == [("A", f"B: {first}\nC: {second}")]  # over MIN_CHARS only together


def test_tiny_passages_dropped():
    assert chunks([Block(("X",), "Video 1: A summary of enthalpy and Hess' law from a video course.")], 1200) == []


def test_attribution_sections_dropped():
    page = ("<h2>Hess's law</h2><p>" + "Enthalpy is a state function. " * 5 + "</p>"
            "<h2>Contributors and Attributions</h2><p>" + "Authored by someone. License: CC BY. " * 5 + "</p>")
    assert [b.path for b in blocks(page)] == [("Hess's law",)]


def test_root_class_limits_text():
    page = ('<div><p>OUTSIDE text here.</p><div class="js-post-body"><p>Inside the post body.</p><div>nested</div>'
            '</div><p>AFTER text here.</p></div>')
    assert [b.text for b in blocks(page, "js-post-body")] == ["Inside the post body.", "nested"]


def test_heading_ends_with_its_section():
    page = '<section class="box"><h5>Learning Objectives</h5><p>Objective one.</p></section><p>Body text.</p>'
    assert blocks(page) == [Block(("Learning Objectives",), "Objective one."), Block((), "Body text.")]
