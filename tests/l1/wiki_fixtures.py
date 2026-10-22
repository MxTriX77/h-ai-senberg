"""Tiny Wikipedia ZIMs for the tests, built with libzim's writer.

Pages copy the markup patterns of the real Kiwix 2026 pages (Parsoid sections, page config,
chembox PubChem row, math spans, reference markers, nested tables, meta-refresh stubs) with short
excerpts of the real text (CC BY-SA 4.0, Wikipedia contributors).
"""

from __future__ import annotations

import json
from pathlib import Path

from libzim.writer import Creator, Hint, Item, StringProvider

MATH_H = (
    '<span class="mwe-math-element mwe-math-element-inline" typeof="mw:Extension/math">'
    '<span class="mwe-math-mathml-a11y" style="display: none;"><math alttext="{\\displaystyle h}">'
    "<semantics><mrow><mi>h</mi></mrow>"
    '<annotation encoding="application/x-tex">{\\displaystyle h}</annotation></semantics></math></span>'
    '<img src="./_assets_/h.svg" class="mwe-math-fallback-image-inline" aria-hidden="true" alt="{\\displaystyle h}">'
    "</span>"
)
REF = '<sup class="mw-ref reference" typeof="mw:Extension/ref"><a href="#cite_note-{n}"><span class="mw-reflink-text">[{n}]</span></a></sup>'


def chembox(title: str, cid: int, formula_html: str) -> str:
    return (
        f'<table class="infobox ib-chembox"><caption>{title}</caption><tbody>'
        f'<tr><td colspan="2"><table><tbody><tr><td>Skeletal formula of {title.lower()}</td></tr></tbody></table></td></tr>'
        '<tr><th colspan="2">Identifiers</th></tr>'
        '<tr><td><div><a rel="mw:WikiLink" href="PubChem" title="PubChem">PubChem</a> <abbr title="Compound ID">CID</abbr></div></td>'
        '<td><div class="plainlist"><ul><li><span title="pubchem.ncbi.nlm.nih.gov">'
        f'<a rel="mw:ExtLink nofollow" href="https://pubchem.ncbi.nlm.nih.gov/compound/{cid}" class="external text">{cid}</a>'
        "</span></li></ul></div></td></tr>"
        f"<tr><td>Chemical formula</td><td>{formula_html}</td></tr>"
        "</tbody></table>"
    )


def page(title: str, first_section: str, rev: int, projects: dict | None = None, categories=()) -> str:
    assessments = {p: {"class": "B", "importance": imp} for p, imp in (projects or {}).items()}
    config = json.dumps(
        {"wgCategories": list(categories), "wgCurRevisionId": rev, "wgPageAssessments": assessments},
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return (
        '<!DOCTYPE html><html><head><meta charset="UTF-8">'
        f"<title>{title}</title><script>RLCONF = {config};</script></head><body>"
        '<div class="mw-content-ltr mw-parser-output">'
        f'<section data-mw-section-id="0" id="mwAQ">{first_section}</section>'
        '<section data-mw-section-id="1" id="mwBQ"><h2 id="History">History</h2>'
        "<p>LATER SECTION text that must not be stored.</p></section></div></body></html>"
    )


def stub(title: str, target: str) -> str:
    return (
        f'<html><head><title>{title}</title><meta http-equiv="refresh" content="0;URL=\'../{target}\'" />'
        f'</head><body><a href="../{target}">{title}</a></body></html>'
    )


CHEM = {
    "pages": {
        "Ethanol": page(
            "Ethanol",
            "<p>Ethanol (also called <b>ethyl alcohol</b>) is an organic compound with the chemical formula "
            f"CH<sub>3</sub>CH<sub>2</sub>OH.{REF.format(n=1)} It is a volatile, flammable, colorless liquid.</p>"
            + chembox("Ethanol", 702, "C<sub>2</sub>H<sub>6</sub>O"),
            rev=100,
            projects={"Chemicals": "Top", "Chemistry": "High"},
        ),
        "Electron": page(
            "Electron",
            "<p>CHEMISTRY COPY. The electron is a subatomic particle.</p>",
            rev=200,
            projects={"Chemistry": "Top"},
        ),
        "Yield_(chemistry)": page(
            "Yield (chemistry)",
            "<p>In chemistry, yield is a measure of the quantity of moles of a product formed.</p>",
            rev=110,
            projects={"Chemistry": "Mid"},
        ),
        "Percent_yield": stub("Percent yield", "Yield_(chemistry)#Percent_yield"),
        "Marie_Curie": page(
            "Marie Curie",
            "<p>Marie Curie was a Polish and naturalised-French physicist and chemist.</p>",
            rev=120,
            projects={"Biography": "Top", "Chemistry": "Top"},
            categories=["1867 births", "1934 deaths"],
        ),
    },
    "redirects": {"%_yield": "Yield_(chemistry)"},
}

PHYS = {
    "pages": {
        "Planck_constant": page(
            "Planck constant",
            f"<p>The <b>Planck constant</b>, denoted by {MATH_H}, is a fundamental physical constant.{REF.format(n=2)}</p>"
            '<table class="infobox"><tbody><tr><th colspan="2" class="infobox-above">Planck constant</th></tr>'
            f'<tr><th scope="row" class="infobox-label">Common symbols</th><td class="infobox-data">{MATH_H}</td></tr>'
            '<tr><th scope="row" class="infobox-label">SI unit</th><td class="infobox-data">J·s</td></tr>'
            '<tr><th scope="row" class="infobox-label">Value</th><td class="infobox-data"><span class="nowrap">'
            '<span data-sort-value="6966662607015000000♠"></span>6.626<span style="margin-left:.25em;">070</span>'
            '<span style="margin-left:.25em;">15</span><span style="margin-left:0.25em;margin-right:0.15em;">×</span>'
            "10<sup>−34</sup><span>&nbsp;</span>J⋅Hz<sup>−1</sup></span></td></tr></tbody></table>",
            rev=400,
            projects={"Physics": "Top"},
        ),
        "Electron": page(
            "Electron",
            "<p>PHYSICS COPY. The electron (e<sup>−</sup>) is a subatomic particle whose electric charge is negative.</p>",
            rev=300,
            projects={"Physics": "Top"},
        ),
        "Iron": page(
            "Iron",
            "<p><b>Iron</b> is a chemical element; it has symbol Fe and atomic number 26.</p>"
            '<table class="infobox ib-element"><tbody><tr><th colspan="2">Iron</th></tr>'
            '<tr><td colspan="2"><table class="ib-periodic-table"><tbody><tr><td>Hydrogen</td><td>Helium</td></tr>'
            "</tbody></table></td></tr><tr><th>Standard atomic weight</th><td>55.845±0.002</td></tr></tbody></table>",
            rev=410,
            projects={"Elements": "Top", "Physics": "Low"},
        ),
        "Uncertainty_principle": page(
            "Uncertainty principle",
            "<p>The uncertainty principle, also known as Heisenberg's indeterminacy principle, is a fundamental concept in quantum mechanics.</p>",
            rev=420,
            projects={"Physics": "Top"},
        ),
        "Max_Planck": page(
            "Max Planck",
            '<p>Max Planck was a German theoretical physicist.</p><table class="infobox biography vcard"><tbody>'
            "<tr><th>Born</th><td>23 April 1858</td></tr></tbody></table>",
            rev=430,
            projects={"Physics": "Top"},
        ),
    },
    "redirects": {"Heisenberg_uncertainty_principle": "Uncertainty_principle"},
}

MINI = {
    "pages": {
        "Benzene": page(
            "Benzene",
            "<p>Benzene is an organic chemical compound with the molecular formula C<sub>6</sub>H<sub>6</sub>."
            " Its boiling point is 80.1 °C.</p>"
            + chembox("Benzene", 241, "C<sub>6</sub>H<sub>6</sub>"),
            rev=500,
            projects={"Chemicals": "Top"},
        ),
        "Deuterated_benzene": page(
            "Deuterated benzene",
            "<p>Deuterated benzene is benzene with deuterium; boiling point 79.1 °C.</p>",
            rev=505,
            projects={"Chemistry": "Mid"},
        ),
        "Sodium_chloride": page(
            "Sodium chloride",
            "<p>Sodium chloride, commonly known as edible salt, is an ionic compound with the chemical formula NaCl.</p>"
            + chembox("Sodium chloride", 5234, "NaCl"),
            rev=510,
        ),
        "Electron": page("Electron", "<p>MINI COPY.</p>", rev=999, projects={"Physics": "Top"}),
        "Thermal_conductance": page(
            "Thermal conductance",
            "<p>Thermal conductance is the quantity of heat that passes in unit time through a plate.</p>",
            rev=540,
            projects={"Physics": "Mid"},
        ),
        "Obscure_low_topic": page(
            "Obscure low topic",
            "<p>A physics article rated low importance and left out of the topic ZIMs.</p>",
            rev=550,
            projects={"Physics": "Low"},
        ),
        "Football": page("Football", "<p>Football is a family of team sports.</p>", rev=520, projects={"Sports": "Top"}),
        "Albert_Einstein": page(
            "Albert Einstein",
            "<p>Albert Einstein was a German-born theoretical physicist.</p>",
            rev=530,
            projects={"Physics": "Top"},
            categories=["1879 births", "1955 deaths"],
        ),
    },
    "redirects": {"C6H6": "Benzene", "Electron_(particle)": "Electron"},
}

ZIMS = {
    "kiwix/chem_fixture.zim": ("wikipedia_en_chemistry_fixture", "2026-10-15", CHEM),
    "kiwix/phys_fixture.zim": ("wikipedia_en_physics_fixture", "2026-10-14", PHYS),
    "kiwix/mini_fixture.zim": ("wikipedia_en_all_mini_fixture", "2026-09-17", MINI),
}


class _Html(Item):
    def __init__(self, path: str, html: str):
        super().__init__()
        self._path, self._html = path, html

    def get_path(self):
        return self._path

    def get_title(self):
        return self._path.replace("_", " ")

    def get_mimetype(self):
        return "text/html"

    def get_contentprovider(self):
        return StringProvider(self._html)

    def get_hints(self):
        return {Hint.FRONT_ARTICLE: True}


def make_zims(raw: Path) -> None:
    for dest, (name, date, spec) in ZIMS.items():
        out = raw / dest
        out.parent.mkdir(parents=True, exist_ok=True)
        with Creator(str(out)).config_indexing(False, "eng") as c:
            c.set_mainpath(next(iter(spec["pages"])))
            for path, html in spec["pages"].items():
                c.add_item(_Html(path, html))
            for path, target in spec["redirects"].items():
                c.add_redirection(path, path.replace("_", " "), target, {Hint.FRONT_ARTICLE: True})
            c.add_metadata("Name", name)
            c.add_metadata("Date", date)
            c.add_metadata("Source", "en.wikipedia.org")
        for leftover in out.parent.glob(out.name + "_*"):  # writer temp files
            if leftover.is_file():
                leftover.unlink()
