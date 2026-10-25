"""Tiny Layer 2 inputs for the tests: ZIMs built with libzim's writer and a ChEBI OBO file.

Pages copy the markup patterns of the real files (Kiwix 2026 Wikipedia/Wikibooks pages, the LibreTexts
`content/*.json` layout, sotoki Stack Exchange question pages, ChEBI release 255 stanzas) with short or
paraphrased text.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import yaml
from libzim.writer import Creator, Hint, Item, StringProvider

WIKI_CITE = "https://en.wikipedia.org/w/index.php?title={path}&oldid={revision}"
MINI_FILTER = {
    "infoboxes": {"ib-chembox": "chemistry", "ib-element": "chemistry"},
    "projects": {"Chemistry": "chemistry", "Chemicals": "chemistry", "Elements": "chemistry", "Physics": "physics"},
    "importance": ["Top", "High", "Mid"],
}
MATH = (
    '<span class="mwe-math-element mwe-math-element-inline"><span class="mwe-math-mathml-a11y" style="display: none;">'
    '<math alttext="{\\displaystyle E=h\\nu }"><semantics><mrow><mi>E</mi></mrow></semantics></math></span>'
    '<img src="./_assets_/e.svg" class="mwe-math-fallback-image-inline" alt="{\\displaystyle E=h\\nu }"></span>'
)
REF = '<sup class="mw-ref reference"><a href="#cite_note-1"><span class="mw-reflink-text">[1]</span></a></sup>'
LONG = " ".join(f"Sentence number {i} about the photoelectric threshold frequency and electrons." for i in range(40))


def wiki_page(title: str, sections: list[tuple[int, str, str]], rev: int = 100, projects: dict | None = None,
              categories=(), box: str = "") -> str:
    """A mwoffliner page: lead (with an optional infobox) then (level, heading, html) sections."""
    assessments = {p: {"class": "B", "importance": imp} for p, imp in (projects or {}).items()}
    config = json.dumps({"wgCategories": list(categories), "wgCurRevisionId": rev, "wgPageAssessments": assessments},
                        separators=(",", ":"))
    body, open_levels = "", []  # Parsoid nests a sub-section inside its parent section
    for n, (lvl, h, content) in enumerate(sections, 1):
        while open_levels and open_levels[-1] >= lvl:
            body += "</section>"
            open_levels.pop()
        body += (f'<section data-mw-section-id="{n}"><div class="mw-heading mw-heading{lvl}">'
                 f'<h{lvl} id="{h.replace(" ", "_")}">{h}</h{lvl}></div>{content}')
        open_levels.append(lvl)
    body += "</section>" * len(open_levels)
    return (
        f'<!DOCTYPE html><html><head><meta charset="UTF-8"><title>{title}</title><script>RLCONF = {config};</script>'
        f'</head><body><h1 id="firstHeading">{title}</h1><div class="mw-parser-output">'
        f'<section data-mw-section-id="0">{box}<p>{title} is a fixture lead paragraph about {title.lower()}.{REF}</p>'
        f"</section>{body}</div></body></html>"
    )


def stub(target: str) -> str:
    return f'<html><head><meta http-equiv="refresh" content="0;URL=\'../{target}\'" /></head><body></body></html>'


PHOTO = wiki_page(
    "Photoelectric effect",
    [
        (2, "Emission mechanism", "<p>Photons carry energy " + MATH + " and eject electrons from a metal surface "
         "when that energy exceeds the work function of the metal.</p>"),
        (3, "Experimental observation of photoelectric emission",
         "<p>The emitted electrons have a maximum kinetic energy that depends on the frequency of the light, "
         "not on its intensity, as Lenard observed in 1902.</p>"
         '<table class="wikitable"><tr><td>TABLE CELL TEXT</td></tr></table>'
         '<div role="navigation" class="navbox">NAVBOX TEXT</div>'),
        (2, "Threshold", f"<p>{LONG}</p>"),
        (2, "References", "<p>REFERENCE LIST TEXT that must not be stored.</p>"),
    ],
    projects={"Physics": "Top"},
)
ELECTRON = wiki_page("Electron", [(2, "Properties", "<p>The electron has a charge of minus one elementary charge "
                                   "and a rest mass of about 9.109e-31 kg.</p>")], projects={"Physics": "Top"})
CURIE = wiki_page("Marie Curie", [(2, "Life", "<p>She studied radioactivity in Paris.</p>")],
                  projects={"Biography": "Top", "Chemistry": "Top"})
BENZENE = wiki_page("Benzene", [(2, "Structure", "<p>Benzene is a planar ring of six carbon atoms with delocalised "
                                 "pi electrons, the archetypal aromatic compound.</p>")],
                    box='<table class="infobox ib-chembox"><tr><td>Benzene</td></tr></table>')
CAFFEINE = wiki_page("Caffeine", [(2, "Pharmacology", "<p>Caffeine blocks adenosine receptors in the brain and so "
                                     "acts as a central nervous system stimulant.</p>")])  # no ratings, no chembox
CURIE_TOP = wiki_page("Marie Curie", [(2, "Work", "<p>Fixture text about the discovery of polonium and radium.</p>")])
CAFFEINE_MINI = wiki_page("Caffeine", [], projects={"Chemicals": "Mid"})  # the mini copy carries the ratings
CURIE_MINI = wiki_page("Marie Curie", [], projects={"Biography": "", "Chemistry": "High"})
FILM = wiki_page("Some Film", [(2, "Plot", "<p>A fixture film whose plot has nothing to do with science at all.</p>")])

INTRO = " ".join(["The solubility of calcium carbonate rises in acid as carbonate ions are protonated."] * 5)
TASK = " ".join(["Compute the solubility of calcium carbonate in water held at pH 7 by a buffer."] * 5)
LT_EXAMPLE = (
    f'<p>{INTRO}</p>'
    '<section class="box-example"><div class="mt-section"><h5 class="box-legend">'
    'Example \\(\\PageIndex{1}\\): Increased Solubility in Acidic Solutions</h5>'
    f'<p>{TASK}</p>'
    '<h6>Strategy</h6><p>Combine the Ksp and Ka2 equilibria.</p>'
    '<h6>Solution</h6><p>\\[K = K_{sp} / K_{a2} = 180 \\nonumber \\]</p>'
    '<h6>Significance</h6><p>Acid dissolves limestone.</p></div></section>'
    '<p>Coupled equilibria also govern complex-ion formation with ammonia and silver ions.</p>'
)
SE_PAGE = (
    '<html><body><h1 itemprop="name" class="fs-headline1"><a href="../../questions/7/why-is-benzene-aromatic" '
    'class="question-hyperlink">Why is benzene &quot;aromatic&quot;?</a></h1>'
    '<div class="postcell"><div class="s-prose js-post-body" itemprop="text"><p>QUESTION BODY TEXT asking about '
    'benzene.</p></div></div><div id="answers">'
    '<div id="answer-11" class="answer " data-answerid="11" data-score="13" itemprop="acceptedAnswer">'
    '<div class="s-prose js-post-body" itemprop="text"><p>Benzene is aromatic because its six pi electrons form a '
    'closed cyclic conjugated system obeying the 4n+2 rule with n = 1.</p></div>'
    '<div class="comments"><span class="comment-copy">COMMENT TEXT</span></div></div>'
    '<div id="answer-12" class="answer " data-answerid="12" data-score="1" itemprop="suggestedAnswer">'
    '<div class="s-prose js-post-body" itemprop="text"><p>LOW SCORE ANSWER TEXT about benzene smell.</p></div></div>'
    "</div></body></html>"
)
BOOK_PAGE = wiki_page("General Chemistry/Acids", [(2, "Definitions", "<p>A Bronsted acid donates a proton to a "
                                                   "base; a Lewis acid accepts an electron pair.</p>")])
COOK_PAGE = wiki_page("Cookbook/Bread", [(2, "Method", "<p>Knead the dough for ten minutes and let it rise.</p>")])

OBO = """format-version: 1.2
data-version: 255

[Term]
id: CHEBI:27732
name: caffeine
def: "A trimethylxanthine in which the three methyl groups are located at positions 1, 3, and 7." []
synonym: "1,3,7-trimethylxanthine" RELATED [nist]
synonym: "Caffeine" RELATED [kegg]
synonym: "theine" RELATED [nist]
is_a: CHEBI:27134 ! trimethylxanthine
relationship: RO:0000087 CHEBI:35337 ! has role central nervous system stimulant
property_value: chemrof:charge "0" xsd:integer
property_value: chemrof:generalized_empirical_formula "C8H10N4O2" xsd:string
property_value: chemrof:mass "194.194" xsd:decimal

[Term]
id: CHEBI:1
name: undefined thing
synonym: "no definition" RELATED []

[Term]
id: CHEBI:2
name: old thing
def: "An obsolete fixture term." []
is_obsolete: true

[Typedef]
id: RO:0000087
name: has role
"""


class _Entry(Item):
    def __init__(self, path: str, content: str, mimetype: str = "text/html", title: str | None = None):
        super().__init__()
        self._path, self._content, self._mime, self._title = path, content, mimetype, title

    def get_path(self):
        return self._path

    def get_title(self):
        return self._title or self._path.replace("_", " ")

    def get_mimetype(self):
        return self._mime

    def get_contentprovider(self):
        return StringProvider(self._content)

    def get_hints(self):
        return {Hint.FRONT_ARTICLE: self._mime == "text/html"}


def _zim(out: Path, title: str, items: list[_Entry]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with Creator(str(out)).config_indexing(False, "eng") as c:
        c.set_mainpath(items[0].get_path())
        for item in items:
            c.add_item(item)
        c.add_metadata("Name", out.stem)
        c.add_metadata("Title", title)
        c.add_metadata("Date", "2026-10-01")
    for leftover in out.parent.glob(out.name + "_*"):  # writer temp files
        if leftover.is_file():
            leftover.unlink()


def make_raw(raw: Path) -> Path:
    """The Layer 2 fixture inputs under raw/."""
    _zim(raw / "kiwix/chem.zim", "Wikipedia Chemistry", [
        _Entry("Electron", ELECTRON), _Entry("Marie_Curie", CURIE), _Entry("Electrons", stub("Electron"))])
    _zim(raw / "kiwix/phys.zim", "Wikipedia Physics", [
        _Entry("Photoelectric_effect", PHOTO), _Entry("Electron", ELECTRON)])
    _zim(raw / "kiwix/top.zim", "Wikipedia top", [
        _Entry("Benzene", BENZENE), _Entry("Some_Film", FILM), _Entry("Electron", ELECTRON),
        _Entry("Caffeine", CAFFEINE), _Entry("Marie_Curie", CURIE_TOP)])
    _zim(raw / "kiwix/mini.zim", "Wikipedia mini",
         [_Entry("Caffeine", CAFFEINE_MINI), _Entry("Marie_Curie", CURIE_MINI)])
    pages = [{"id": "1", "title": "Home", "path": ""},
             {"id": "2", "title": "15.4: Coupled Equilibria",
              "path": "Bookshelves/General_Chemistry/15.04:_Coupled_Equilibria"},
             {"id": "3", "title": "TitlePage", "path": "Bookshelves/General_Chemistry/00:_Front_Matter/01:_TitlePage"},
             {"id": "4", "title": "Copy of 15.4", "path": "Courses/X/15.04:_Coupled_Equilibria"}]
    _zim(raw / "kiwix/libretexts.zim", "LibreTexts Chemistry", [
        _Entry("index.html", "<html></html>"),
        _Entry("content/shared.json", json.dumps({"pages": pages}), "application/json"),
        _Entry("content/page_content_2.json", json.dumps({"htmlBody": LT_EXAMPLE}), "application/json"),
        _Entry("content/page_content_3.json", json.dumps({"htmlBody": "<p>TITLE PAGE TEXT that must be skipped by "
                                                                      "the build.</p>"}), "application/json"),
        _Entry("content/page_content_4.json", json.dumps({"htmlBody": LT_EXAMPLE}), "application/json"),
    ])
    _zim(raw / "stackexchange/chem_se.zim", "Chemistry Stack Exchange", [
        _Entry("questions/7/why-is-benzene-aromatic", SE_PAGE),
        _Entry("users/5/someone", "<html><body><p>USER PAGE TEXT</p></body></html>")])
    _zim(raw / "kiwix/wikibooks.zim", "Wikibooks", [
        _Entry("General_Chemistry/Acids", BOOK_PAGE), _Entry("Cookbook/Bread", COOK_PAGE)])
    (raw / "chebi").mkdir(parents=True, exist_ok=True)
    (raw / "chebi/chebi.obo.gz").write_bytes(gzip.compress(OBO.encode()))
    return raw


def _entry(name, use, domains, dest, cite_url, flt=None) -> dict:
    e = {"name": name, "use": use, "layer": 2, "domains": domains, "url": f"https://example.org/{name}",
         "dest": dest, "sha256": "0" * 64, "license": "test licence", "cite_url": cite_url}
    return {**e, "filter": flt} if flt else e


SOURCES = [
    _entry("chem_fixture", "wikipedia-zim", ["chemistry"], "kiwix/chem.zim", WIKI_CITE),
    _entry("phys_fixture", "wikipedia-zim", ["physics"], "kiwix/phys.zim", WIKI_CITE),
    _entry("top_fixture", "wikipedia-filtered-zim", ["chemistry", "physics"], "kiwix/top.zim", WIKI_CITE, MINI_FILTER),
    {**_entry("mini_fixture", "wikipedia-mini-zim", ["chemistry", "physics"], "kiwix/mini.zim", WIKI_CITE, MINI_FILTER),
     "layer": 1},  # a Layer 1 input; the Layer 2 build reads only its ratings
    _entry("libretexts_fixture", "libretexts-zim", ["chemistry"], "kiwix/libretexts.zim",
           "https://chem.libretexts.org/{path}"),
    _entry("se_fixture", "stackexchange-zim", ["chemistry"], "stackexchange/chem_se.zim",
           "https://chemistry.stackexchange.com/{path}", {"min_answer_score": 3}),
    _entry("wikibooks_fixture", "wikibooks-zim", ["chemistry"], "kiwix/wikibooks.zim",
           "https://en.wikibooks.org/wiki/{path}", {"books": {"General_Chemistry": "chemistry"}}),
    _entry("chebi_fixture", "chebi-obo", ["chemistry"], "chebi/chebi.obo.gz",
           "https://www.ebi.ac.uk/chebi/searchId.do?chebiId={path}"),
]


def write_manifest(path: Path) -> Path:
    path.write_text(yaml.safe_dump({"sources": SOURCES}), encoding="utf-8")
    return path
