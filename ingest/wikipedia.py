"""Wikipedia articles from Kiwix ZIMs into `article`, `article_alias` and the FTS index.

Topic ZIMs (use `wikipedia-zim`) contribute every article except person articles. The all-articles
mini ZIM (use `wikipedia-mini-zim`) only fills gaps: articles not already stored that pass the
relevance filter (`mini_relevant`). From each article we keep the first section only: the lead
paragraphs and the infobox rows, as plain text. Redirects and meta-refresh stubs become aliases.
Infobox links to PubChem become `ctx.linked_cids` for the compound subset.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import unquote

from haisenberg.l1.text import normalize
from ingest.l1base import BuildContext
from ingest.zim import Zim

STUB_MAX_BYTES = 2048
CHEM_PHYS_PROJECTS = {"Chemistry", "Chemicals", "Physics", "Elements"}
MINI_IMPORTANCE = {"Top", "High", "Mid"}  # WikiProject ratings kept from the mini ZIM
RELEVANT_BOXES = (b"ib-chembox", b"ib-element")  # compound and element infoboxes

_ASSESSMENTS = re.compile(rb'"wgPageAssessments":\{(.*?\})\}')
_PROJECT = re.compile(rb'"([^"/]+)(?:/[^"]*)?":\{"class":"[^"]*","importance":"([^"]*)"')
_PERSON_CATEGORY = re.compile(rb'"(?:Living people|\d{1,4}s?(?: BC)? (?:births|deaths))"')
_PERSON_BOX = re.compile(rb'<table class="infobox[^"]*\bbiography\b')
_REVISION = re.compile(rb'"wgCurRevisionId":(\d+)')
_REFRESH = re.compile(rb"http-equiv=\"refresh\"[^>]*URL='([^']*)'")
_PUBCHEM_HREF = re.compile(r"/compound/(\d+)$")
_HEAD = 60000  # page config (assessments, categories) sits in the first bytes of the page


# --- page classification on raw bytes (cheap, before any parsing) -----------------------------


def projects(html: bytes) -> dict[str, str]:
    """WikiProject -> importance, from the page's assessments ('Physics/…' task forces fold into 'Physics')."""
    m = _ASSESSMENTS.search(html, 0, _HEAD)
    if not m:
        return {}
    found: dict[str, str] = {}
    for name, importance in _PROJECT.findall(m.group(1)):
        found.setdefault(name.decode("utf-8", "replace"), importance.decode("utf-8", "replace"))
    return found


def is_person(html: bytes, page_projects: dict[str, str]) -> bool:
    """WikiProject Biography, a biography infobox, or a births/deaths/'Living people' category."""
    return (
        "Biography" in page_projects
        or _PERSON_BOX.search(html) is not None
        or _PERSON_CATEGORY.search(html, 0, _HEAD) is not None
    )


def mini_relevant(html: bytes, page_projects: dict[str, str]) -> bool:
    """Relevance filter for the all-articles mini ZIM (design D11): a compound or element infobox,
    or a chemistry/physics WikiProject rating of Top, High or Mid importance."""
    if any(box in html for box in RELEVANT_BOXES):
        return True
    return any(page_projects.get(p) in MINI_IMPORTANCE for p in CHEM_PHYS_PROJECTS)


def stub_target(html: bytes) -> str | None:
    """Target path of a meta-refresh redirect page, or None if the page is a real article."""
    if len(html) > STUB_MAX_BYTES:
        return None
    m = _REFRESH.search(html)
    if not m:
        return None
    target = unquote(m.group(1).decode("utf-8", "replace")).split("#", 1)[0]
    while target.startswith(("../", "./")):
        target = target.split("/", 1)[1]
    return target or None


# --- first-section text --------------------------------------------------------------------------


class _FirstSection(HTMLParser):
    """Lead paragraphs (outside tables) and infobox rows (top-level `table.infobox`)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.paragraphs: list[str] = []
        self.rows: list[str] = []
        self.cids: set[int] = set()
        self._skip_tag: str | None = None  # element whose content is dropped
        self._skip_depth = 0
        self._math_pending = False  # inside a math element whose TeX is not emitted yet
        self._tables = 0
        self._box = 0  # table depth of the current infobox, 0 outside
        self._para: list[str] | None = None
        self._cell: list[str] | None = None
        self._row: list[str] = []
        self._row_hrefs: list[str] = []

    def _emit(self, text: str) -> None:
        if self._box and self._tables == self._box and self._cell is not None:
            self._cell.append(text)
        elif self._para is not None and self._tables == 0:
            self._para.append(text)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class") or ""
        if self._skip_tag:
            if tag == self._skip_tag:
                self._skip_depth += 1
            if self._math_pending and tag in ("math", "img"):
                tex = a.get("alttext") or a.get("alt")
                if tex:
                    self._emit(f" {tex} ")
                    self._math_pending = False
            return
        if tag in ("style", "script") or (tag == "sup" and "reference" in cls):
            self._skip_tag, self._skip_depth = tag, 1
            return
        if tag == "span" and "mwe-math-element" in cls:
            self._skip_tag, self._skip_depth, self._math_pending = "span", 1, True
            return
        if tag == "table":
            self._tables += 1
            if not self._box and self._tables == 1 and "infobox" in cls.split():
                self._box = 1
        elif tag == "p" and self._tables == 0:
            self._para = []
        elif self._box and self._tables == self._box:
            if tag == "tr":
                self._row, self._row_hrefs = [], []
            elif tag in ("th", "td"):
                self._cell = []
            elif tag == "a" and a.get("href"):
                self._row_hrefs.append(a["href"])
        if tag == "sup":
            self._emit("^")
        elif tag in ("br", "li"):
            self._emit(" ")

    def handle_endtag(self, tag):
        if self._skip_tag:
            if tag == self._skip_tag:
                self._skip_depth -= 1
                if self._skip_depth == 0:
                    self._skip_tag, self._math_pending = None, False
            return
        if tag == "table" and self._tables:
            if self._tables == self._box:
                self._box = 0
            self._tables -= 1
        elif tag == "p" and self._para is not None and self._tables == 0:
            text = " ".join("".join(self._para).split())
            if text:
                self.paragraphs.append(text)
            self._para = None
        elif self._box and self._tables == self._box:
            if tag in ("th", "td") and self._cell is not None:
                self._row.append(" ".join("".join(self._cell).split()))
                self._cell = None
            elif tag == "tr":
                cells = [c for c in self._row if c]
                if cells:
                    self.rows.append(": ".join(cells[:2]) if len(cells) > 1 else cells[0])
                    if cells[0].startswith("PubChem"):
                        for href in self._row_hrefs:
                            if m := _PUBCHEM_HREF.search(href):
                                self.cids.add(int(m.group(1)))
                self._row = []

    def handle_data(self, data):
        if not self._skip_tag:
            self._emit(data)


@dataclass
class Page:
    index: int  # ZIM entry index
    path: str
    title: str
    lead: str
    infobox: str
    revision: int | None
    cids: list[int] = field(default_factory=list)


def first_section(html: bytes) -> tuple[str, str, list[int]] | None:
    """(lead, infobox, linked PubChem CIDs) from the page's first section; None if it has none."""
    start = html.find(b'<section data-mw-section-id="0"')
    if start < 0:
        return None
    end = html.find(b"<section", start + 10)
    parser = _FirstSection()
    parser.feed(html[start : end if end > 0 else len(html)].decode("utf-8", "replace"))
    parser.close()
    return "\n".join(parser.paragraphs), "\n".join(parser.rows), sorted(parser.cids)


# --- scanning a ZIM (in worker processes) ------------------------------------------------------


def scan(zim_path: str, ids: list[int], mini: bool, skip_paths: frozenset[str]):
    """Pages to store and (alias title, target path) stubs from the given entries.

    Topic ZIMs keep every non-person article; the mini ZIM keeps relevant non-person articles
    whose path is not in skip_paths. Counts of dropped pages come back in `dropped`.
    """
    pages: list[Page] = []
    stubs: list[tuple[str, str]] = []
    dropped = {"skipped_person_articles": 0, "skipped_no_first_section": 0}
    with Zim(zim_path) as z:
        for i in ids:
            html = z.content(i)
            if target := stub_target(html):
                stubs.append((z.strings(i)[1], target))
                continue
            page_projects = projects(html)
            if mini and not mini_relevant(html, page_projects):
                continue
            if is_person(html, page_projects):
                dropped["skipped_person_articles"] += 1
                continue
            path, title = z.strings(i)
            if mini and path in skip_paths:
                continue
            section = first_section(html)
            if section is None:
                dropped["skipped_no_first_section"] += 1
                continue
            lead, infobox, cids = section
            rev = _REVISION.search(html, 0, _HEAD)
            pages.append(Page(i, path, title, lead, infobox, int(rev.group(1)) if rev else None, cids))
    return pages, stubs, dropped


def _scan_job(args):
    return scan(*args)


def _scan_parallel(zim_path: str, ids: list[int], mini: bool, skip: frozenset[str]):
    workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    size = max(1, -(-len(ids) // (workers * 4)))  # contiguous runs keep clusters together
    jobs = [(zim_path, ids[k : k + size], mini, skip) for k in range(0, len(ids), size)]
    if len(jobs) == 1:
        yield scan(*jobs[0])
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(_scan_job, jobs)


# --- load -----------------------------------------------------------------------------------------


@dataclass
class _Article:
    path: str
    title: str
    lead: str
    infobox: str
    revision: int | None
    source_id: int
    zims: list[str]
    cids: list[int]
    aliases: dict[str, str] = field(default_factory=dict)  # normalized -> display


def load(ctx: BuildContext) -> None:
    articles: dict[str, _Article] = {}
    stubs: list[tuple[str, str]] = []
    zims: list[tuple[Zim, dict[int, str]]] = []  # open ZIMs with entry index -> stored path

    def add_pages(entry, z: Zim, mini: bool) -> None:
        source_id = ctx.add_source(entry, z.metadata.get("Date", ""))
        stored: dict[int, str] = {}
        skip = frozenset(articles) if mini else frozenset()
        for pages, page_stubs, dropped in _scan_parallel(str(z.path), z.html_by_cluster(), mini, skip):
            stubs.extend(page_stubs)
            for k, v in dropped.items():
                ctx.report[k] = ctx.report.get(k, 0) + v
            for p in pages:
                stored[p.index] = p.path
                old = articles.get(p.path)
                if old is None:
                    articles[p.path] = _Article(
                        p.path, p.title, p.lead, p.infobox, p.revision, source_id, [entry.name], p.cids
                    )
                    continue
                old.zims.append(entry.name)
                if (p.revision or 0) > (old.revision or 0):
                    old.lead, old.infobox, old.revision, old.source_id, old.cids = (
                        p.lead, p.infobox, p.revision, source_id, p.cids
                    )
        zims.append((z, stored))

    try:
        for entry, path in ctx.files("wikipedia-zim"):
            add_pages(entry, Zim(path), mini=False)
        for entry, path in ctx.files("wikipedia-mini-zim"):
            z = Zim(path)
            add_pages(entry, z, mini=True)
            # the mini ZIM's redirects also name articles stored from the topic ZIMs
            stored = zims[-1][1]
            for p in articles:
                if (i := z.find(p)) is not None:
                    stored.setdefault(i, p)

        for a in articles.values():
            a.aliases[normalize(a.title)] = a.title
        for z, stored in zims:
            for i, target in z.redirects():
                if (path := stored.get(target)) is not None:
                    title = z.strings(i)[1]
                    articles[path].aliases.setdefault(normalize(title), title)
        for title, target in stubs:
            if (a := articles.get(target)) is not None:
                a.aliases.setdefault(normalize(title), title)
    finally:
        for z, _ in zims:
            z.close()

    con = ctx.con
    for a in articles.values():
        display = "\n".join(t for n, t in a.aliases.items() if t != a.title)
        cur = con.execute(
            "INSERT INTO article(path, title, aliases, infobox, lead, revision, source_id, zims)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (a.path, a.title, display, a.infobox, a.lead, a.revision, a.source_id, " ".join(a.zims)),
        )
        article_id = cur.lastrowid
        con.executemany(
            "INSERT OR IGNORE INTO article_alias(alias, article_id) VALUES (?, ?)",
            [(n, article_id) for n in a.aliases if n],
        )
        for cid in a.cids:
            ctx.linked_cids.setdefault(cid, article_id)
    con.execute("INSERT INTO article_fts(article_fts) VALUES ('rebuild')")
