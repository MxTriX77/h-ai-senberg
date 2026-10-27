"""Layer 2 source adapters: each turns the raw files of one kind of manifest entry into documents with passages.

Wikipedia ZIMs (`wikipedia-zim`, `wikipedia-text-zim`, `wikipedia-filtered-zim`) and Wikibooks (`wikibooks-zim`)
are mwoffliner HTML; LibreTexts (`libretexts-zim`) keeps its page tree in `content/shared.json` and each page body
in `content/page_content_<id>.json`; Stack Exchange (`stackexchange-zim`) has one HTML page per question with its
answers; ChEBI (`chebi-obo`) is an OBO text file. Pages are parsed in worker processes (ZIM entries in cluster
order); the build writes what they return.
"""

from __future__ import annotations

import gzip
import html as html_lib
import json
import os
import re
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

from ingest.l2text import blocks, chunks
from ingest.manifest import BookFilter, MiniFilter
from ingest.wikipedia import _HEAD, _REVISION, is_person, mini_relevant, projects, stub_target
from ingest.zim import Zim

LIBRETEXTS_SKIP = frozenset({"TitlePage", "InfoPage", "Table of Contents", "Licensing", "Detailed Licensing", "Index"})
_STOP_H2 = re.compile(
    rb'<h2 id="(?:References|External_links|See_also|Further_reading|Notes|Bibliography|Sources|Citations|Footnotes)"'
)
_PAGE_INDEX = re.compile(r"(?:\\\(\s*)?\\PageIndex\{([^}]*)\}(?:\s*\\\))?")  # \(\PageIndex{2}\) -> 2
_SE_QUESTION = re.compile(r"questions/\d+/")
_SE_TITLE = re.compile(r'<h1 itemprop="name"[^>]*>(.*?)</h1>', re.S)
_SE_ANSWER = re.compile(r'<div id="answer-(\d+)" class="answer[^"]*"[^>]*?data-score="(-?\d+)"')
_TAG = re.compile(r"<[^>]+>")


@dataclass
class Doc:
    path: str
    title: str
    passages: list[tuple[str, str]]  # (heading path, text)
    revision: int | None = None
    domains: list[str] = field(default_factory=list)  # empty: the entry's domains


@dataclass
class Scan:
    docs: list[Doc]
    dropped: dict[str, int]  # report counts, e.g. skipped person articles


def _parallel(job, args: list[tuple]) -> Iterator:
    """Run job over args in worker processes, results in order."""
    workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    if len(args) == 1 or workers == 1:
        yield from map(job, args)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(job, args)


def _jobs(ids: list, n_workers_x4: int = 32) -> list[list]:
    size = max(1, -(-len(ids) // n_workers_x4))  # contiguous runs keep clusters together
    return [ids[k : k + size] for k in range(0, len(ids), size)]


# --- Wikipedia and Wikibooks ----------------------------------------------------------------------------------


def _article(html: bytes, limit: int) -> list[tuple[str, str]]:
    """Passages of a whole mwoffliner page; the trailing reference sections are cut off before parsing."""
    end = _STOP_H2.search(html)
    return chunks(blocks(html[: end.start() if end else len(html)].decode("utf-8", "replace")), limit)


def scan_wiki(zim_path: str, ids: list[int], limit: int, mini: MiniFilter | None, skip: frozenset[str],
              rated: dict[str, list[str]]) -> Scan:
    """Articles of a Wikipedia ZIM: every non-person article, or (with mini) the ones not in skip that are relevant
    under mini by their own page or, by path, in rated (a page without WikiProject ratings rated elsewhere)."""
    docs, dropped = [], {"skipped_person_articles": 0}
    with Zim(zim_path) as z:
        for i in ids:
            html = z.content(i)
            if stub_target(html):
                continue
            path, title = z.strings(i)
            page_projects = projects(html)
            domains = mini_relevant(html, page_projects, mini) | set(rated.get(path, ())) if mini else set()
            if mini and not domains:
                continue
            if path in skip:
                continue
            if is_person(html, page_projects):
                dropped["skipped_person_articles"] += 1
                continue
            rev = _REVISION.search(html, 0, _HEAD)
            docs.append(Doc(path, title, _article(html, limit), int(rev.group(1)) if rev else None, sorted(domains)))
    return Scan(docs, dropped)


def scan_books(zim_path: str, ids: list[int], limit: int, books: BookFilter) -> Scan:
    """Pages of the listed Wikibooks books, each with its book's domain."""
    docs = []
    with Zim(zim_path) as z:
        for i in ids:
            path, title = z.strings(i)
            domain = books.books.get(path.split("/", 1)[0])
            if domain is None:
                continue
            html = z.content(i)
            if stub_target(html):
                continue
            rev = _REVISION.search(html, 0, _HEAD)
            docs.append(Doc(path, title, _article(html, limit), int(rev.group(1)) if rev else None, [domain]))
    return Scan(docs, {})


def wiki_scans(zim: Zim, limit: int, mini: MiniFilter | None = None, skip: frozenset[str] = frozenset(),
               books: BookFilter | None = None, rated: dict[str, list[str]] | None = None) -> Iterator[Scan]:
    ids = zim.html_by_cluster()
    if books is not None:
        yield from _parallel(_books_job, [(str(zim.path), part, limit, books) for part in _jobs(ids)])
    else:
        yield from _parallel(_wiki_job, [(str(zim.path), part, limit, mini, skip, rated or {}) for part in _jobs(ids)])


def scan_rated(zim_path: str, ids: list[int], rules: MiniFilter) -> dict[str, list[str]]:
    """Path -> domains of the non-person articles of a ZIM relevant under rules (the mini ZIM's WikiProject ratings)."""
    found = {}
    with Zim(zim_path) as z:
        for i in ids:
            html = z.content(i)
            page_projects = projects(html)
            domains = mini_relevant(html, page_projects, rules)
            if domains and not stub_target(html) and not is_person(html, page_projects):
                found[z.strings(i)[0]] = sorted(domains)
    return found


def _rated_job(args) -> dict[str, list[str]]:
    return scan_rated(*args)


def rated(zim: Zim, rules: MiniFilter) -> dict[str, list[str]]:
    """scan_rated over the whole ZIM."""
    found: dict[str, list[str]] = {}
    for part in _parallel(_rated_job, [(str(zim.path), ids, rules) for ids in _jobs(zim.html_by_cluster())]):
        found.update(part)
    return found


def _wiki_job(args) -> Scan:
    return scan_wiki(*args)


def _books_job(args) -> Scan:
    return scan_books(*args)


# --- LibreTexts -----------------------------------------------------------------------------------------------


def libretexts_pages(zim: Zim) -> list[tuple[int, str, str]]:
    """(page id, title, path) of every page in the ZIM's page tree that has a body, bodies in cluster order."""
    shared = json.loads(zim.content(zim.find("content/shared.json")))
    found = {}
    for p in shared["pages"]:
        if p["title"] in LIBRETEXTS_SKIP:
            continue
        i = zim.find(f"content/page_content_{p['id']}.json")
        if i is not None:
            found[i] = (int(p["id"]), p["title"], p["path"])
    return [found[i] for i in zim.by_cluster(list(found))]


def scan_libretexts(zim_path: str, pages: list[tuple[int, str, str]], limit: int) -> Scan:
    docs = []
    with Zim(zim_path) as z:
        for page_id, title, path in pages:
            body = json.loads(z.content(z.find(f"content/page_content_{page_id}.json"))).get("htmlBody") or ""
            docs.append(Doc(path, title, chunks(blocks(_PAGE_INDEX.sub(r"\1", body)), limit)))
    return Scan(docs, {})


def _libretexts_job(args) -> Scan:
    return scan_libretexts(*args)


def libretexts_scans(zim: Zim, limit: int) -> Iterator[Scan]:
    yield from _parallel(_libretexts_job, [(str(zim.path), part, limit) for part in _jobs(libretexts_pages(zim))])


# --- Stack Exchange -------------------------------------------------------------------------------------------


def answers(page: str, min_score: int, limit: int) -> list[Doc]:
    """The answers of one question page voted at least min_score, each a document titled with the question."""
    m = _SE_TITLE.search(page)
    title = " ".join(html_lib.unescape(_TAG.sub("", m.group(1))).split()) if m else ""
    found = list(_SE_ANSWER.finditer(page))
    docs = []
    for n, a in enumerate(found):
        score = int(a.group(2))
        if score < min_score:
            continue
        part = page[a.start() : found[n + 1].start() if n + 1 < len(found) else len(page)]
        head = f"Answer (score {score})"
        passages = [(f"{head} › {h}" if h else head, t) for h, t in chunks(blocks(part, "js-post-body"), limit)]
        docs.append(Doc(f"a/{a.group(1)}", title, passages))
    return docs


def scan_stackexchange(zim_path: str, ids: list[int], limit: int, min_score: int) -> Scan:
    docs = []
    with Zim(zim_path) as z:
        for i in ids:
            if _SE_QUESTION.match(z.strings(i)[0]):
                docs.extend(answers(z.content(i).decode("utf-8", "replace"), min_score, limit))
    return Scan(docs, {})


def _stackexchange_job(args) -> Scan:
    return scan_stackexchange(*args)


def stackexchange_scans(zim: Zim, limit: int, min_score: int) -> Iterator[Scan]:
    ids = zim.html_by_cluster()
    yield from _parallel(_stackexchange_job, [(str(zim.path), part, limit, min_score) for part in _jobs(ids)])


# --- ChEBI ----------------------------------------------------------------------------------------------------

_OBO_QUOTED = re.compile(r'"((?:[^"\\]|\\.)*)"')


def chebi_terms(path) -> tuple[str, list[dict]]:
    """(release, terms) from a ChEBI OBO file (gzip or plain); each term is a dict of its lines' values."""
    opener = gzip.open if str(path).endswith(".gz") else open
    release, terms, term = "", [], None
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("data-version:"):
                release = line.split(":", 1)[1].strip()
            if line.startswith("["):
                term = {} if line == "[Term]" else None
                if term is not None:
                    terms.append(term)
            elif term is not None and ": " in line:
                key, value = line.split(": ", 1)
                term.setdefault(key, []).append(value)
    return release, terms


def chebi_docs(terms: list[dict], limit: int) -> list[Doc]:
    """One document per defined, non-obsolete term: definition, formula/mass/charge, parents, roles, synonyms."""
    docs = []
    for t in terms:
        if "def" not in t or t.get("is_obsolete") == ["true"] or "name" not in t:
            continue
        name, cid = t["name"][0], t["id"][0]
        props = {}
        for pv in t.get("property_value", []):
            key, _, rest = pv.partition(" ")
            if q := _OBO_QUOTED.search(rest):
                props[key.split(":")[-1]] = q.group(1)
        parts = [f"{name} ({cid}): {_unquote(t['def'][0])}"]
        facts = [f"{label} {props[key]}" for key, label in
                 (("generalized_empirical_formula", "formula"), ("mass", "mass"), ("charge", "charge")) if key in props]
        if facts:
            line = "; ".join(facts)
            parts.append(line[0].upper() + line[1:] + ".")
        if parents := [v.split(" ! ", 1)[1] for v in t.get("is_a", []) if " ! " in v]:
            parts.append(f"Is a: {'; '.join(parents)}.")
        if roles := [v.split(" ! has role ", 1)[1] for v in t.get("relationship", []) if " ! has role " in v]:
            parts.append(f"Roles: {'; '.join(roles)}.")
        seen, synonyms = {name.lower()}, []
        for s in t.get("synonym", []):
            s = _unquote(s)
            if s.lower() not in seen and len(synonyms) < 10:
                seen.add(s.lower())
                synonyms.append(s)
        if synonyms:
            parts.append(f"Synonyms: {'; '.join(synonyms)}.")
        text = " ".join(parts)
        if len(text) > limit:
            text = text[:limit].rsplit(" ", 1)[0]
        docs.append(Doc(cid, name, [("", text)]))
    return docs


def _unquote(value: str) -> str:
    m = _OBO_QUOTED.match(value)
    return m.group(1).replace('\\"', '"') if m else value
