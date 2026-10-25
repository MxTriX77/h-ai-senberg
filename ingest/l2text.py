"""HTML pages to headed text blocks, and blocks to passages, for the Layer 2 literature store.

One walker serves every HTML source (Wikipedia, Wikibooks, LibreTexts, Stack Exchange). It keeps the stack of
headings (h2-h6; h1 is the page title; a heading ends with the `section` element holding it, e.g. a LibreTexts
example box) and cuts the text into blocks at block-level elements. Math is written as
its TeX (`alttext`); tables, figures, reference markers, scripts, styles and elements with a SKIP_CLASSES class are
left out, and so are sections headed by one of STOP_HEADINGS. With `root_class`, only text inside an element of
that class is kept (Stack Exchange post bodies).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser

MIN_CHARS = 40  # shorter passages are dropped
STOP_HEADINGS = frozenset(
    "references|external links|see also|further reading|notes|bibliography|sources|citations|footnotes"
    "|notes and references|references and notes".split("|")
)
SKIP_TAGS = frozenset("head script style table figure nav footer noscript svg button form select textarea".split())
SKIP_CLASSES = frozenset(
    "reference mw-editsection navbox sidebar hatnote catlinks reflist mw-references-wrap noprint infobox metadata"
    " mw-cite-backlink toc thumb gallery subpages zim-footer".split()
)
BOUNDARY = frozenset("p li dd dt blockquote pre div section ul ol dl hr article header center".split())
VOID = frozenset("area base br col embed hr img input link meta param source track wbr".split())
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Block:
    path: tuple[str, ...]  # headings above the block inside the page, outermost first
    text: str


class _Walker(HTMLParser):
    def __init__(self, root_class: str | None):
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self._root_class = root_class
        self._root: list = [None, 0]  # [tag, depth] of the root element we are in; tag None: outside
        self._skip: list = [None, 0]  # [tag, depth] of the element whose content is dropped
        self._math_pending = False  # inside a math element whose TeX is not emitted yet
        self._heading: list[str] | None = None  # text of the heading being read
        self._level = 0
        self._path: list[tuple[int, str, int]] = []  # (level, heading, sections open around it)
        self._sections = 0
        self._stop = 0  # level of the stop section being skipped, 0 if none
        self._buf: list[str] = []

    def _collecting(self) -> bool:
        return not self._stop and (self._root_class is None or self._root[0] is not None)

    def _flush(self) -> None:
        text = " ".join("".join(self._buf).split())
        self._buf = []
        if text and self._collecting():
            self.blocks.append(Block(tuple(t for _, t, _ in self._path), text))

    def _emit(self, text: str) -> None:
        if self._heading is not None:
            self._heading.append(text)
        elif self._collecting():
            self._buf.append(text)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        classes = set((a.get("class") or "").split())
        if self._skip[0]:
            if tag == self._skip[0]:
                self._skip[1] += 1
            if self._math_pending and tag in ("math", "img") and (tex := a.get("alttext") or a.get("alt")):
                self._emit(f" {tex} ")
                self._math_pending = False
            return
        if self._root[0] == tag:
            self._root[1] += 1
        elif self._root[0] is None and self._root_class in classes:
            self._flush()
            self._root = [tag, 1]
            return
        if tag == "span" and "mwe-math-element" in classes:
            self._skip, self._math_pending = ["span", 1], True
            return
        if tag in SKIP_TAGS or classes & SKIP_CLASSES:
            if tag not in VOID:
                self._skip = [tag, 1]
            return
        if len(tag) == 2 and tag[0] == "h" and tag[1] in "123456":
            self._flush()
            self._heading, self._level = [], int(tag[1])
        elif tag in BOUNDARY:
            self._flush()
            self._sections += tag == "section"
        elif tag == "sup":
            self._emit("^")
        elif tag == "br":
            self._emit(" ")

    def handle_endtag(self, tag):
        if self._skip[0]:
            if tag == self._skip[0]:
                self._skip[1] -= 1
                if self._skip[1] == 0:
                    self._skip, self._math_pending = [None, 0], False
            return
        if self._heading is not None and tag == f"h{self._level}":
            title = " ".join("".join(self._heading).split())
            self._heading = None
            if self._level == 1:
                return
            if self._stop and self._level > self._stop:
                return
            self._stop = self._level if title.lower() in STOP_HEADINGS else 0
            kept = [h for h in self._path if h[0] < self._level]
            self._path = kept + ([(self._level, title, self._sections)] if title else [])
            return
        if tag in BOUNDARY:
            self._flush()
        if tag == "section" and self._sections:
            self._sections -= 1
            self._path = [h for h in self._path if h[2] <= self._sections]
        if self._root[0] == tag:
            self._root[1] -= 1
            if self._root[1] == 0:
                self._flush()
                self._root = [None, 0]

    def handle_data(self, data):
        if not self._skip[0]:
            self._emit(data)


def blocks(html: str, root_class: str | None = None) -> list[Block]:
    """The page's text blocks in reading order, each with the headings above it."""
    w = _Walker(root_class)
    w.feed(html)
    w.close()
    w._flush()
    return w.blocks


def _split(text: str, limit: int) -> list[str]:
    """text in pieces of at most limit characters: at sentence ends, else at word boundaries."""
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        while len(sentence) > limit:
            cut = sentence[:limit].rsplit(" ", 1)[0] if " " in sentence[:limit] else sentence[:limit]
            pieces.append(cut)
            sentence = sentence[len(cut):].lstrip()
        if pieces and len(pieces[-1]) + 1 + len(sentence) <= limit:
            pieces[-1] += " " + sentence
        elif sentence:
            pieces.append(sentence)
    return pieces


def _shared(a: tuple[str, ...], b: tuple[str, ...]) -> tuple[str, ...]:
    """The headings two paths share from the outermost one on."""
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return a[:n]


def chunks(page_blocks: list[Block], limit: int) -> list[tuple[str, str]]:
    """(heading path, text) passages of about limit characters at most. Blocks are packed while they share a heading
    (a worked example keeps its Solution) or sit under the same headings; a passage shorter than a third of the
    limit also takes in the next section. A passage carries the headings its blocks share; the first block of each
    sub-section in it is led by that sub-heading."""
    out: list[tuple[str, str]] = []
    cur: list[tuple[tuple[str, ...], str]] = []  # (path, text) of the blocks in the passage
    shared: tuple[str, ...] = ()
    size = 0

    def flush() -> None:
        lines, last = [], None
        for path, text in cur:
            sub = " › ".join(path[len(shared):])
            lines.append(f"{sub}: {text}" if sub and path != last else text)
            last = path
        text = "\n".join(lines)
        if len(text) >= MIN_CHARS:
            out.append((" › ".join(shared), text))
        cur.clear()

    for b in page_blocks:
        for piece in _split(b.text, limit):
            if cur:
                same = b.path == cur[-1][0]
                extra = len(piece) + 1 + (0 if same or not b.path else len(b.path[-1]) + 2)
                related = same or _shared(shared, b.path)
                if size + extra <= limit and (related or size < limit // 3):
                    cur.append((b.path, piece))
                    shared, size = _shared(shared, b.path), size + extra
                    continue
                flush()
            cur.append((b.path, piece))
            shared, size = b.path, len(piece) + 1
    if cur:
        flush()
    return out
