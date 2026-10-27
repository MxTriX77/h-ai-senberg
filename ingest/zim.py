"""Read a ZIM file's directory directly, to visit articles in cluster order.

python-libzim has no cluster-order iterator, and reading entries in path order makes libzim
decompress the same clusters again and again (physics ZIM: 113 s in path order, 7 s in cluster
order). The ZIM header and directory entries are a small binary format documented by openZIM
("ZIM file format"), so we read them here and fetch content with libzim.
"""

from __future__ import annotations

import mmap
import struct
from collections.abc import Iterator
from pathlib import Path

from libzim.reader import Archive

MAGIC = 72173914
REDIRECT = 0xFFFF
ARTICLE_NS = b"C"


class Zim:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._file = self.path.open("rb")
        self._m = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        magic, _major, _minor = struct.unpack_from("<IHH", self._m, 0)
        if magic != MAGIC:
            raise ValueError(f"{self.path}: not a ZIM file")
        self.entry_count, _clusters, path_pos, _title_pos, _cluster_pos, mime_pos = struct.unpack_from(
            "<IIQQQQ", self._m, 24
        )
        self._ptrs = memoryview(self._m)[path_pos : path_pos + 8 * self.entry_count].cast("Q")
        self.mimetypes: list[str] = []
        pos = mime_pos
        while (end := self._m.find(b"\0", pos)) != pos:
            self.mimetypes.append(self._m[pos:end].decode())
            pos = end + 1
        self.archive = Archive(self.path)
        meta = set(self.archive.metadata_keys)
        self.metadata = {
            k: self.archive.get_metadata(k).decode("utf-8", "replace") for k in ("Date", "Name", "Title") if k in meta
        }

    def close(self) -> None:
        self._ptrs.release()
        self._m.close()
        self._file.close()

    def __enter__(self) -> Zim:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def html_by_cluster(self) -> list[int]:
        """Entry indices of the HTML entries in the article namespace, sorted by (cluster, blob)."""
        html = {i for i, m in enumerate(self.mimetypes) if m.startswith("text/html")}
        found = []
        unpack, m = struct.unpack_from, self._m
        for i, off in enumerate(self._ptrs):
            mime, _, ns, _, cluster, blob = unpack("<HBcIII", m, off)
            if ns == ARTICLE_NS and mime in html:
                found.append((cluster, blob, i))
        found.sort()
        return [i for _, _, i in found]

    def by_cluster(self, ids: list[int]) -> list[int]:
        """The given entry indices sorted by (cluster, blob), the order that decompresses each cluster once."""
        unpack, m = struct.unpack_from, self._m
        return sorted(ids, key=lambda i: unpack("<II", m, self._ptrs[i] + 8))

    def redirects(self) -> Iterator[tuple[int, int]]:
        """(entry index, target index) for every redirect in the article namespace."""
        unpack, m = struct.unpack_from, self._m
        for i, off in enumerate(self._ptrs):
            mime, _, ns = unpack("<HBc", m, off)
            if mime == REDIRECT and ns == ARTICLE_NS:
                yield i, unpack("<I", m, off + 8)[0]

    def strings(self, i: int) -> tuple[str, str]:
        """(path, title) of entry i; the title defaults to the path."""
        off = self._ptrs[i]
        start = off + (12 if struct.unpack_from("<H", self._m, off)[0] == REDIRECT else 16)
        end = self._m.find(b"\0", start)
        tend = self._m.find(b"\0", end + 1)
        path = self._m[start:end].decode("utf-8", "replace")
        return path, (self._m[end + 1 : tend].decode("utf-8", "replace") or path)

    def find(self, path: str) -> int | None:
        """Entry index of an article-namespace path (binary search on the path-ordered directory)."""
        key = ARTICLE_NS + path.encode("utf-8")
        lo, hi = 0, self.entry_count
        while lo < hi:
            mid = (lo + hi) // 2
            off = self._ptrs[mid]
            start = off + (12 if struct.unpack_from("<H", self._m, off)[0] == REDIRECT else 16)
            here = self._m[off + 3 : off + 4] + self._m[start : self._m.find(b"\0", start)]
            if here < key:
                lo = mid + 1
            else:
                hi = mid
        if lo < self.entry_count and self.strings(lo)[0] == path:
            off = self._ptrs[lo]
            if self._m[off + 3 : off + 4] == ARTICLE_NS:
                return lo
        return None

    def content(self, i: int) -> bytes:
        return bytes(self.archive._get_entry_by_id(i).get_item().content)
