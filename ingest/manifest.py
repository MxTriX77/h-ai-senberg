"""Load and validate `config/sources.yaml`, the list of raw source files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

import yaml

USES = {
    "doc",
    "codata",
    "pubchem-mass",
    "pubchem-mesh",
    "pubchem-synonyms",
    "pubchem-title",
    "wikipedia-zim",
    "wikipedia-mini-zim",
    "libretexts-zim",
}
CHECKSUM_FIELDS = ("sha256", "md5", "sha256_url", "md5_url")
HEX_LEN = {"sha256": 64, "md5": 32}
FIELDS = {"name", "use", "layer", "domains", "url", "dest", "license", "cite_url", "filter", *CHECKSUM_FIELDS}
REQUIRED = ("name", "use", "layer", "domains", "url", "dest", "license")
FILTER_USE = "wikipedia-mini-zim"  # the only kind whose entry carries a relevance filter
FILTER_FIELDS = {"infoboxes", "projects", "importance"}
_DOMAIN = re.compile(r"^[a-z][a-z0-9_-]*$")


class ManifestError(Exception):
    """The manifest is invalid; the message names the entry and the field."""


@dataclass(frozen=True)
class MiniFilter:
    """Which all-articles mini ZIM pages fill gaps, and the domain each match gives the article."""

    infoboxes: dict[str, str]  # infobox class (e.g. ib-chembox) -> domain
    projects: dict[str, str]  # WikiProject name -> domain
    importance: tuple[str, ...]  # WikiProject ratings that count (e.g. Top, High, Mid)


@dataclass(frozen=True)
class Entry:
    name: str
    use: str
    layer: int
    url: str
    dest: str
    license: str
    cite_url: str | None
    algo: str  # "sha256" or "md5"
    pinned: str | None  # pinned hex digest, or None for a checksum-URL entry
    checksum_url: str | None
    domains: tuple[str, ...] = ()
    filter: MiniFilter | None = None

    def path(self, raw_dir: Path) -> Path:
        return raw_dir / self.dest


def load_manifest(path: Path | str) -> list[Entry]:
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as e:
        raise ManifestError(f"{path}: cannot read manifest: {e}") from e
    items = data.get("sources") if isinstance(data, dict) else None
    if not isinstance(items, list) or not items:
        raise ManifestError(f"{path}: expected a non-empty 'sources' list")

    entries: list[Entry] = []
    seen: set[str] = set()
    for i, item in enumerate(items):
        label = item.get("name", f"#{i}") if isinstance(item, dict) else f"#{i}"
        if not isinstance(item, dict):
            raise ManifestError(f"entry {label}: must be a mapping")
        entry = _entry(item, label)
        if entry.name in seen:
            raise ManifestError(f"entry {label}: duplicate name")
        seen.add(entry.name)
        entries.append(entry)
    return entries


def _entry(item: dict, label: str) -> Entry:
    def bad(field: str, why: str) -> ManifestError:
        return ManifestError(f"entry {label}: field '{field}' {why}")

    for field in item:
        if field not in FIELDS:
            raise bad(field, "is unknown")
    for field in REQUIRED:
        if item.get(field) in (None, ""):
            raise bad(field, "is required")
    for field in ("name", "use", "url", "dest", "license"):
        if not isinstance(item[field], str):
            raise bad(field, "must be a string")
    if item["use"] not in USES:
        raise bad("use", f"must be one of {sorted(USES)}")
    if item["layer"] not in (1, 2):
        raise bad("layer", "must be 1 or 2")
    if not re.match(r"^https?://", item["url"]):
        raise bad("url", "must be an http(s) URL")
    dest = item["dest"]
    if (
        PurePosixPath(dest).is_absolute()
        or PureWindowsPath(dest).is_absolute()
        or PureWindowsPath(dest).drive
        or ".." in re.split(r"[\\/]", dest)
    ):
        raise bad("dest", "must be a relative path inside the raw directory")

    given = [f for f in CHECKSUM_FIELDS if item.get(f)]
    if len(given) != 1:
        raise bad("checksum", f"needs exactly one of {list(CHECKSUM_FIELDS)}, got {given or 'none'}")
    field = given[0]
    algo = field.removesuffix("_url")
    pinned = checksum_url = None
    if field.endswith("_url"):
        checksum_url = item[field]
        if not isinstance(checksum_url, str) or not re.match(r"^https?://", checksum_url):
            raise bad(field, "must be an http(s) URL")
    else:
        pinned = item[field]
        if not isinstance(pinned, str) or not re.fullmatch(rf"[0-9a-f]{{{HEX_LEN[algo]}}}", pinned):
            raise bad(field, f"must be {HEX_LEN[algo]} lowercase hex digits")

    domains = item["domains"]
    if not isinstance(domains, list) or not domains or not all(isinstance(d, str) and _DOMAIN.match(d) for d in domains):
        raise bad("domains", "must be a non-empty list of lowercase domain names")
    mini_filter = _filter(item, domains, bad)

    cite_url = item.get("cite_url")
    if cite_url is not None and not isinstance(cite_url, str):
        raise bad("cite_url", "must be a string")
    return Entry(
        name=item["name"],
        use=item["use"],
        layer=item["layer"],
        url=item["url"],
        dest=dest,
        license=item["license"],
        cite_url=cite_url,
        algo=algo,
        pinned=pinned,
        checksum_url=checksum_url,
        domains=tuple(domains),
        filter=mini_filter,
    )


def _filter(item: dict, domains: list[str], bad) -> MiniFilter | None:
    raw = item.get("filter")
    if item["use"] != FILTER_USE:
        if raw is not None:
            raise bad("filter", f"is only allowed for use '{FILTER_USE}'")
        return None
    if not isinstance(raw, dict) or set(raw) != FILTER_FIELDS:
        raise bad("filter", f"is required for use '{FILTER_USE}' with exactly {sorted(FILTER_FIELDS)}")
    maps = {}
    for key in ("infoboxes", "projects"):
        m = raw[key]
        if not isinstance(m, dict) or not all(isinstance(k, str) and k for k in m):
            raise bad("filter", f"'{key}' must map names to domains")
        if undeclared := sorted(str(d) for d in m.values() if d not in domains):
            raise bad("filter", f"'{key}' names domains not in the entry's domains: {undeclared}")
        maps[key] = dict(m)
    importance = raw["importance"]
    if not isinstance(importance, list) or not all(isinstance(r, str) and r for r in importance):
        raise bad("filter", "'importance' must be a list of ratings")
    return MiniFilter(maps["infoboxes"], maps["projects"], tuple(importance))
