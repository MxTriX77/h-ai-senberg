from pathlib import Path

import pytest
import yaml

from ingest.manifest import FILTER_FIELDS, BookFilter, ManifestError, MiniFilter, ScoreFilter, load_manifest

from .l1_helpers import MINI_FILTER

REPO = Path(__file__).resolve().parents[2]
SHIPPED = REPO / "config" / "sources.yaml"

# What the one-off bootstrap download put under data/raw (URL -> dest), plus the files added later.
BOOTSTRAP = {
    "https://download.kiwix.org/zim/wikipedia/wikipedia_en_chemistry_nopic_2026-10.zim": "kiwix/wikipedia_en_chemistry_nopic_2026-10.zim",
    "https://download.kiwix.org/zim/wikipedia/wikipedia_en_physics_nopic_2026-10.zim": "kiwix/wikipedia_en_physics_nopic_2026-10.zim",
    "https://download.kiwix.org/zim/libretexts/libretexts.org_en_phys_2026-01.zim": "kiwix/libretexts.org_en_phys_2026-01.zim",
    "https://download.kiwix.org/zim/libretexts/libretexts.org_en_chem_2025-01.zim": "kiwix/libretexts.org_en_chem_2025-01.zim",
    "https://download.kiwix.org/zim/wikipedia/wikipedia_en_all_mini_2026-09.zim": "kiwix/wikipedia_en_all_mini_2026-09.zim",
    "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/README-Extras": "pubchem/README-Extras",
    "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-Mass.gz": "pubchem/CID-Mass.gz",
    "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-Synonym-filtered.gz": "pubchem/CID-Synonym-filtered.gz",
    "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-Title.gz": "pubchem/CID-Title.gz",
    "https://physics.nist.gov/cuu/Constants/Table/allascii.txt": "codata/allascii.txt",
}
EXTRA = {"https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-MeSH": "pubchem/CID-MeSH"}


def entry(**over) -> dict:
    e = {
        "name": "x",
        "use": "codata",
        "layer": 1,
        "domains": ["physics"],
        "url": "https://example.org/x.txt",
        "dest": "codata/x.txt",
        "sha256": "0" * 64,
        "license": "test",
    }
    e.update(over)
    return {k: v for k, v in e.items() if v is not None}


def write(tmp_path: Path, *entries: dict) -> Path:
    p = tmp_path / "sources.yaml"
    p.write_text(yaml.safe_dump({"sources": list(entries)}), encoding="utf-8")
    return p


def test_shipped_manifest_reproduces_bootstrap():
    entries = load_manifest(SHIPPED)
    assert BOOTSTRAP.items() <= {e.url: e.dest for e in entries}.items()
    assert {e.url: e.dest for e in entries if e.url not in BOOTSTRAP and e.layer == 1} == EXTRA  # the rest is Layer 2


def test_shipped_manifest_checksum_kinds():
    by_dest = {e.dest: e for e in load_manifest(SHIPPED)}
    for dest, e in by_dest.items():
        regenerated = dest.startswith("pubchem/CID-")
        assert (e.checksum_url is not None) == regenerated, dest
        assert (e.pinned is not None) != regenerated, dest


def test_valid_entry(tmp_path):
    (e,) = load_manifest(write(tmp_path, entry(md5_url="https://example.org/x.md5", sha256=None)))
    assert (e.algo, e.pinned, e.checksum_url) == ("md5", None, "https://example.org/x.md5")
    assert e.path(tmp_path) == tmp_path / "codata" / "x.txt"


@pytest.mark.parametrize(
    "over, field",
    [
        ({"md5_url": "https://example.org/x.md5"}, "checksum"),  # two checksum sources
        ({"sha256": None}, "checksum"),  # none
        ({"dest": "../outside.txt"}, "dest"),
        ({"dest": "C:/abs.txt"}, "dest"),
        ({"dest": "/abs.txt"}, "dest"),
        ({"layer": 3}, "layer"),
        ({"use": "nope"}, "use"),
        ({"sha256": "ABC"}, "sha256"),
        ({"url": "ftp://example.org/x"}, "url"),
        ({"licence": "typo"}, "licence"),
        ({"license": None}, "license"),
        ({"domains": None}, "domains"),
        ({"domains": []}, "domains"),
        ({"domains": "physics"}, "domains"),
        ({"domains": ["Physics"]}, "domains"),
        ({"use": "embedding-model"}, "domains"),  # a model file has no domains
        ({"filter": MINI_FILTER}, "filter"),  # filter on a codata entry
        ({"use": "wikipedia-mini-zim"}, "filter"),  # mini entry without its filter
        ({"use": "wikipedia-mini-zim", "filter": MINI_FILTER}, "filter"),  # filter names chemistry, entry only physics
        ({"use": "wikipedia-mini-zim", "domains": ["chemistry", "physics"], "filter": {**MINI_FILTER, "importance": "Top"}}, "filter"),
        ({"use": "wikipedia-filtered-zim"}, "filter"),  # filtered ZIM without its filter
        ({"use": "stackexchange-zim"}, "filter"),  # no score threshold
        ({"use": "stackexchange-zim", "filter": {"min_answer_score": -1}}, "filter"),
        ({"use": "stackexchange-zim", "filter": {"min_answer_score": 1.5}}, "filter"),
        ({"use": "wikibooks-zim", "filter": {"books": {"General_Chemistry": "chemistry"}}}, "filter"),  # undeclared domain
        ({"use": "wikibooks-zim", "filter": {"books": ["General_Chemistry"]}}, "filter"),
    ],
)
def test_invalid_entry_names_entry_and_field(tmp_path, over, field):
    with pytest.raises(ManifestError) as err:
        load_manifest(write(tmp_path, entry(name="bad_one", **over)))
    assert "bad_one" in str(err.value) and f"'{field}'" in str(err.value)


def test_embedding_model_entry_without_domains(tmp_path):
    (e,) = load_manifest(write(tmp_path, entry(use="embedding-model", layer=2, domains=None, dest="models/m.gguf")))
    assert (e.use, e.domains, e.filter) == ("embedding-model", (), None)


def test_duplicate_name(tmp_path):
    with pytest.raises(ManifestError, match="duplicate"):
        load_manifest(write(tmp_path, entry(), entry(dest="codata/y.txt")))


def test_shipped_manifest_domains_and_mini_filter():
    by_name = {e.name: e for e in load_manifest(SHIPPED)}
    assert all(bool(e.domains) == (e.use != "embedding-model") for e in by_name.values())
    assert by_name["codata_2022_allascii"].domains == ("physics", "chemistry")
    mini = by_name["wikipedia_en_all_mini_2026-09"].filter
    assert mini == MiniFilter(**{**MINI_FILTER, "importance": tuple(MINI_FILTER["importance"])})
    assert all((e.filter is None) == (e.use not in FILTER_FIELDS) for e in by_name.values())


def test_layer2_filters(tmp_path):
    se, books = load_manifest(write(
        tmp_path,
        entry(name="se", use="stackexchange-zim", layer=2, filter={"min_answer_score": 3}),
        entry(name="wb", use="wikibooks-zim", layer=2, dest="kiwix/wb.zim", filter={"books": {"Physics_Study_Guide": "physics"}}),
    ))
    assert se.filter == ScoreFilter(3)
    assert books.filter == BookFilter({"Physics_Study_Guide": "physics"})
