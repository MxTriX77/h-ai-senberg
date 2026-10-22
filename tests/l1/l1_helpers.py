"""Helpers shared by the Layer 1 tests (fixture locations, fixture manifest, real-data paths)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "l1"
REPO = Path(__file__).resolve().parents[2]

# Manifest entries for the fixture raw directory (URLs are never fetched by the build).
FIXTURE_SOURCES = [
    {
        "name": "codata_fixture",
        "use": "codata",
        "dest": "codata/allascii.txt",
        "license": "NIST, US government work (public domain)",
        "cite_url": "https://physics.nist.gov/cgi-bin/cuu/Results?search_for={name}",
    },
    *(
        {
            "name": f"pubchem_{use.split('-')[1]}_fixture",
            "use": use,
            "dest": f"pubchem/{dest}",
            "license": "NCBI public data",
            "cite_url": "https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
        }
        for use, dest in [
            ("pubchem-mesh", "CID-MeSH"),
            ("pubchem-mass", "CID-Mass.gz"),
            ("pubchem-title", "CID-Title.gz"),
            ("pubchem-synonyms", "CID-Synonym-filtered.gz"),
        ]
    ),
]


def write_manifest(path: Path, sources=None) -> Path:
    entries = [
        {"layer": 1, "url": f"https://example.org/{s['dest']}", "sha256": "0" * 64, **s}
        for s in (FIXTURE_SOURCES if sources is None else sources)
    ]
    path.write_text(yaml.safe_dump({"sources": entries}), encoding="utf-8")
    return path


def real_path(attr: str) -> Path:
    """retrieval.<attr> from the real config (honours DATA_DIR); skips the test if it is absent."""
    from haisenberg.config import load_config

    p = getattr(load_config(REPO / "config" / "config.yaml").retrieval, attr)
    if not p.exists():
        pytest.skip(f"real data not present: {p}")
    return p
