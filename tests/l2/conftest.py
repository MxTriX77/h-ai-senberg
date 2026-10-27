from pathlib import Path

import pytest

from .fixtures import make_raw, write_manifest

LIMIT = 1200


def build_store(root: Path) -> tuple[Path, dict]:
    from ingest.build_l2 import build

    db = root / "literature.db"
    report = build(write_manifest(root / "sources.yaml"), make_raw(root / "raw"), db, LIMIT)
    return db, report


@pytest.fixture(scope="session")
def l2_built(tmp_path_factory) -> tuple[Path, dict]:
    """One Layer 2 store built from the fixtures, shared read-only by the tests, and its build report."""
    return build_store(tmp_path_factory.mktemp("l2store"))


@pytest.fixture(scope="session")
def l2_store(l2_built) -> Path:
    return l2_built[0]
