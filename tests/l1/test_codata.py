import pytest

from ingest.codata import parse
from ingest.l1base import BuildError

from .l1_helpers import FIXTURES, real_path


def by_name(path):
    year, rows = parse(path)
    return year, {r.name: r for r in rows}


def test_exact_truncated_constant():
    year, rows = by_name(FIXTURES / "raw/codata/allascii.txt")
    r = rows["molar gas constant"]
    assert (r.value, r.uncertainty, r.exact, r.truncated, r.unit) == (8.314462618, 0.0, True, True, "J mol^-1 K^-1")
    assert r.value_text == "8.314 462 618..."
    assert year == "2022"


def test_measured_constant():
    _, rows = by_name(FIXTURES / "raw/codata/allascii.txt")
    r = rows["electron mass"]
    assert r.value == pytest.approx(9.1093837139e-31, rel=1e-15)
    assert r.uncertainty == pytest.approx(2.8e-40, rel=1e-12)
    assert (r.exact, r.truncated, r.unit) == (False, False, "kg")


def test_dimensionless_constant_has_empty_unit():
    _, rows = by_name(FIXTURES / "raw/codata/allascii.txt")
    assert rows["fine-structure constant"].unit == ""


def test_malformed_row_fails(tmp_path):
    text = (FIXTURES / "raw/codata/allascii.txt").read_text(encoding="utf-8")
    bad = tmp_path / "allascii.txt"
    bad.write_text(text + "broken constant" + " " * 45 + "not a number\n", encoding="utf-8")
    with pytest.raises(BuildError, match="cannot parse"):
        parse(bad)


def test_no_rows_fails(tmp_path):
    text = (FIXTURES / "raw/codata/allascii.txt").read_text(encoding="utf-8")
    empty = tmp_path / "allascii.txt"
    empty.write_text(text.split("-----")[0] + "-" * 20 + "\n", encoding="utf-8")
    with pytest.raises(BuildError, match="no constants"):
        parse(empty)


def test_real_listing_has_355_constants():
    year, rows = parse(real_path("raw_dir") / "codata" / "allascii.txt")
    assert (year, len(rows), len({r.name for r in rows})) == ("2022", 355, 355)
