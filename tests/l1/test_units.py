"""Units as records and claims write them (claim-verifier spec, unit-aware code checks)."""

import pytest

from haisenberg.l1.units import parse_unit, read_unit, same_unit


@pytest.mark.parametrize("record, written", [
    ("J Hz^-1", " J s."),
    ("J Hz^-1", " J·s"),
    ("J Hz^-1", " kg m2 s-1"),  # superscripts arrive as plain digits (verifier normalisation)
    ("J Hz^-1", r"\,\mathrm{J\,s}$"),
    ("m s^-1", " m/s"),
    ("m s^-1", " metres per second"),
    ("m^3 kg^-1 s^-2", " N m2 kg-2"),
    ("J mol^-1 K^-1", " J/(mol·K)"),
    ("J mol^-1 K^-1", " joules per mole per kelvin"),
    ("mol^-1", " per mole"),
    ("g/mol", " g mol-1"),
    ("Da", " amu"),
])
def test_equal_spellings(record, written):
    unit, word = read_unit(written)
    assert word and same_unit(unit, parse_unit(record))


@pytest.mark.parametrize("record, written", [
    ("J Hz^-1", " eV s"),
    ("m s^-1", " km per hour"),
    ("m s^-1", " m"),
    ("g/mol", " Da"),
    ("g/mol", " °C"),
    ("", " %"),
])
def test_other_units(record, written):
    unit, _ = read_unit(written)
    assert not same_unit(unit, parse_unit(record))


@pytest.mark.parametrize("written, word", [(" [S1].", False), (", so", False), (" exactly", True), (" g/cup", True)])
def test_no_unit_read(written, word):
    assert read_unit(written) == (None, word)  # a word it does not know, or a unit broken off after `/`


def test_unreadable_record_units():
    assert parse_unit("MeV/c") is None and parse_unit("E_h") is None
    assert parse_unit("") == (1.0, {})
