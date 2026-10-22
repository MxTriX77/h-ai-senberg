"""Tripwire: committed text files are valid UTF-8 without mojibake (UTF-8 read as cp1252 and saved again)."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# U+00C3, or U+00C2 + U+0080-U+00BF, is what a UTF-8 lead byte becomes in cp1252; U+00E2 U+20AC starts quotes, dashes, ellipsis.
MOJIBAKE = re.compile("\u00c3|\u00c2[\u0080-\u00bf]|\u00e2\u20ac")


def text_files() -> list[Path]:
    files = [*REPO.glob("config/*.yaml"), REPO / "README.md", REPO / ".env.example"]
    for root in ("src", "ingest", "tests", "evals", "scripts"):
        files += [p for p in (REPO / root).rglob("*") if p.is_file() and p.suffix in {".py", ".yaml", ".md", ".sh", ".tsv"}]
    return [p for p in files if p.exists() and "fixtures" not in p.relative_to(REPO).parts]


def test_text_files_are_utf8_without_mojibake():
    bad = {}
    for p in text_files():
        try:
            text = p.read_bytes().decode("utf-8")
        except UnicodeDecodeError as e:
            bad[str(p.relative_to(REPO))] = f"not UTF-8: {e}"
            continue
        if hits := [f"line {text.count(chr(10), 0, m.start()) + 1}: {m.group()!r}" for m in MOJIBAKE.finditer(text)]:
            bad[str(p.relative_to(REPO))] = hits
    assert not bad, bad


def test_tripwire_catches_double_encoding():
    assert MOJIBAKE.search("78.2 °C".encode("utf-8").decode("cp1252"))
    assert MOJIBAKE.search("…".encode("utf-8").decode("cp1252"))
    assert not MOJIBAKE.search("78.2 °C, 10^−34, Å, é, …")
