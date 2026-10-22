"""Tripwire: model IDs, endpoints and prompts belong in config/config.yaml, not in code."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOTS = [REPO / "src" / "haisenberg", REPO / "ingest"]
PATTERNS = [r"https?://[\w.-]", r"\.gguf", r"qwen", r"you are an? "]


def offences(text: str) -> list[str]:
    return [p for p in PATTERNS if re.search(p, text, re.I)]


def test_source_has_no_hardcoded_policy():
    found = {
        str(f.relative_to(REPO)): hits
        for root in ROOTS
        for f in root.rglob("*.py")
        if (hits := offences(f.read_text(encoding="utf-8")))
    }
    assert not found, found


def test_tripwire_catches_a_planted_endpoint():
    assert offences('URL = "http://127.0.0.1:8080"')
    assert offences('PROMPT = "You are a helpful assistant"')
