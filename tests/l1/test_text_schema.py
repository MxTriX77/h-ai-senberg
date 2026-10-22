import sqlite3

from haisenberg.l1.schema import DDL
from haisenberg.l1.text import normalize


def test_normalize():
    assert normalize("  Planck_constant ") == "planck constant"
    assert normalize("Heisenberg   Uncertainty\tPrinciple") == "heisenberg uncertainty principle"
    assert normalize("ＣＯ２") == "co2"  # NFKC folds full-width forms
    assert normalize("Straße") == "strasse"  # casefold, not lower
    assert normalize("71-43-2") == "71-43-2"


def test_schema_creates_with_fts5():
    con = sqlite3.connect(":memory:")
    con.executescript(DDL)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"source", "article", "article_alias", "article_fts", "constant", "compound", "compound_name"} <= tables
