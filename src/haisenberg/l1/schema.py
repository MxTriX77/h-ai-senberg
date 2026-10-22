"""Layer 1 store schema, shared by the build (ingest/build_l1.py) and the query side."""

SCHEMA_VERSION = 2  # stored as PRAGMA user_version; bump on any change below

DDL = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE source(
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,        -- manifest entry name
  use TEXT NOT NULL,                -- manifest use tag
  file TEXT NOT NULL,               -- path relative to the raw directory
  sha256 TEXT NOT NULL,             -- computed by the build
  version TEXT NOT NULL,            -- ZIM Date, CODATA year, or download date
  license TEXT NOT NULL,
  cite_url TEXT NOT NULL,           -- citation template from the manifest
  domains TEXT NOT NULL             -- the manifest entry's domains, space-separated
);

CREATE TABLE article(
  id INTEGER PRIMARY KEY,
  path TEXT NOT NULL UNIQUE,        -- ZIM entry path, e.g. 'Planck_constant'
  title TEXT NOT NULL,
  aliases TEXT NOT NULL,            -- redirect titles, newline-separated (FTS column)
  infobox TEXT NOT NULL,            -- 'label: value' lines
  lead TEXT NOT NULL,
  revision INTEGER,                 -- Wikipedia revision ID
  source_id INTEGER NOT NULL REFERENCES source(id),  -- ZIM of the copy kept
  zims TEXT NOT NULL                -- every ZIM source name containing it, space-separated
);

CREATE TABLE article_alias(          -- normalized titles and redirect names
  alias TEXT NOT NULL,
  article_id INTEGER NOT NULL REFERENCES article(id),
  PRIMARY KEY(alias, article_id)
) WITHOUT ROWID;

CREATE TABLE article_domain(         -- an article's domains (topic ZIM domains, or the mini filter's match)
  article_id INTEGER NOT NULL REFERENCES article(id),
  domain TEXT NOT NULL,
  PRIMARY KEY(article_id, domain)
) WITHOUT ROWID;
CREATE INDEX article_domain_by_domain ON article_domain(domain, article_id);

CREATE VIRTUAL TABLE article_fts USING fts5(
  title, aliases, infobox, lead,
  content='article', content_rowid='id',
  tokenize='porter unicode61 remove_diacritics 2'
);

CREATE TABLE constant(
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE COLLATE NOCASE,
  value REAL NOT NULL,
  uncertainty REAL NOT NULL,        -- standard uncertainty, 0 when exact
  unit TEXT NOT NULL,               -- '' when dimensionless
  value_text TEXT NOT NULL,         -- as printed by NIST
  exact INTEGER NOT NULL,
  truncated INTEGER NOT NULL,       -- printed digits end with '...'
  source_id INTEGER NOT NULL REFERENCES source(id)
);

CREATE TABLE compound(
  cid INTEGER PRIMARY KEY,
  title TEXT NOT NULL,
  formula TEXT NOT NULL,            -- PubChem Hill formula, charge suffix included
  monoisotopic_mass REAL NOT NULL,
  molar_mass REAL,                  -- g/mol from standard atomic weights; NULL with mass_note
  mass_note TEXT,
  synonyms TEXT NOT NULL,           -- first 10, '|'-separated
  selected_by TEXT NOT NULL CHECK (selected_by IN ('mesh', 'wikipedia', 'both')),
  article_id INTEGER REFERENCES article(id)
);
CREATE INDEX compound_formula ON compound(formula);

CREATE TABLE compound_name(
  name TEXT NOT NULL,               -- normalized
  cid INTEGER NOT NULL REFERENCES compound(cid),
  rank INTEGER NOT NULL,            -- 0 = title, 1..20 = synonym order
  PRIMARY KEY(name, cid)
) WITHOUT ROWID;
"""
