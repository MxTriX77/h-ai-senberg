"""Layer 2 store schema, shared by the build (ingest/build_l2.py) and the query side."""

SCHEMA_VERSION = 1  # stored as PRAGMA user_version; bump on any change below
APPLICATION_ID = 0x4C495431  # 'LIT1', PRAGMA application_id: tells this store apart from the Layer 1 file

DDL = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE source(
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,        -- manifest entry name
  use TEXT NOT NULL,                -- manifest use tag
  file TEXT NOT NULL,               -- path relative to the raw directory
  sha256 TEXT NOT NULL,             -- computed by the build
  version TEXT NOT NULL,            -- ZIM Date or ChEBI release
  license TEXT NOT NULL,
  cite_url TEXT NOT NULL,           -- citation template from the manifest
  domains TEXT NOT NULL,            -- the manifest entry's domains, space-separated
  title TEXT NOT NULL               -- display name, e.g. 'LibreTexts Chemistry'
);

CREATE TABLE doc(
  id INTEGER PRIMARY KEY,
  source_id INTEGER NOT NULL REFERENCES source(id),
  path TEXT NOT NULL,               -- page path in the source: ZIM path, LibreTexts path, a/<answer id>, CHEBI:<n>
  title TEXT NOT NULL,
  revision INTEGER                  -- Wikipedia/Wikibooks revision ID when the page states one
);

CREATE TABLE passage(
  id INTEGER PRIMARY KEY,
  doc_id INTEGER NOT NULL REFERENCES doc(id),
  title TEXT NOT NULL,              -- the document's title (FTS column)
  heading TEXT NOT NULL,            -- headings above the passage inside the page, ' › '-separated
  text TEXT NOT NULL,
  domains TEXT NOT NULL             -- space-separated (FTS column, weight 0: used as a filter)
);
CREATE INDEX passage_doc ON passage(doc_id);

CREATE VIRTUAL TABLE passage_fts USING fts5(
  title, heading, text, domains,
  content='passage', content_rowid='id',
  tokenize='porter unicode61 remove_diacritics 2'
);
"""
