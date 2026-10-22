"""Shared fixtures for the Layer 1 tests."""

from __future__ import annotations

import gzip
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from .l1_helpers import FIXTURES, write_manifest
from .wiki_fixtures import make_zims


class FileServer:
    """Serves `files` (path -> bytes) on 127.0.0.1 with Range support; records every request."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.requests: list[tuple[str, str | None]] = []  # (path, Range header)
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                server.requests.append((self.path, self.headers.get("Range")))
                body = server.files.get(self.path)
                if body is None:
                    self.send_error(404)
                    return
                rng = self.headers.get("Range")
                if rng:
                    start = int(rng.removeprefix("bytes=").split("-")[0])
                    if start >= len(body):
                        self.send_error(416)
                        return
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
                    body = body[start:]
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def gets(self, path: str) -> int:
        return sum(1 for p, _ in self.requests if p == path)

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def file_server():
    srv = FileServer()
    yield srv
    srv.close()


# --- Layer 1 store fixtures -------------------------------------------------------------------


def make_raw(root: Path) -> Path:
    """A writable copy of tests/fixtures/l1/raw; PubChem *.txt extracts are gzipped like the real files."""
    raw = root / "raw"
    shutil.copytree(FIXTURES / "raw", raw)
    for txt in (raw / "pubchem").glob("*.txt"):
        txt.with_suffix(".gz").write_bytes(gzip.compress(txt.read_bytes()))
        txt.unlink()
    make_zims(raw)
    return raw


@pytest.fixture
def fixture_raw(tmp_path) -> Path:
    return make_raw(tmp_path)


@pytest.fixture
def fixture_manifest(tmp_path) -> Path:
    return write_manifest(tmp_path / "sources.yaml")


@pytest.fixture
def fresh_store(fixture_raw, fixture_manifest, tmp_path) -> Path:
    """A store built from the fixtures for this test only (tests that modify it)."""
    from ingest.build_l1 import build

    db = tmp_path / "l1.db"
    build(fixture_manifest, fixture_raw, db)
    return db


@pytest.fixture(scope="session")
def built_store(tmp_path_factory) -> Path:
    """One store built from the fixtures, shared read-only by the tests."""
    from ingest.build_l1 import build

    root = tmp_path_factory.mktemp("l1store")
    db = root / "l1.db"
    build(write_manifest(root / "sources.yaml"), make_raw(root), db)
    return db
