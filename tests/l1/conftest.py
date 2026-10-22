"""Shared fixtures for the Layer 1 tests."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


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
