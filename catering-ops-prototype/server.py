"""Serve the catering ops prototype with history fallback."""

from __future__ import annotations

import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PORT = 8765


class AppHandler(SimpleHTTPRequestHandler):
    """Static file handler that sends app routes to index.html."""

    def translate_path(self, path: str) -> str:
        raw_path = super().translate_path(path)
        requested = Path(raw_path)
        if requested.exists():
            return str(requested)
        return str(ROOT / "index.html")


if __name__ == "__main__":
    os.chdir(ROOT)
    server = ThreadingHTTPServer(("127.0.0.1", PORT), AppHandler)
    print(f"Catering ops prototype running at http://127.0.0.1:{PORT}/today")
    server.serve_forever()
