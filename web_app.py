from __future__ import annotations

import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from mc_quadrants.api import (
    build_compare_response,
    build_load_response,
    build_simulate_response,
    build_wealth_csv,
    load_data_source,
)

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
PORT = int(os.getenv("PORT", "7860"))


class WebHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: str) -> None:
        relative = path.lstrip("/")
        if relative == "":
            relative = "index.html"
        file_path = os.path.normpath(os.path.join(WEB_DIR, relative))
        if not file_path.startswith(WEB_DIR) or not os.path.isfile(file_path):
            self._send_json(404, {"ok": False, "error": f"Not found: {path}"})
            return
        content_type, _ = mimetypes.guess_type(file_path)
        with open(file_path, "rb") as handle:
            body = handle.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send_json(200, {"ok": True, "app": "mc-quadrant-sim web UI"})
            return
        self._send_file(path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "Invalid JSON body."})
            return
        try:
            if path == "/api/load":
                macro, returns, tickers, growth_col, inflation_col, message = load_data_source(payload)
                self._send_json(200, build_load_response(macro, returns, tickers, growth_col, inflation_col, message))
            elif path == "/api/simulate":
                self._send_json(200, build_simulate_response(payload))
            elif path == "/api/compare":
                self._send_json(200, build_compare_response(payload))
            elif path == "/api/wealth":
                self._send_json(200, build_wealth_csv(payload))
            else:
                self._send_json(404, {"ok": False, "error": f"Unknown endpoint: {path}"})
        except Exception as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", PORT), WebHandler)
    server.daemon_threads = True
    print(f"MC Quadrant Simulator web UI running at http://127.0.0.1:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
