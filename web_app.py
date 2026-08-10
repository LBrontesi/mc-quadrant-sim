from __future__ import annotations

import json
import logging
import mimetypes
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

# Allow `python web_app.py` to work directly from a source checkout.
PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mc_quadrants.api import (  # noqa: E402
    build_compare_response,
    build_load_response,
    build_simulate_response,
    build_wealth_csv,
    load_data_source,
)

WEB_DIR = str((PROJECT_DIR / "web").resolve())
PORT = int(os.getenv("PORT", "7860"))
MAX_REQUEST_BYTES = int(os.getenv("MAX_REQUEST_BYTES", str(10 * 1024 * 1024)))
MAX_CONCURRENT_JOBS = max(1, int(os.getenv("MAX_CONCURRENT_JOBS", "1")))
HEAVY_ENDPOINTS = {"/api/simulate", "/api/compare", "/api/wealth"}
JOB_SEMAPHORE = threading.BoundedSemaphore(MAX_CONCURRENT_JOBS)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("mc_quadrants.web")


class WebHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), format % args)

    def _send_security_headers(self, cache_control: str) -> None:
        self.send_header("Cache-Control", cache_control)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'",
        )

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._send_security_headers("no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: str) -> None:
        relative = unquote(path).lstrip("/")
        if relative == "":
            relative = "index.html"
        web_root = Path(WEB_DIR)
        file_path = (web_root / relative).resolve()
        if not file_path.is_relative_to(web_root) or not file_path.is_file():
            self._send_json(404, {"ok": False, "error": f"Not found: {path}"})
            return
        content_type, _ = mimetypes.guess_type(str(file_path))
        with file_path.open("rb") as handle:
            body = handle.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        cache_control = (
            "no-cache"
            if file_path.suffix.lower() in {".html", ".css", ".js"}
            else "public, max-age=3600"
        )
        self._send_security_headers(cache_control)
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
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"ok": False, "error": "Invalid Content-Length header."})
            return
        if length < 0:
            self._send_json(400, {"ok": False, "error": "Invalid Content-Length header."})
            return
        if length > MAX_REQUEST_BYTES:
            self._send_json(
                413,
                {
                    "ok": False,
                    "error": f"Request body exceeds the {MAX_REQUEST_BYTES // 1024**2} MB limit.",
                },
            )
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(400, {"ok": False, "error": "Invalid JSON body."})
            return
        if not isinstance(payload, dict):
            self._send_json(400, {"ok": False, "error": "JSON body must be an object."})
            return

        acquired = path not in HEAVY_ENDPOINTS or JOB_SEMAPHORE.acquire(blocking=False)
        if not acquired:
            self._send_json(
                429,
                {"ok": False, "error": "Another simulation is running. Try again when it finishes."},
            )
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
        except ValueError as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
        except MemoryError:
            LOGGER.exception("Simulation could not be completed by the server")
            self._send_json(503, {"ok": False, "error": "The server could not complete this analysis."})
        except Exception:
            LOGGER.exception("Unhandled request failure for %s", path)
            self._send_json(500, {"ok": False, "error": "Unexpected server error."})
        finally:
            if path in HEAVY_ENDPOINTS:
                JOB_SEMAPHORE.release()


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", PORT), WebHandler)
    server.daemon_threads = True
    LOGGER.info(
        "MC Quadrant Simulator listening on port %s (max jobs=%s, max request=%s bytes)",
        PORT,
        MAX_CONCURRENT_JOBS,
        MAX_REQUEST_BYTES,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
