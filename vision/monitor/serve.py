"""Run:  .venv/bin/python -m vision.monitor.serve  [--port 8600] [--host 127.0.0.1]

Routes: /            the board
        /api/status  JSON snapshot (memoised 1 s)
        /img/<name>  label | track | court | numbers  as jpg, 404 with "noch nichts" if missing
        /qa/<file>   read-only static files from out/qa/ (the verification sheets)
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from vision.monitor import images, page, status

STATIC_TYPES = {".html": "text/html; charset=utf-8", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".json": "application/json; charset=utf-8",
                ".txt": "text/plain; charset=utf-8", ".csv": "text/csv; charset=utf-8", ".svg": "image/svg+xml"}


class Handler(BaseHTTPRequestHandler):
    server_version = "followcam-monitor/1"

    def _send(self, code: int, ctype: str, body: bytes, cache: str = "no-store") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        try:
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", page.HTML.encode("utf-8"))
            elif path == "/favicon.ico":
                self._send(204, "image/x-icon", b"")
            elif path == "/api/status":
                body = json.dumps(status.collect(), ensure_ascii=False, default=str).encode("utf-8")
                self._send(200, "application/json; charset=utf-8", body)
            elif path.startswith("/img/"):
                data = images.get(path[5:])
                if data:
                    self._send(200, "image/jpeg", data, cache="max-age=3600")
                else:
                    self._send(404, "text/plain; charset=utf-8", "noch nichts".encode("utf-8"))
            elif path.startswith("/qa/"):
                self._serve_qa(path[4:] or "index.html")
            else:
                self._send(404, "text/plain; charset=utf-8", b"404")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # noqa: BLE001 - keep serving no matter what
            try:
                self._send(500, "application/json; charset=utf-8", json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode("utf-8"))
            except Exception:  # noqa: BLE001
                pass

    def _serve_qa(self, rel: str) -> None:
        root = status.QA_DIR.resolve()
        target = (root / rel).resolve()
        if root not in target.parents and target != root:
            self._send(403, "text/plain; charset=utf-8", b"403")
            return
        if target.is_dir():
            target = target / "index.html"
        ctype = STATIC_TYPES.get(target.suffix.lower())
        if ctype is None or not target.is_file():
            self._send(404, "text/plain; charset=utf-8", "noch nichts".encode("utf-8"))
            return
        self._send(200, ctype, target.read_bytes())

    def log_message(self, fmt, *args):  # quiet: only errors go to stderr
        if args and str(args[0]).startswith(("4", "5")) and not str(self.path).startswith("/img/"):
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="FollowCam pipeline status board")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8600)
    args = ap.parse_args(argv)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    print(f"monitor up: http://{args.host}:{args.port}  (root {status.ROOT})", flush=True)
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
