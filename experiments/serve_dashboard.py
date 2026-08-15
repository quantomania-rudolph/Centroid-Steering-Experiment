#!/usr/bin/env python
"""Serve the J-lens dashboard over HTTP and open it in the browser."""

from __future__ import annotations

import argparse
import http.server
import socket
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASH_DIR = ROOT / "experiments" / "dashboard"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=0, help="Port (0 = auto)")
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild dashboard first")
    args = parser.parse_args()

    if args.rebuild:
        build = ROOT / "experiments" / "build_dashboard.py"
        rc = subprocess.call([sys.executable, str(build)], cwd=str(ROOT))
        if rc != 0:
            return rc

    if not (DASH_DIR / "index.html").is_file():
        print("Missing dashboard — run: python experiments/build_dashboard.py", file=sys.stderr)
        return 1

    port = args.port or _free_port()
    handler = http.server.SimpleHTTPRequestHandler

    class QuietHandler(handler):
        def log_message(self, fmt: str, *args) -> None:
            if args and str(args[1]) != "200":
                super().log_message(fmt, *args)

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), QuietHandler)
    httpd.daemon_threads = True

    import os

    os.chdir(DASH_DIR)
    url = f"http://127.0.0.1:{port}/index.html"
    print(f"Serving {DASH_DIR}")
    print(f"Open: {url}")
    if not args.no_open:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
