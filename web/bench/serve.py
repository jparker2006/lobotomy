"""Render bench server (U1, issue #19). Standard library only.

Serves web/dist (the T2 page at /, the bench at /bench/) and saves each POST /api/results to
web/bench/results/<utc>-<browser>.json.

Run:  cd web && npm run bench            (build, then serve on 127.0.0.1:8019)
      python web/bench/serve.py --port 8019
"""
import argparse
import json
import re
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIST = HERE.parent / "dist"
RESULTS = HERE / "results"


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript", ".json": "application/json", ".html": "text/html; charset=utf-8"}

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(DIST), **kw)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        if self.path in ("/bench", "/b"):
            self.send_response(302)
            self.send_header("Location", "/bench/")
            self.end_headers()
            return
        super().do_GET()

    def do_POST(self):
        if self.path != "/api/results":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length") or 0)
        if n > 2_000_000:
            self.send_error(413)
            return
        try:
            data = json.loads(self.rfile.read(n))
        except ValueError:
            self.send_error(400, "bad json")
            return
        browser = re.sub(r"[^a-z0-9]+", "-", str(data.get("browser", "unknown")).lower())[:20] or "unknown"
        RESULTS.mkdir(exist_ok=True)
        path = RESULTS / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{browser}.json"
        path.write_text(json.dumps(data, indent=1) + "\n")
        print(f"saved {path.relative_to(HERE.parent.parent)}: {data.get('verdict')} {data.get('backend')}", flush=True)
        body = json.dumps({"saved": str(path.relative_to(HERE.parent.parent))}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8019)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    if not (DIST / "bench" / "index.html").exists():
        raise SystemExit("web/dist/bench is missing: run `npm run build` in web/ first")
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"bench: http://{a.host}:{a.port}/bench/   results -> {RESULTS}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
