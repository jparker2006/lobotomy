"""Record one generation from a running lobotomy server into a replay file.

Usage: python web/mockups/tools/record.py "prompt" out.bin [--url http://localhost:8000]
The file is the raw T2 frame stream (see server/server.py), so pages parse it like a live one.
Also writes calibration.f32 and meta.json next to it.
"""
import argparse
import json
import struct
import urllib.request
from pathlib import Path


def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prompt")
    p.add_argument("out")
    p.add_argument("--url", default="http://localhost:8000")
    a = p.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / "meta.json").write_bytes(get(a.url + "/api/meta"))
    (out.parent / "calibration.f32").write_bytes(get(a.url + "/api/calibration"))
    req = urllib.request.Request(a.url + "/api/generate", json.dumps({"prompt": a.prompt}).encode(),
                                 {"Content-Type": "application/json"})
    gid = json.loads(urllib.request.urlopen(req, timeout=30).read())["id"]
    buf, n, done = b"", 0, False
    while not done:
        chunk = get(f"{a.url}/api/frames?id={gid}&from={n}")
        o = 0
        while o < len(chunk):
            (hl,) = struct.unpack_from("<I", chunk, o)
            h = json.loads(chunk[o + 4:o + 4 + hl])
            o += 4 + hl + h["n"]
            n += 1
            done = done or h["kind"] == "done"
        buf += chunk
    out.write_bytes(buf)
    print(f"{out}: {n} frames, {len(buf) / 1e6:.1f} MB, last header {h}")


if __name__ == "__main__":
    main()
