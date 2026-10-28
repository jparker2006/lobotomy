"""U3 (#21) stats fixture: the numbers the Cinema UI / Console show, computed with numpy from a recorded session.

  python web/test/fixtures/stats_fixture.py trim REC.bin --keep 0,3-5,15-20 [--out DIR]
      copies frames (by position) of a record.py recording, plus its done frame, into DIR/session.bin (fixture <= 2 MB),
      and REC's calibration.f32 + meta.json into DIR
  python web/test/fixtures/stats_fixture.py expected [--dir DIR]
      replays DIR/session.bin one frame at a time the way the page does (web/src/app/app.ts ingest) and writes
      DIR/expected.json: per token the top-1% threshold, the counter, the per-layer counts and the 6 rarest neurons, plus
      the tape. web/test/stats.test.ts runs the same file through the TypeScript and compares exactly.

The page's math, redone here independently: r = f16(activation) * f32(1 / p99); F = the baked |r| quantiles (parsed from
web/src/data/rarity.ts) as a 512-bin table, weighted as 32 tokens, plus a running histogram of every token at position >= 1;
a token counts against F as it was before that token arrived; top 1% = |r| >= F's 99th percentile.
"""
import argparse
import json
import math
import re
import shutil
import struct
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RARITY_TS = HERE.parents[1] / "src" / "data" / "rarity.ts"
BINS, LOG_LO, LOG_HI = 512, math.log(1e-3), math.log(1e3)
SEED_TOKENS, PER_TOKEN, TOP1_P, RAREST = 32, 86016, 0.99, 6


def read_frames(buf):
    out, o = [], 0
    while o < len(buf):
        (hl,) = struct.unpack_from("<I", buf, o)
        h = json.loads(buf[o + 4:o + 4 + hl])
        out.append((h, buf[o:o + 4 + hl + h["n"]], buf[o + 4 + hl:o + 4 + hl + h["n"]]))
        o += 4 + hl + h["n"]
    return out


def baked_quantiles():
    src = RARITY_TS.read_text()
    body = src[src.index("BAKED_QUANTILES"):]
    body = body[:body.index("];")]
    return [(float(p), float(v)) for p, v in re.findall(r"\[([\d.e-]+),\s*([\d.e-]+)\]", body)]


def quantile_f(q, a):  # data/rarity.ts quantileF
    x, n = math.log(a), len(q)
    if x <= math.log(q[0][1]):
        (p0, v0), (p1, v1) = q[0], q[1]
        return max(0.0, p0 + ((x - math.log(v0)) * (p1 - p0)) / (math.log(v1) - math.log(v0)))
    for j in range(1, n):
        (p0, v0), (p1, v1) = q[j - 1], q[j]
        if x <= math.log(v1):
            return p0 + ((x - math.log(v0)) * (p1 - p0)) / (math.log(v1) - math.log(v0))
    (p0, v0), (p1, v1) = q[n - 2], q[n - 1]
    alpha = math.log((1 - p0) / (1 - p1)) / math.log(v1 / v0)
    return 1 - (1 - p1) * (v1 / a) ** alpha


class Rarity:  # data/rarity.ts LiveRarity, seeded from tableFromQuantiles()
    def __init__(self, q):
        cdf = np.array([quantile_f(q, math.exp(LOG_LO + ((b + 1) * (LOG_HI - LOG_LO)) / BINS)) for b in range(BINS)], np.float32)
        cdf[-1] = 1
        weight = SEED_TOKENS * PER_TOKEN
        self.hist, prev = np.zeros(BINS), 0.0
        for b in range(BINS):
            c = float(cdf[b])
            self.hist[b] = max(0.0, c - prev) * weight
            prev = max(prev, c)
        self.total = float(weight)
        self.edge_risk = 0

    def add(self, r):
        a = np.abs(r.astype(np.float64))
        a = a[(a > 0) & np.isfinite(a)]
        x = (np.log(a) - LOG_LO) * (BINS / (LOG_HI - LOG_LO))
        near = np.abs(x - np.round(x)) < 1e-6  # vectorised log may differ from Math.log by an ulp: redo these with libm
        x[near] = [(math.log(v) - LOG_LO) * (BINS / (LOG_HI - LOG_LO)) for v in a[near]]
        self.edge_risk += int(np.sum(np.abs(x - np.round(x)) < 1e-12))
        np.add.at(self.hist, np.clip(np.floor(x).astype(np.int64), 0, BINS - 1), 1)
        self.total += len(a)

    def top1_at(self):  # quantileOf(table(), 0.99)
        cdf = (np.cumsum(self.hist) / self.total).astype(np.float32).astype(np.float64)
        b = int(np.argmax(cdf >= TOP1_P)) if (cdf >= TOP1_P).any() else BINS
        if b >= BINS:
            return math.exp(LOG_HI)
        lo = cdf[b - 1] if b > 0 else 0.0
        f = (TOP1_P - lo) / (cdf[b] - lo) if cdf[b] > lo else 1.0
        return math.exp(LOG_LO + ((b + max(0.0, min(1.0, f))) * (LOG_HI - LOG_LO)) / BINS)


def expected(d):
    meta = json.loads((d / "meta.json").read_text())
    L, N = meta["layers"], meta["neurons"]
    cal = np.fromfile(d / "calibration.f32", "<f4")
    inv = (1.0 / cal.astype(np.float64)).astype(np.float32)
    F, toks, margin = Rarity(baked_quantiles()), [], math.inf
    for h, _, payload in read_frames((d / "session.bin").read_bytes()):
        if h["kind"] == "done":
            done = {k: h[k] for k in ("text", "tok_s", "n_prompt", "n_gen")}
            continue
        r = (np.frombuffer(payload, "<f2").astype(np.float64) * inv).astype(np.float32)
        at = F.top1_at()
        a = np.abs(r.astype(np.float64))
        hit = a >= at
        margin = min(margin, float(np.min(np.abs(a - at))) / at)
        by_layer = hit.reshape(L, N).sum(1)
        nz = np.flatnonzero(a > 0)
        order = nz[np.lexsort((nz, -a[nz]))][:RAREST]
        toks.append({"i": h["i"], "kind": h["kind"], "text": h["text"], "top1At": at, "top1": int(hit.sum()),
                     "top1ByLayer": by_layer.tolist(),
                     "rarest": [{"layer": int(k // N), "neuron": int(k % N), "r": float(r[k])} for k in order]})
        if h["i"] >= 1:
            F.add(r)
    later = [t["top1"] for t in toks if t["i"] >= 1]
    out = {"tokens": toks, "tape": {"heights": [t["top1"] for t in toks], "scale": max([1] + later)}, "done": done,
           "checks": {"closest |r| to a threshold (relative)": margin, "values within 1e-12 of a histogram bin edge": F.edge_risk}}
    body = ",\n".join(json.dumps(t, ensure_ascii=False) for t in toks)  # one token per line
    rest = json.dumps({k: v for k, v in out.items() if k != "tokens"}, ensure_ascii=False)[1:]
    (d / "expected.json").write_text(f'{{"tokens": [\n{body}\n],\n{rest}\n')
    print(f"{d / 'expected.json'}: {len(toks)} tokens, counters {[t['top1'] for t in toks]}, checks {out['checks']}")


def trim(rec, keep, d):
    want = set()
    for part in keep.split(","):
        lo, _, hi = part.partition("-")
        want.update(range(int(lo), int(hi or lo) + 1))
    frames = read_frames(rec.read_bytes())
    buf = b"".join(raw for h, raw, _ in frames if h["i"] in want or h["kind"] == "done")
    d.mkdir(parents=True, exist_ok=True)
    (d / "session.bin").write_bytes(buf)
    for f in ("calibration.f32", "meta.json"):
        shutil.copy(rec.parent / f, d / f)
    print(f"{d / 'session.bin'}: {len(buf) / 1e6:.2f} MB")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trim")
    t.add_argument("rec", type=Path)
    t.add_argument("--keep", required=True)
    t.add_argument("--out", type=Path, default=HERE / "stats")
    e = sub.add_parser("expected")
    e.add_argument("--dir", type=Path, default=HERE / "stats")
    a = p.parse_args()
    if a.cmd == "trim":
        trim(a.rec, a.keep, a.out)
    else:
        expected(a.dir)


if __name__ == "__main__":
    main()
