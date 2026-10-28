# Render bench (U1, #19)

A bare page that runs the real deck pipeline ([../src/render/](../src/render/README.md)) on **SYNTHETIC** activations at the real
frame size, and measures frame intervals. It answers E6: p95 ≤ 16.7 ms and ≤ 1% of frames over 33 ms on Jake's MacBook Air
(Chrome; Safari is untested and not a target; native Retina, bloom on).

**Run:** `cd web && npm install && npm run bench` (builds, then serves `web/dist` on `127.0.0.1:8019`; `-- --port N` to change).
Open `http://localhost:8019/bench/` in a full window, keep the tab in front and don't touch it for about 2 minutes.

**Plan** (starts 1.5 s after load): main run 60 s at native DPR, bloom 1× (full-resolution bloom, heavier than the 0.5× default),
24 tok/s; then 15 s each with one change: bloom 0.5×, DPR 1, 6 tok/s. The first 2 s of each run are dropped.
The page shows p50/p95/p99, % of frames over 33 ms, CPU time in `deck.frame()`, GPU ms per frame (where `timestamp-query` exists),
and POSTs everything to `serve.py`, which saves `results/<utc>-<browser>.json`.

**Verdict:** PASS when the main run has p95 ≤ 16.7 ms with intervals snapped to whole vsync periods (raw p95 also reported) and ≤ 1% over 33 ms,
with the tab never hidden. If it fails, the runs say which knob helps: bloom 0.5× (bloom cost), DPR 1 (fill cost), 6 tok/s (upload cost).

URL options: `?auto=0` (wait for Start), `?backend=webgl2` (force the fallback), `?scale=0.1` (shorter runs, for smoke tests).
Click a dot to see which neuron set it. `window.__showSnapshot()` draws one frame offscreen and shows it in a 2D canvas
(for sandboxes whose Chromium never composites WebGPU canvases).

`shots/` are sandbox renders (SwiftShader WebGPU at DPR 2, llvmpipe WebGL2 at DPR 1). `results/sandbox-*.json` is a sandbox run:
its timings are software rendering and mean nothing for the Mac.
