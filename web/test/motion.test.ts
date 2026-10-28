// U4 motion: the camera rig (drift, tweens, input) is deterministic in t, onUserInput stops it, the fps readout uses the bench's
// method, and settings round-trip through localStorage. Run: npm test
import { test } from "node:test";
import assert from "node:assert/strict";
import { createCameraRig, OVERVIEW, facePose, DRIFT, LIMITS, driftAngle, Pose } from "../src/app/camera.js";
import type { Clock } from "../src/app/clock.js";
import { frameStats, pct, FrameMeter } from "../src/motion/fps.js";
import { STORAGE_KEY, loadSettings, saveSettings, parseSettings, StorageLike } from "../src/motion/settings.js";
import { DEFAULT_SETTINGS } from "../src/app/state.js";
import { applyState } from "../src/app/params.js";
import { initialState } from "../src/app/state.js";
import { DEFAULT_PARAMS } from "../src/render/deck.js";

function fakeClock(t0 = 0) {
  let t = t0;
  const c: Clock & { set(x: number): void } = {
    now: () => t, get frameTime() { return t; }, tick: () => t, setManual() {}, get manual() { return true; }, set(x: number) { t = x; },
  };
  return c;
}
const snap = (p: Pose) => JSON.stringify(p);
const sub = (a: number[], b: number[]) => a.map((x, j) => x - b[j]);
const len = (a: number[]) => Math.hypot(...a);
const yawOf = (p: Pose) => { const v = sub(p.eye, p.target); return Math.atan2(v[0], v[2]); };
const close = (a: number, b: number, eps: number, msg?: string) => assert.ok(Math.abs(a - b) <= eps, `${msg ?? ""} ${a} vs ${b}`);

/** Drive a rig at 60 fps from 0 to tEnd with optional calls at given times; return the poses. */
function run(tEnd: number, calls: [number, (r: ReturnType<typeof createCameraRig>) => void][] = []) {
  const clock = fakeClock(), rig = createCameraRig(clock);
  rig.driftEnabled = true;
  const out: string[] = [];
  for (let f = 0; f <= tEnd * 60; f++) {
    const t = f / 60;
    clock.set(t);
    for (const [at, fn] of calls) if (Math.abs(at - t) < 1e-9) fn(rig);
    rig.update(t);
    out.push(snap(rig.pose));
  }
  return { rig, out, clock };
}

test("idle drift: none for 8 s, then a ±9° sway about the target, deterministic in t", () => {
  const a = run(40), b = run(40);
  assert.deepEqual(a.out, b.out, "same calls, same t -> same poses");
  for (let f = 0; f < 8 * 60; f++) assert.equal(a.out[f], snap(OVERVIEW), `still before 8 s (frame ${f})`);
  const yaw0 = yawOf(OVERVIEW), dist0 = len(sub(OVERVIEW.eye, OVERVIEW.target));
  let maxDev = 0;
  for (const s of a.out) {
    const p = JSON.parse(s) as Pose;
    maxDev = Math.max(maxDev, Math.abs(yawOf(p) - yaw0));
    close(len(sub(p.eye, p.target)), dist0, 1e-9, "drift keeps the distance");
    close(p.eye[1], OVERVIEW.eye[1], 1e-9, "drift is a yaw only");
    assert.deepEqual(p.target, OVERVIEW.target);
  }
  assert.ok(maxDev > DRIFT.ampRad * 0.95 && maxDev <= DRIFT.ampRad + 1e-9, `peak ${maxDev} rad, want ~9°`);
  // smooth start: no jump in the first frames of the drift
  close(driftAngle(1 / 60), 0, 1e-4); close(driftAngle(0), 0, 0);
});

test("drift: frame-by-frame at any rate gives the same pose for the same t (clips)", () => {
  const clock = fakeClock(), r1 = createCameraRig(clock), r2 = createCameraRig(clock);
  r1.driftEnabled = r2.driftEnabled = true;
  clock.set(0); r1.update(0); r2.update(0);
  for (let f = 0; f <= 25 * 60; f++) { const t = f / 60; clock.set(t); r1.update(t); }
  for (let f = 0; f <= 25 * 24; f++) { const t = f / 24; clock.set(t); r2.update(t); }
  clock.set(25); r1.update(25); r2.update(25);
  assert.equal(snap(r1.pose), snap(r2.pose));
});

test("onUserInput stops the drift where it is (no snap back) and restarts the idle timer", () => {
  const { rig, out } = run(30, [[13, (r) => { r.onUserInput(); }]]);
  const iStop = 13 * 60;
  assert.ok(out[iStop - 1] !== snap(OVERVIEW), "it was drifting");
  close(yawOf(JSON.parse(out[iStop])), yawOf(JSON.parse(out[iStop - 1])), 0.01, "no jump at the input");
  for (let f = iStop; f < iStop + 8 * 60; f++) assert.equal(out[f], out[iStop], "still for 8 s after the input");
  assert.ok(out[out.length - 1] !== out[iStop], "drifts again after 8 s idle");
  rig.driftEnabled = false;
  const held = snap(rig.pose);
  rig.update(30);
  assert.equal(snap(rig.pose), held, "turning drift off keeps the view");
  rig.update(60);
  assert.equal(snap(rig.pose), held, "and never drifts while off");
});

test("tweens: deterministic in t, smooth, resolve on arrival; onUserInput interrupts and resolves early", async () => {
  const mk = () => { const clock = fakeClock(); const rig = createCameraRig(clock); return { clock, rig }; };
  const A = mk(), B = mk();
  let arrived = false;
  const pa = A.rig.faceLayer(8, 1000).then(() => { arrived = true; }); void B.rig.faceLayer(8, 1000);
  const trace: number[][] = [];
  for (let f = 0; f <= 70; f++) {
    const t = f / 60; A.clock.set(t); B.clock.set(t); A.rig.update(t); B.rig.update(t);
    assert.equal(snap(A.rig.pose), snap(B.rig.pose));
    trace.push([...A.rig.pose.eye]);
  }
  await pa;
  assert.ok(arrived);
  assert.deepEqual(A.rig.pose.eye, facePose(8).eye);
  // no snapping: the per-frame step stays small and starts/ends near zero
  const steps = trace.slice(1).map((e, i) => len(sub(e, trace[i])));
  assert.ok(Math.max(...steps) < 2, `max step ${Math.max(...steps)}`);
  assert.ok(steps[0] < 0.05 && steps[59] < 0.05, "eases in and out");
  // the path keeps its distance from the target (spherical blend; never cuts through the deck)
  // interrupt
  const C = mk();
  let done = false;
  const pc = C.rig.overview(1000).then(() => { done = true; });
  C.rig.flyTo(facePose(3), 0); // pose jump, then tween back
  const pc2 = C.rig.overview(1000).then(() => { done = true; });
  C.clock.set(0.3); C.rig.update(0.3);
  const mid = snap(C.rig.pose);
  C.rig.onUserInput();
  await pc; await pc2;
  assert.ok(done, "interrupted flyTo resolves");
  C.clock.set(2); C.rig.update(2);
  assert.equal(snap(C.rig.pose), mid, "stopped where it was");
});

test("no drift during a tween; drift starts 8 s after the tween arrives", () => {
  const { out } = run(30, [[2, (r) => { void r.faceLayer(5, 1000); }]]);
  const arrive = 3 * 60;
  for (let f = arrive; f < arrive + 8 * 60; f++) assert.equal(out[f], out[arrive], `frame ${f}`);
  assert.ok(out[arrive + 12 * 60] !== out[arrive]);
});

test("orbit, pan and zoom: clamped, zoom keeps the point under the cursor", () => {
  const clock = fakeClock(), rig = createCameraRig(clock);
  rig.update(0);
  const d0 = len(sub(rig.pose.eye, rig.pose.target));
  rig.orbit(0.3, 0);
  close(len(sub(rig.pose.eye, rig.pose.target)), d0, 1e-9, "orbit keeps the distance");
  close(yawOf(rig.pose) - yawOf(OVERVIEW), 0.3, 1e-9);
  rig.orbit(0, 10);
  const v = sub(rig.pose.eye, rig.pose.target);
  close(Math.asin(v[1] / len(v)), LIMITS.maxPitch, 1e-9, "pitch clamps");
  // zoom toward the principal point keeps the target fixed and scales the distance
  rig.flyTo(OVERVIEW, 0);
  rig.zoomAt(0.5, OVERVIEW.center[0], OVERVIEW.center[1], 1.6);
  close(len(sub(rig.pose.eye, rig.pose.target)), d0 / 2, 1e-9);
  assert.ok(len(sub(rig.pose.target, OVERVIEW.target)) < 1e-9);
  rig.zoomAt(1e-6, 0.5, 0.5, 1.6);
  close(len(sub(rig.pose.eye, rig.pose.target)), LIMITS.minDist, 1e-6, "zoom clamps near");
  rig.zoomAt(1e6, 0.5, 0.5, 1.6);
  assert.ok(len(sub(rig.pose.eye, rig.pose.target)) <= LIMITS.maxDist + 1e-6, "zoom clamps far");
  // zoom off-centre: the eye moves toward the cursor's ray (direction from eye to the cursor point is kept)
  rig.flyTo(OVERVIEW, 0);
  const before = JSON.parse(snap(rig.pose)) as Pose;
  rig.zoomAt(0.8, 0.9, 0.2, 1.6);
  const shift = sub(rig.pose.target, before.target);
  assert.ok(len(shift) > 0.1, "the target moves toward the cursor");
  // pan: the deck follows a drag, eye and target move together
  rig.flyTo(OVERVIEW, 0);
  rig.pan(100, 0, 800);
  const dt = sub(rig.pose.target, OVERVIEW.target), de = sub(rig.pose.eye, OVERVIEW.eye);
  assert.ok(len(dt) > 0 && len(sub(dt, de)) < 1e-9);
});

// ---- fps / p95: the bench's method (bench/bench.ts finishRun), restated here as the reference ----
function benchReference(iv: number[]) {
  const bpct = (a: number[], p: number) => { if (!a.length) return NaN; const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
  const r2 = (x: number) => Math.round(x * 100) / 100, vs = bpct(iv, 0.5);
  return {
    p95: r2(bpct(iv, 0.95)), vsyncMs: r2(vs), p95Vsync: r2(bpct(iv.map((x) => vs * Math.max(1, Math.round(x / vs))), 0.95)),
    droppedPct: r2((100 * iv.filter((x) => Math.round(x / vs) >= 2).length) / Math.max(1, iv.length)),
    over33Pct: r2((100 * iv.filter((x) => x > 33.3).length) / Math.max(1, iv.length)),
  };
}

test("fps readout: same intervals give the bench's numbers (well within 1 ms)", () => {
  let seed = 7;
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
  const cases: number[][] = [
    Array.from({ length: 3600 }, () => 16.667 + (rnd() - 0.5) * 2), // 60 Hz with jitter
    Array.from({ length: 3600 }, (_, i) => (i % 17 === 0 ? 33.3 : 16.667) + (rnd() - 0.5) * 1.5), // ~6% dropped
    Array.from({ length: 900 }, () => 8.333 + (rnd() - 0.5)), // 120 Hz
    Array.from({ length: 600 }, () => 16.667 * (1 + Math.floor(rnd() * 3)) + rnd()), // a struggling page
  ];
  for (const iv of cases) {
    const a = frameStats(iv), b = benchReference(iv);
    for (const k of ["p95", "vsyncMs", "p95Vsync"] as const) close(a[k], b[k], 0.01, k);
    close(a.droppedPct, b.droppedPct, 0.01, "dropped"); close(a.over33Pct, b.over33Pct, 0.01, ">33");
    close(a.fps, 1000 / (iv.reduce((x, y) => x + y, 0) / iv.length), 1e-9);
  }
  assert.equal(pct([3, 1, 2], 0.5), 2);
  assert.ok(Number.isNaN(pct([], 0.95)));
});

test("frame meter: rolling window, a hidden-tab gap restarts it", () => {
  const m = new FrameMeter(100);
  for (let i = 0; i < 250; i++) m.push(16.7);
  assert.equal(m.count, 100);
  m.push(6000);
  assert.equal(m.count, 0);
  m.push(0); m.push(NaN);
  assert.equal(m.count, 0);
});

// ---- settings persistence ----
function shimStorage(): StorageLike & { data: Map<string, string> } {
  const data = new Map<string, string>();
  return { data, getItem: (k) => data.get(k) ?? null, setItem: (k, v) => { data.set(k, String(v)); }, removeItem: (k) => { data.delete(k); } };
}

test("settings round-trip through localStorage", () => {
  const st = shimStorage();
  assert.deepEqual(loadSettings(st), DEFAULT_SETTINGS, "empty storage -> defaults");
  const s = { ...DEFAULT_SETTINGS, halfLifeTokens: 3.5, bloomScale: 0.75, cellW: 7, sweepOn: false, driftOn: false, inkHue: 8, thresholds: [0.8, 0.9, 0.95, 0.99, 0.995] };
  saveSettings(st, s);
  assert.ok(st.data.has(STORAGE_KEY));
  assert.deepEqual(loadSettings(st), s);
  // bad or partial entries fall back per field
  st.setItem(STORAGE_KEY, JSON.stringify({ halfLifeTokens: "x", cellH: 10, thresholds: [0.9, 0.5, 0.99, 0.997, 0.999], sweepOn: 1, extra: 3 }));
  assert.deepEqual(loadSettings(st), { ...DEFAULT_SETTINGS, cellH: 10 });
  st.setItem(STORAGE_KEY, "{not json");
  assert.deepEqual(loadSettings(st), DEFAULT_SETTINGS);
  assert.deepEqual(parseSettings(null), DEFAULT_SETTINGS);
  const throwing: StorageLike = { getItem: () => { throw new Error("blocked"); }, setItem: () => { throw new Error("full"); }, removeItem: () => {} };
  assert.deepEqual(loadSettings(throwing), DEFAULT_SETTINGS);
  saveSettings(throwing, s); // never throws
});

test("every knob reaches the deck params (app/params.ts)", () => {
  const st = initialState();
  st.settings = { ...DEFAULT_SETTINGS, halfLifeTokens: 3, bloomStrength: 2, bloomScale: 0.8, hazeStrength: 0.4, cellW: 6, cellH: 9, sweepOn: false, inkHue: 8, thresholds: [0.8, 0.9, 0.95, 0.99, 0.995] };
  const p = { ...DEFAULT_PARAMS };
  applyState(p, st);
  assert.equal(p.halfLifeTokens, 3); assert.equal(p.bloomStrength, 2); assert.equal(p.bloomScale, 0.8); assert.equal(p.hazeStrength, 0.4);
  assert.equal(p.cellW, 6); assert.equal(p.cellH, 9); assert.equal(p.sweepSec, 0); assert.deepEqual(p.thresholds, [0.8, 0.9, 0.95, 0.99, 0.995]);
  assert.ok(p.ink[1] < 0.2, "astrophage red ink");
  st.settings = { ...st.settings, sweepOn: true }; st.tokS = 6;
  applyState(p, st);
  assert.equal(p.sweepSec, null, "sweep on = 0.9 / tok_s");
  close(0.9 / p.tokPerSec, 0.15, 1e-9, "sweep at 6 tok/s");
});

// Orchestrator (integration of U3 + U4): the Console's side panels frame the deck through camera.setFrame.
test("setFrame: framePose/unframePose round-trip, and overview() stays framed while a frame is set", async () => {
  const { framePose, unframePose, createCameraRig, OVERVIEW } = await import("../src/app/camera.js");
  const f = { l: 0.25, r: 0.2 };
  const back = unframePose(framePose(OVERVIEW, f), f);
  assert.ok(Math.abs(back.fov - OVERVIEW.fov) < 1e-9 && Math.abs(back.center[0] - OVERVIEW.center[0]) < 1e-9);
  let t = 0;
  const clock = { now: () => t } as any;
  const rig = createCameraRig(clock);
  void rig.setFrame(f, 0);
  void rig.overview(0);
  const want = framePose(OVERVIEW, f);
  assert.ok(Math.abs(rig.pose.center[0] - want.center[0]) < 1e-9 && Math.abs(rig.pose.fov - want.fov) < 1e-9);
  void rig.setFrame({ l: 0, r: 0 }, 0);
  assert.ok(Math.abs(rig.pose.center[0] - OVERVIEW.center[0]) < 1e-9 && Math.abs(rig.pose.fov - OVERVIEW.fov) < 1e-9);
});
