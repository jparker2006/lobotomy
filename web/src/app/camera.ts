// The camera rig. FROZEN API (ARCHITECTURE.md): overview(), faceLayer(l, ms?), flyTo(pose, ms), pose, onUserInput().
// U4 owns the internals (orbit / zoom / pan, idle drift) and may add to them without changing this API; U5 only calls it.
// The app loop calls update(t) once per frame, then hands `pose` to deck.setView().
//
// Internals (U4): `base` is the resting pose (where the user or a tween left the camera); `pose` = base plus the idle drift,
// a slow sway of the eye around the target (design.md Q7: ±9°, after 8 s without input, view only). Everything update(t)
// writes depends only on t and the calls made so far, so a clip renders the same frame by frame (U6).
import { DEFAULT_VIEW, View } from "../render/geometry.js";
import { layerX } from "../render/geometry.js";
import type { Clock } from "./clock.js";

/** A camera pose: the deck's View (eye, target, vertical fov in degrees, principal point as a fraction of the canvas). */
export type Pose = View;

export interface CameraRig {
  /** The pose to draw this frame (read-only; copy before keeping). */
  readonly pose: Pose;
  /** Fly to the default overview (design.md: front left, looking down the deck). */
  overview(ms?: number): Promise<void>;
  /** Fly to layer l face-on (lock-on). */
  faceLayer(l: number, ms?: number): Promise<void>;
  /** Fly to a pose (missing fields keep their target values). Resolves on arrival, or early if interrupted. */
  flyTo(pose: Partial<Pose>, ms: number): Promise<void>;
  /** The user touched the view (mouse, wheel, keys): stop any tween and idle drift now. */
  onUserInput(): void;
  /** Advance tweens to clock time t (seconds). Called by the app loop. */
  update(t: number): void;

  // ---- U4 additions (motion/ drives these; others may read them) ----
  /** Idle drift allowed (motion/ sets settings.driftOn && !locked). Off by default. Turning it off keeps the drifted pose. */
  driftEnabled: boolean;
  /** True while the drift is moving the view. */
  readonly drifting: boolean;
  /** Pointer activity that must not cancel a tween (hover): stops the drift and restarts the idle timer. */
  markActive(): void;
  /** Orbit around the target: yaw about world up, pitch (elevation) clamped to ±MAX_PITCH. Radians. */
  orbit(dYaw: number, dPitch: number): void;
  /** Pan by a drag of (dx, dy) CSS px on a canvas cssH tall: the deck follows the pointer. */
  pan(dx: number, dy: number, cssH: number): void;
  /** Zoom by factor (< 1 = closer) toward the point under (fx, fy) (fractions of the canvas), distance clamped. */
  zoomAt(factor: number, fx: number, fy: number, aspect: number): void;
}

export const OVERVIEW: Pose = { ...DEFAULT_VIEW };

/** Face-on pose of layer l: looking along +x at the sheet, neuron index rows left to right. */
export function facePose(l: number): Pose {
  const x = layerX(l);
  return { eye: [x - 10.8, 0, 0], target: [x, 0, 0], fov: DEFAULT_VIEW.fov, center: [0.5, 0.5] };
}

/** Idle drift (design.md Q7, mockup q7 C): ±9° yaw sway, 14 s period, starting 8 s after the last input. */
export const DRIFT = { idleSec: 8, ampRad: (9 * Math.PI) / 180, omega: 0.45, rampSec: 4 };
export const LIMITS = { minDist: 1.5, maxDist: 110, maxPitch: (82 * Math.PI) / 180, panBox: [16, 8, 8] as const };

type V3 = [number, number, number];
const clone = (p: Pose): Pose => ({ eye: [...p.eye], target: [...p.target], fov: p.fov, center: [...p.center] });
const copyInto = (dst: Pose, src: Pose) => {
  for (let j = 0; j < 3; j++) { dst.eye[j] = src.eye[j]; dst.target[j] = src.target[j]; }
  dst.fov = src.fov; dst.center[0] = src.center[0]; dst.center[1] = src.center[1];
};
const lerp = (a: number, b: number, f: number) => a + (b - a) * f;
const clamp = (x: number, a: number, b: number) => Math.min(b, Math.max(a, x));
/** Smootherstep: zero velocity and acceleration at both ends, so tweens never snap. */
const ease = (f: number) => f * f * f * (f * (f * 6 - 15) + 10);
const sub = (a: readonly number[], b: readonly number[]): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const len = (a: readonly number[]) => Math.hypot(a[0], a[1], a[2]);
const cross = (a: readonly number[], b: readonly number[]): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: readonly number[]): V3 => { const l = len(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

/** Eye offset from the target as (distance, azimuth about +y, elevation). */
function toSpherical(v: readonly number[]) {
  const r = len(v) || 1e-9;
  return { r, az: Math.atan2(v[0], v[2]), el: Math.asin(clamp(v[1] / r, -1, 1)) };
}
const fromSpherical = (r: number, az: number, el: number): V3 => [r * Math.cos(el) * Math.sin(az), r * Math.sin(el), r * Math.cos(el) * Math.cos(az)];

/** Camera basis of a pose: forward, right, up (unit vectors). */
function basis(p: Pose) {
  const f = norm(sub(p.target, p.eye)), r = norm(cross(f, [0, 1, 0])), u = cross(r, f);
  return { f, r, u };
}

/** Spherical blend about the target: target lerps, distance blends geometrically, direction slerps (no cutting through the deck). */
export function blendPose(a: Pose, b: Pose, e: number, out: Pose = clone(a)): Pose {
  const va = sub(a.eye, a.target), vb = sub(b.eye, b.target), ra = len(va) || 1e-9, rb = len(vb) || 1e-9;
  const ua = va.map((x) => x / ra), ub = vb.map((x) => x / rb);
  const dot = clamp(ua[0] * ub[0] + ua[1] * ub[1] + ua[2] * ub[2], -1, 1), om = Math.acos(dot);
  let dir: number[];
  if (om < 1e-4 || om > Math.PI - 1e-3) dir = norm(ua.map((x, j) => lerp(x, ub[j], e)));
  else { const s = Math.sin(om), wa = Math.sin((1 - e) * om) / s, wb = Math.sin(e * om) / s; dir = ua.map((x, j) => wa * x + wb * ub[j]); }
  const r = ra * Math.pow(rb / ra, e);
  for (let j = 0; j < 3; j++) { out.target[j] = lerp(a.target[j], b.target[j], e); out.eye[j] = out.target[j] + dir[j] * r; }
  out.fov = lerp(a.fov, b.fov, e);
  out.center[0] = lerp(a.center[0], b.center[0], e); out.center[1] = lerp(a.center[1], b.center[1], e);
  return out;
}

/** The drift's yaw offset s seconds after it started: a sine under a smooth ramp, so it starts and stops without a jolt. */
export function driftAngle(s: number): number {
  if (s <= 0) return 0;
  const f = Math.min(1, s / DRIFT.rampSec), env = f * f * (3 - 2 * f);
  return DRIFT.ampRad * env * Math.sin(DRIFT.omega * s);
}

/** Rotate a pose's eye about the target's vertical axis. */
function yawed(p: Pose, angle: number, out: Pose) {
  copyInto(out, p);
  if (!angle) return out;
  const v = sub(p.eye, p.target), c = Math.cos(angle), s = Math.sin(angle);
  out.eye[0] = p.target[0] + v[0] * c + v[2] * s;
  out.eye[2] = p.target[2] - v[0] * s + v[2] * c;
  return out;
}

export function createCameraRig(clock: Clock, start: Pose = OVERVIEW): CameraRig {
  const pose = clone(start), base = clone(start);
  let tween: { from: Pose; to: Pose; t0: number; dur: number; done: () => void } | null = null;
  let idleSince: number | null = null; // clock time of the last input or tween arrival (null: the first update sets it)
  let angle = 0, driftEnabled = false;

  function finish() { const d = tween?.done; tween = null; d?.(); }
  /** Keep the drifted view as the new resting pose, so stopping the drift never snaps the camera back. */
  function settle() { if (angle) { copyInto(base, pose); angle = 0; } }
  function touch() { settle(); idleSince = clock.now(); }
  function clampBase() {
    const v = sub(base.eye, base.target), { r, az, el } = toSpherical(v);
    const rc = clamp(r, LIMITS.minDist, LIMITS.maxDist), ec = clamp(el, -LIMITS.maxPitch, LIMITS.maxPitch);
    const shift = base.target.map((x, j) => clamp(x, -LIMITS.panBox[j], LIMITS.panBox[j]) - x);
    for (let j = 0; j < 3; j++) base.target[j] += shift[j];
    if (rc !== r || ec !== el) { const w = fromSpherical(rc, az, ec); for (let j = 0; j < 3; j++) base.eye[j] = base.target[j] + w[j]; }
    else for (let j = 0; j < 3; j++) base.eye[j] += shift[j];
    copyInto(pose, base);
  }
  /** Direct manipulation: stop tweens and drift, then edit the resting pose. */
  function manipulate(fn: () => void) { finish(); touch(); fn(); clampBase(); }

  const rig: CameraRig = {
    pose,
    overview: (ms = 900) => rig.flyTo(OVERVIEW, ms),
    faceLayer: (l, ms = 900) => rig.flyTo(facePose(l), ms),
    flyTo(p, ms) {
      finish();
      settle();
      const to = clone({ ...base, ...p } as Pose);
      if (!(ms > 0)) { copyInto(base, to); copyInto(pose, to); idleSince = clock.now(); return Promise.resolve(); }
      return new Promise<void>((done) => { tween = { from: clone(base), to, t0: clock.now(), dur: ms / 1000, done }; });
    },
    onUserInput() { finish(); touch(); },
    markActive() { touch(); },
    get driftEnabled() { return driftEnabled; },
    set driftEnabled(on: boolean) { if (!on) settle(); driftEnabled = on; },
    get drifting() { return angle !== 0; },
    orbit(dYaw, dPitch) {
      manipulate(() => {
        const { r, az, el } = toSpherical(sub(base.eye, base.target));
        const w = fromSpherical(r, az + dYaw, clamp(el + dPitch, -LIMITS.maxPitch, LIMITS.maxPitch));
        for (let j = 0; j < 3; j++) base.eye[j] = base.target[j] + w[j];
      });
    },
    pan(dx, dy, cssH) {
      manipulate(() => {
        const { r, u } = basis(base), k = (2 * len(sub(base.eye, base.target)) * Math.tan((base.fov * Math.PI) / 360)) / Math.max(1, cssH);
        for (let j = 0; j < 3; j++) { const d = (-r[j] * dx + u[j] * dy) * k; base.eye[j] += d; base.target[j] += d; }
      });
    },
    zoomAt(factor, fx, fy, aspect) {
      manipulate(() => {
        const dist = len(sub(base.eye, base.target)), f = clamp(factor, LIMITS.minDist / dist, LIMITS.maxDist / dist);
        // the point under the cursor on the plane through the target (the principal point sits at base.center)
        const { r, u } = basis(base), hh = dist * Math.tan((base.fov * Math.PI) / 360), hw = hh * aspect;
        const nx = (fx - base.center[0]) * 2, ny = (base.center[1] - fy) * 2;
        const P = base.target.map((x, j) => x + r[j] * nx * hw + u[j] * ny * hh);
        for (let j = 0; j < 3; j++) { base.eye[j] = P[j] + (base.eye[j] - P[j]) * f; base.target[j] = P[j] + (base.target[j] - P[j]) * f; }
      });
    },
    update(t) {
      if (idleSince === null) idleSince = t;
      if (tween) {
        const f = clamp((t - tween.t0) / tween.dur, 0, 1);
        blendPose(tween.from, tween.to, ease(f), base);
        if (f >= 1) { copyInto(base, tween.to); idleSince = Math.max(idleSince, tween.t0 + tween.dur); finish(); }
      }
      const s = t - (idleSince + DRIFT.idleSec);
      angle = driftEnabled && !tween ? driftAngle(s) : 0;
      yawed(base, angle, pose);
    },
  };
  return rig;
}
