// The camera rig. FROZEN API (ARCHITECTURE.md): overview(), faceLayer(l, ms?), flyTo(pose, ms), pose, onUserInput().
// U4 owns the internals (orbit / zoom / pan, idle drift) and may add to them without changing this API; U5 only calls it.
// The app loop calls update(t) once per frame, then hands `pose` to deck.setView().
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
}

export const OVERVIEW: Pose = { ...DEFAULT_VIEW };

/** Face-on pose of layer l: looking along +x at the sheet, neuron index rows left to right. */
export function facePose(l: number): Pose {
  const x = layerX(l);
  return { eye: [x - 10.8, 0, 0], target: [x, 0, 0], fov: DEFAULT_VIEW.fov, center: [0.5, 0.5] };
}

const clone = (p: Pose): Pose => ({ eye: [...p.eye], target: [...p.target], fov: p.fov, center: [...p.center] });
const lerp = (a: number, b: number, f: number) => a + (b - a) * f;
const ease = (f: number) => f * f * (3 - 2 * f);

export function createCameraRig(clock: Clock, start: Pose = OVERVIEW): CameraRig {
  const pose = clone(start);
  let tween: { from: Pose; to: Pose; t0: number; dur: number; done: () => void } | null = null;

  function finish() { const d = tween?.done; tween = null; d?.(); }

  const rig: CameraRig = {
    pose,
    overview: (ms = 900) => rig.flyTo(OVERVIEW, ms),
    faceLayer: (l, ms = 900) => rig.flyTo(facePose(l), ms),
    flyTo(p, ms) {
      finish();
      const to = clone({ ...pose, ...p } as Pose);
      if (!(ms > 0)) { Object.assign(pose, clone(to)); return Promise.resolve(); }
      return new Promise<void>((done) => { tween = { from: clone(pose), to, t0: clock.now(), dur: ms / 1000, done }; });
    },
    onUserInput() { finish(); },
    update(t) {
      if (!tween) return;
      const f = Math.min(1, Math.max(0, (t - tween.t0) / tween.dur)), e = ease(f), { from, to } = tween;
      for (let j = 0; j < 3; j++) { pose.eye[j] = lerp(from.eye[j], to.eye[j], e); pose.target[j] = lerp(from.target[j], to.target[j], e); }
      pose.fov = lerp(from.fov, to.fov, e);
      pose.center[0] = lerp(from.center[0], to.center[0], e); pose.center[1] = lerp(from.center[1], to.center[1], e);
      if (f >= 1) finish();
    },
  };
  return rig;
}
