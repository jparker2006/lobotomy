// Tiny perspective camera for canvas mockups (no three.js needed for stills).
import { L, N } from "./data.js";

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const l = Math.hypot(...a); return [a[0] / l, a[1] / l, a[2] / l]; };
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

export function camera(eye, target, fovDeg, W, H, cx = W / 2, cy = H / 2) {
  const f = norm(sub(target, eye)), r = norm(cross(f, [0, 1, 0])), u = cross(r, f);
  const k = (H / 2) / Math.tan((fovDeg * Math.PI) / 360);
  return (p) => {
    const d = sub(p, eye), z = dot(d, f);
    return { x: cx + (dot(d, r) * k) / z, y: cy - (dot(d, u) * k) / z, z, s: k / z };
  };
}

// Layer sheets: 28 sheets of 64 x 48 neurons, standing in the y-z plane, spaced along x.
export const GRID_W = 64, GRID_H = 48;
export function sheetPos(l, n, gap = 0.62, pitch = 0.05) {
  const row = Math.floor(n / GRID_W), col = n % GRID_W;
  return [(l - (L - 1) / 2) * gap, ((GRID_H - 1) / 2 - row) * pitch, (col - (GRID_W - 1) / 2) * pitch];
}
export function sheetCorners(l, gap = 0.62, pitch = 0.05) {
  const x = (l - (L - 1) / 2) * gap, hy = (GRID_H / 2) * pitch, hz = (GRID_W / 2) * pitch;
  return [[x, hy, -hz], [x, hy, hz], [x, -hy, hz], [x, -hy, -hz]];
}

// Projects every neuron once. Returns typed arrays sx, sy, sz (depth), ss (pixels per world unit).
export function projectAll(cam, gap, pitch) {
  const M = L * N, sx = new Float32Array(M), sy = new Float32Array(M), sz = new Float32Array(M), ss = new Float32Array(M);
  for (let l = 0; l < L; l++) for (let n = 0; n < N; n++) {
    const p = cam(sheetPos(l, n, gap, pitch)), k = l * N + n;
    sx[k] = p.x; sy[k] = p.y; sz[k] = p.z; ss[k] = p.s;
  }
  return { sx, sy, sz, ss };
}
