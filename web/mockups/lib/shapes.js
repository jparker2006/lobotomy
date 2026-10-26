// Brain shapes for mockups. Each shape gives a world position per neuron (layer l, index n),
// outlines per layer (lists of 3D points, drawn as edge characters) and a label anchor per layer.
import { L, N } from "./data.js";

const GOLD = Math.PI * (3 - Math.sqrt(5));
// Fermat spiral in a unit disc, index order (v0.0.2 has no layout yet).
export function disc(n) {
  const r = Math.sqrt((n + 0.5) / N), a = n * GOLD;
  return [r * Math.cos(a), r * Math.sin(a)];
}
const circle = (f, steps = 96) => Array.from({ length: steps + 1 }, (_, i) => f((i / steps) * 2 * Math.PI));

export const SHAPES = {
  // 28 square sheets (64 x 48) standing side by side along x.
  deck: {
    pos(l, n, o = {}) {
      const gap = o.gap ?? 0.8, pitch = 0.05, row = Math.floor(n / 64), col = n % 64;
      return [(l - (L - 1) / 2) * gap, (23.5 - row) * pitch, (col - 31.5) * pitch];
    },
    outline(l, o = {}) {
      // full frame for the first and last layer, a top rule for the rest
      const x = (l - (L - 1) / 2) * (o.gap ?? 0.8), hy = 1.2, hz = 1.6;
      if (l === 0 || l === L - 1) return [[[x, hy, -hz], [x, hy, hz], [x, -hy, hz], [x, -hy, -hz], [x, hy, -hz]]];
      return [[[x, hy, -hz], [x, hy, hz]]];
    },
    label(l, o = {}) { return [(l - (L - 1) / 2) * (o.gap ?? 0.8), 1.45, -1.6]; },
  },
  // 28 round discs side by side along x (lenses).
  lenses: {
    pos(l, n, o = {}) {
      const [a, b] = disc(n), R = 1.6;
      return [(l - (L - 1) / 2) * (o.gap ?? 0.8), a * R, b * R];
    },
    outline(l, o = {}) {
      const x = (l - (L - 1) / 2) * (o.gap ?? 0.8);
      return [circle((t) => [x, Math.sin(t) * 1.68, Math.cos(t) * 1.68])];
    },
    label(l, o = {}) { return [(l - (L - 1) / 2) * (o.gap ?? 0.8), 2.0, 0]; },
  },
  // 28 discs stacked bottom (L0) to top (L27): a spine.
  spine: {
    pos(l, n, o = {}) {
      const [a, b] = disc(n), R = o.R ?? 3.4;
      return [a * R, (l - (L - 1) / 2) * (o.gap ?? 0.3), b * R];
    },
    outline(l, o = {}) {
      const y = (l - (L - 1) / 2) * (o.gap ?? 0.3), R = (o.R ?? 3.4) * 1.03;
      if (l % 3 && l !== L - 1) return [];
      return [circle((t) => [Math.cos(t) * R, y, Math.sin(t) * R])];
    },
    label(l, o = {}) { return [(o.R ?? 3.4) * 1.1, (l - (L - 1) / 2) * (o.gap ?? 0.3), 0]; },
  },
  // The deck bent into a ring: 28 sheets fanned like turbine blades, with a gap between L27 and L0.
  ring: {
    ang(l) { return (-75 + l * (330 / (L - 1))) * Math.PI / 180; },
    pos(l, n) {
      const row = Math.floor(n / 64), col = n % 64, a = this.ang(l), s = 0.9 + (col / 63) * 3.2, y = (23.5 - row) * 0.05;
      return [s * Math.cos(a), y, s * Math.sin(a)];
    },
    outline(l) {
      const a = this.ang(l), c = Math.cos(a), s = Math.sin(a), r0 = 0.85, r1 = 4.15, h = 1.2;
      return [[[r0 * c, h, r0 * s], [r1 * c, h, r1 * s], [r1 * c, -h, r1 * s], [r0 * c, -h, r0 * s], [r0 * c, h, r0 * s]]];
    },
    label(l) { const a = this.ang(l); return [4.5 * Math.cos(a), 1.35, 4.5 * Math.sin(a)]; },
  },
  // 28 equal-area rings in one disc: L0 at the centre, L27 at the rim (a radar scope).
  radar: {
    pos(l, n, o = {}) {
      const R = o.R ?? 4.2, r0 = R * Math.sqrt(l / L), r1 = R * Math.sqrt((l + 1) / L) * 0.985;
      const u = (n + 0.5) / N, r = Math.sqrt(r0 * r0 + u * (r1 * r1 - r0 * r0)), a = n * GOLD;
      return [r * Math.cos(a), 0, r * Math.sin(a)];
    },
    outline(l, o = {}) {
      if ((l + 1) % 7) return [];
      const r = (o.R ?? 4.2) * Math.sqrt((l + 1) / L);
      return [circle((t) => [Math.cos(t) * r, 0, Math.sin(t) * r], 128)];
    },
    label(l, o = {}) { const R = o.R ?? 4.2, r = R * Math.sqrt((l + 0.5) / L); return [r * 0.72, 0, r * 0.72]; },
  },
  // 28 concentric shells: L0 is the core, L27 the skin. A wedge facing the camera is cut away.
  onion: {
    pos(l, n, o = {}) {
      const r = 0.9 + l * 0.12, y = 1 - (2 * (n + 0.5)) / N, rho = Math.sqrt(1 - y * y), a = n * GOLD;
      const p = [r * rho * Math.cos(a), r * y, r * rho * Math.sin(a)];
      const cut = o.cut ?? true;
      return cut && p[0] < 0 && p[2] > 0 ? null : p;
    },
    outline(l) {
      // the two cut faces: x = 0 (z > 0) and z = 0 (x < 0), every third shell
      if (l % 3 && l !== L - 1) return [];
      const r = 0.9 + l * 0.12, half = (f) => Array.from({ length: 49 }, (_, i) => f(-Math.PI / 2 + (i / 48) * Math.PI));
      return [half((t) => [0, r * Math.sin(t), r * Math.cos(t)]), half((t) => [-r * Math.cos(t), r * Math.sin(t), 0])];
    },
    label(l) { const r = 0.9 + l * 0.12; return [0, -r - 0.05, r]; },
  },
};
