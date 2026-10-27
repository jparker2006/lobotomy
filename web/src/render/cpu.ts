// The compute stages on the CPU: the WebGL2 fallback and the reference the GPU kernels follow
// (same math, same cell output). No three.js here, so tests run it under node.
import { NEURONS, PER_LAYER, LAYERS, THRESHOLDS, INDEX_MASK, RarityTable, packKey, cellLevel } from "./rarity.js";
import { neuronPos } from "./geometry.js";

export const SLOTS = 4; // tokens that can be mid-sweep at once (sweep 0.9/tok_s < 1 token, so 2 would do)

export interface StageInputs {
  t: number; tPrev: number; hlSec: number; sweepSec: number; t0: Float64Array; // per slot: when the token starts sweeping
  vp: ArrayLike<number>; // view-projection, column-major (three.js Matrix4.elements)
  cssW: number; cssH: number; cellW: number; cellH: number; cols: number; rows: number;
}

/** Signed max-hold: keep v if |v| >= |g|. */
export const maxHold = (g: number, v: number) => (Math.abs(v) >= Math.abs(g) ? v : g);

export class CpuStages {
  readonly glow = new Float32Array(NEURONS);
  readonly tok = new Float32Array(SLOTS * NEURONS);
  readonly pos = new Float32Array(3 * NEURONS);
  keys = new Uint32Array(0);
  cnt = new Uint32Array(0);
  constructor() { for (let i = 0; i < NEURONS; i++) neuronPos(i, this.pos, 3 * i); }

  /** Trail + sweep: glow = maxHold(glow * 0.5^(dt/HL), token) once the sweep reaches the neuron's layer. */
  trail(u: StageInputs) {
    const { t, tPrev, hlSec, sweepSec, t0 } = u, g = this.glow, decay = Math.pow(0.5, (t - tPrev) / hlSec);
    for (let l = 0; l < LAYERS; l++) {
      const off = (l / LAYERS) * sweepSec, base = l * PER_LAYER;
      for (let i = base; i < base + PER_LAYER; i++) g[i] *= decay;
      for (let s = 0; s < SLOTS; s++) {
        const ta = t0[s] + off;
        if (!(ta > tPrev && ta <= t)) continue;
        const w = Math.pow(0.5, (t - ta) / hlSec), src = this.tok, so = s * NEURONS;
        for (let i = base; i < base + PER_LAYER; i++) g[i] = maxHold(g[i], src[so + i] * w);
      }
    }
  }

  /** Project every neuron; per cell keep max(packed key) and the count k. */
  splat(u: StageInputs) {
    const n = u.cols * u.rows;
    if (this.keys.length !== n) { this.keys = new Uint32Array(n); this.cnt = new Uint32Array(n); }
    const m = u.vp, P = this.pos, keys = this.keys, cnt = this.cnt;
    for (let i = 0; i < NEURONS; i++) {
      const x = P[3 * i], y = P[3 * i + 1], z = P[3 * i + 2];
      const cw = m[3] * x + m[7] * y + m[11] * z + m[15];
      if (cw <= 0.1) continue;
      const sx = ((m[0] * x + m[4] * y + m[8] * z + m[12]) / cw * 0.5 + 0.5) * u.cssW;
      const sy = (0.5 - (m[1] * x + m[5] * y + m[9] * z + m[13]) / cw * 0.5) * u.cssH;
      const cx = Math.floor(sx / u.cellW), cy = Math.floor(sy / u.cellH);
      if (cx < 0 || cy < 0 || cx >= u.cols || cy >= u.rows) continue;
      const c = cy * u.cols + cx, k = packKey(this.glow[i], i);
      if (k > keys[c]) keys[c] = k;
      cnt[c]++;
    }
  }

  /** Per cell: (level or -1 if empty, sign, view depth of the winner, winner index). Clears the cell. */
  resolve(u: StageInputs, table: RarityTable, out: Float32Array, th: readonly number[] = THRESHOLDS) {
    const m = u.vp, P = this.pos;
    for (let c = 0; c < this.keys.length; c++) {
      const k = this.cnt[c], o = 4 * c;
      if (!k) { out[o] = -1; out[o + 1] = 0; out[o + 2] = 0; out[o + 3] = 0; continue; }
      const w = this.keys[c] & INDEX_MASK, g = this.glow[w];
      out[o] = cellLevel(table, g, k, th);
      out[o + 1] = g < 0 ? -1 : 1;
      out[o + 2] = m[3] * P[3 * w] + m[7] * P[3 * w + 1] + m[11] * P[3 * w + 2] + m[15];
      out[o + 3] = w;
      this.keys[c] = 0; this.cnt[c] = 0;
    }
  }
}
