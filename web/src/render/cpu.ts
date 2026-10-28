// The compute stages on the CPU: the WebGL2 fallback and the reference the GPU kernels follow
// (same math, same cell output). No three.js here, so tests run it under node.
import { NEURONS, PER_LAYER, LAYERS, THRESHOLDS, INDEX_MASK, RarityTable, packKey, cellLevel } from "./rarity.js";
import { neuronPos } from "./geometry.js";

export const SLOTS = 4; // tokens that can be mid-sweep at once (sweep 0.9/tok_s < 1 token, so 2 would do)

export interface StageInputs {
  t: number; tPrev: number; hlSec: number; sweepSec: number; t0: Float64Array; // per slot: when the token starts sweeping
  vp: ArrayLike<number>; // view-projection, column-major (three.js Matrix4.elements)
  cssW: number; cssH: number; cellW: number; cellH: number; cols: number; rows: number;
  mask?: ArrayLike<number>; // per layer: 0 = hidden (no cells); absent = all shown
}

/** Signed max-hold: keep v if |v| >= |g|. */
export const maxHold = (g: number, v: number) => (Math.abs(v) >= Math.abs(g) ? v : g);

export class CpuStages {
  readonly glow = new Float32Array(NEURONS);
  readonly tok = new Float32Array(SLOTS * NEURONS);
  readonly pos = new Float32Array(3 * NEURONS);
  /** A fixed glow state to draw instead of the live trail (Deck.setGlow); the trail keeps running. */
  shown: Float32Array | null = null;
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
    const m = u.vp, P = this.pos, keys = this.keys, cnt = this.cnt, g = this.shown ?? this.glow, mask = u.mask;
    for (let i = 0; i < NEURONS; i++) {
      if (mask && !(mask[Math.floor(i / PER_LAYER)] > 0.5)) { i += PER_LAYER - 1 - (i % PER_LAYER); continue; }
      const x = P[3 * i], y = P[3 * i + 1], z = P[3 * i + 2];
      const cw = m[3] * x + m[7] * y + m[11] * z + m[15];
      if (cw <= 0.1) continue;
      const sx = ((m[0] * x + m[4] * y + m[8] * z + m[12]) / cw * 0.5 + 0.5) * u.cssW;
      const sy = (0.5 - (m[1] * x + m[5] * y + m[9] * z + m[13]) / cw * 0.5) * u.cssH;
      const cx = Math.floor(sx / u.cellW), cy = Math.floor(sy / u.cellH);
      if (cx < 0 || cy < 0 || cx >= u.cols || cy >= u.rows) continue;
      const c = cy * u.cols + cx, k = packKey(g[i], i);
      if (k > keys[c]) keys[c] = k;
      cnt[c]++;
    }
  }

  /** Per cell: (level or -1 if empty, sign, view depth of the winner, winner index). Clears the cell. */
  resolve(u: StageInputs, table: RarityTable, out: Float32Array, th: readonly number[] = THRESHOLDS) {
    const m = u.vp, P = this.pos, glow = this.shown ?? this.glow;
    for (let c = 0; c < this.keys.length; c++) {
      const k = this.cnt[c], o = 4 * c;
      if (!k) { out[o] = -1; out[o + 1] = 0; out[o + 2] = 0; out[o + 3] = 0; continue; }
      const w = this.keys[c] & INDEX_MASK, g = glow[w];
      out[o] = cellLevel(table, g, k, th);
      out[o + 1] = g < 0 ? -1 : 1;
      out[o + 2] = m[3] * P[3 * w] + m[7] * P[3 * w + 1] + m[11] * P[3 * w + 2] + m[15];
      out[o + 3] = w;
      this.keys[c] = 0; this.cnt[c] = 0;
    }
  }
}

/**
 * The trail state right after token k, for showing a past token (Deck.setGlow): the signed max-hold over tokens j <= k of
 * r_j * 0.5^((k - j) / halfLifeTokens), the stage-1 trail with every sweep complete and an even pace. Tokens older than
 * 12 half-lives (below 1/4096) are skipped.
 */
export function glowAt(tokens: ArrayLike<Float32Array>, k: number, halfLifeTokens: number, out = new Float32Array(NEURONS)): Float32Array {
  out.fill(0);
  const hl = Math.max(1e-3, halfLifeTokens), first = Math.max(0, k - Math.ceil(12 * hl));
  for (let j = first; j <= k && j < tokens.length; j++) {
    const r = tokens[j], w = Math.pow(0.5, (k - j) / hl);
    for (let i = 0; i < NEURONS; i++) out[i] = maxHold(out[i], r[i] * w);
  }
  return out;
}
