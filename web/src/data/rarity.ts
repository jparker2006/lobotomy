// The rarity ladder's F (design.md "Rarity"): the pooled CDF of |r| = |activation / p99| over all neurons and positions >= 1
// (position 0, the attention sink, is drawn but never counted). The page seeds F from a baked table and updates it with a
// running histogram of everything this backend streams. A cell of k neurons with strongest |r| = m gets u = F(m)^k.
import { RarityTable, THRESHOLDS, rarityF } from "../render/rarity.js";

/**
 * Baked |r| quantiles [p, |r|] from the CPU stand-in (Qwen3-0.6B, transformers fp32, T2 calibration): 4 recorded answers
 * ("Describe the ocean at night in three sentences.", "Why is the sky blue? Answer in two sentences.", "Write a short poem
 * about a cat who learns to code.", "Explain how a bicycle stays upright."), 513 tokens at positions >= 1, 44.1 M values.
 */
export const BAKED_QUANTILES: readonly (readonly [number, number])[] = [
  [0.01, 0.0006691], [0.02, 0.0015], [0.05, 0.004436], [0.1, 0.01033], [0.2, 0.02524], [0.3, 0.04434], [0.4, 0.06849],
  [0.5, 0.09941], [0.6, 0.1401], [0.7, 0.196], [0.8, 0.2801], [0.85, 0.3425], [0.9, 0.434], [0.93, 0.5182], [0.95, 0.6017],
  [0.97, 0.7394], [0.98, 0.8647], [0.99, 1.142], [0.993, 1.339], [0.995, 1.574], [0.997, 2.035], [0.998, 2.502], [0.999, 3.555],
  [0.9995, 5.027], [0.9997, 6.494], [0.9999, 11.03], [0.99997, 18.98], [0.99999, 32.72],
];

export const BINS = 512, LOG_LO = Math.log(1e-3), LOG_HI = Math.log(1e3);

/** F at |r| = a from quantile knots: p linear in log|r| between knots, a power-law tail past the last knot. */
export function quantileF(q: readonly (readonly [number, number])[], a: number): number {
  if (!(a > 0)) return 0;
  const x = Math.log(a), n = q.length;
  if (x <= Math.log(q[0][1])) { // below the first knot: extend the first segment down to 0
    const [p0, v0] = q[0], [p1, v1] = q[1];
    return Math.max(0, p0 + ((x - Math.log(v0)) * (p1 - p0)) / (Math.log(v1) - Math.log(v0)));
  }
  for (let j = 1; j < n; j++) {
    const [p0, v0] = q[j - 1], [p1, v1] = q[j];
    if (x <= Math.log(v1)) return p0 + ((x - Math.log(v0)) * (p1 - p0)) / (Math.log(v1) - Math.log(v0));
  }
  const [p0, v0] = q[n - 2], [p1, v1] = q[n - 1];
  const alpha = Math.log((1 - p0) / (1 - p1)) / Math.log(v1 / v0); // 1 - F ~ a^-alpha
  return 1 - (1 - p1) * Math.pow(v1 / a, alpha);
}

const logKnots = new WeakMap<object, Float64Array>();

/** |r| at percentile p from quantile knots: the inverse of quantileF (for SYNTHETIC draws and tests). */
export function quantileR(q: readonly (readonly [number, number])[], p: number): number {
  let L = logKnots.get(q);
  if (!L) logKnots.set(q, (L = Float64Array.from(q, ([, v]) => Math.log(v))));
  const n = q.length;
  if (p > q[n - 1][0]) {
    const p0 = q[n - 2][0], p1 = q[n - 1][0], alpha = Math.log((1 - p0) / (1 - p1)) / (L[n - 1] - L[n - 2]);
    return Math.exp(L[n - 1] + Math.log((1 - p1) / (1 - p)) / alpha);
  }
  let j = 0;
  while (j < n - 2 && p > q[j + 1][0]) j++;
  return Math.exp(L[j] + ((p - q[j][0]) * (L[j + 1] - L[j])) / (q[j + 1][0] - q[j][0]));
}

/** A rarity table (render/rarity.ts layout: cdf[b] = F at the upper edge of bin b) from quantile knots. */
export function tableFromQuantiles(q: readonly (readonly [number, number])[] = BAKED_QUANTILES, bins = BINS): RarityTable {
  const cdf = new Float32Array(bins);
  for (let b = 0; b < bins; b++) cdf[b] = quantileF(q, Math.exp(LOG_LO + ((b + 1) * (LOG_HI - LOG_LO)) / bins));
  cdf[bins - 1] = 1;
  return { logLo: LOG_LO, logHi: LOG_HI, cdf };
}

/** Level 0..5 of a cell of k >= 1 neurons whose strongest |r| is max: u = F(max)^k against the ladder. */
export function levelOf(max: number, k: number, F: RarityTable, thresholds: readonly number[] = THRESHOLDS): number {
  const u = Math.pow(rarityF(F, max), Math.max(1, k));
  let lv = 0;
  while (lv < thresholds.length && u >= thresholds[lv]) lv++;
  return lv;
}

/** Smallest |r| with F(|r|) >= p (bin-linear, the inverse of rarityF). */
export function quantileOf(F: RarityTable, p: number): number {
  const c = F.cdf, B = c.length;
  let b = 0;
  while (b < B && c[b] < p) b++;
  if (b >= B) return Math.exp(F.logHi);
  const lo = b > 0 ? c[b - 1] : 0, f = c[b] > lo ? (p - lo) / (c[b] - lo) : 1;
  return Math.exp(F.logLo + ((b + Math.max(0, Math.min(1, f))) * (F.logHi - F.logLo)) / B);
}

/**
 * Pooled F of |r|, seeded from a table and updated with every token streamed. The seed counts as `seedTokens` tokens of
 * 86,016 values, so a backend whose numbers differ from the CPU stand-in's takes over within a few answers.
 */
export class LiveRarity {
  readonly hist = new Float64Array(BINS);
  total = 0;
  private cached: RarityTable | null = null;

  constructor(seed: RarityTable = tableFromQuantiles(), seedTokens = 32, perToken = 86016) {
    this.reset(seed, seedTokens * perToken);
  }

  reset(seed: RarityTable, weight: number) {
    if (seed.cdf.length !== BINS || seed.logLo !== LOG_LO || seed.logHi !== LOG_HI) throw new Error("LiveRarity: seed table must use the shared bins");
    let prev = 0;
    for (let b = 0; b < BINS; b++) { this.hist[b] = Math.max(0, seed.cdf[b] - prev) * weight; prev = Math.max(prev, seed.cdf[b]); }
    this.total = weight;
    this.cached = null;
  }

  /** Adds |r| of one token (zeros and non-finite values skipped). Do not call it for position 0. */
  add(r: ArrayLike<number>) {
    const h = this.hist, s = BINS / (LOG_HI - LOG_LO);
    let n = 0;
    for (let j = 0; j < r.length; j++) {
      const a = Math.abs(r[j]);
      if (!(a > 0) || a === Infinity) continue;
      const b = Math.floor((Math.log(a) - LOG_LO) * s);
      h[b < 0 ? 0 : b >= BINS ? BINS - 1 : b]++;
      n++;
    }
    this.total += n;
    this.cached = null;
  }

  table(): RarityTable {
    if (this.cached) return this.cached;
    const cdf = new Float32Array(BINS);
    let acc = 0;
    for (let b = 0; b < BINS; b++) { acc += this.hist[b]; cdf[b] = this.total ? acc / this.total : 1; }
    return (this.cached = { logLo: LOG_LO, logHi: LOG_HI, cdf });
  }
}
