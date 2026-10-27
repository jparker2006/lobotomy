// Rarity math shared by the GPU pipeline, the CPU fallback and the tests (no three.js here).
// design.md "Rarity": r = a / p99, F = pooled CDF of |r|, a cell of k neurons uses u = F(max|r|)^k.

export const LAYERS = 28, PER_LAYER = 3072, NEURONS = LAYERS * PER_LAYER;
export const THRESHOLDS = [0.90, 0.97, 0.99, 0.997, 0.999];

/** Pooled CDF of |r| on log-spaced bins: cdf[b] = P(|r| < exp(logLo + (b + 1) * (logHi - logLo) / cdf.length)). */
export interface RarityTable { logLo: number; logHi: number; cdf: Float32Array }

/** Builds a table from samples of r (signs ignored, zeros skipped). Values below the range land in bin 0. */
export function rarityTableFromSamples(r: ArrayLike<number>, bins = 512, logLo = Math.log(1e-3), logHi = Math.log(1e3)): RarityTable {
  const hist = new Float64Array(bins);
  let n = 0;
  for (let j = 0; j < r.length; j++) {
    const a = Math.abs(r[j]);
    if (!(a > 0)) continue;
    const b = Math.floor(((Math.log(a) - logLo) / (logHi - logLo)) * bins);
    hist[Math.min(bins - 1, Math.max(0, b))]++;
    n++;
  }
  const cdf = new Float32Array(bins);
  let acc = 0;
  for (let b = 0; b < bins; b++) { acc += hist[b]; cdf[b] = n ? acc / n : 1; }
  return { logLo, logHi, cdf };
}

/** F(|r|), linear inside a bin (the GPU does the same). */
export function rarityF(t: RarityTable, a: number): number {
  a = Math.abs(a);
  if (!(a > 0)) return 0;
  const B = t.cdf.length, x = ((Math.log(a) - t.logLo) / (t.logHi - t.logLo)) * B;
  if (x <= 0) return 0;
  if (x >= B) return 1;
  const b = Math.floor(x), f = x - b;
  return (b > 0 ? t.cdf[b - 1] : 0) * (1 - f) + t.cdf[b] * f;
}

/** Level 0..5 for a crowding-corrected percentile u. */
export function levelOf(u: number, th: readonly number[] = THRESHOLDS): number {
  let lv = 0;
  while (lv < th.length && u >= th[lv]) lv++;
  return lv;
}

/** Level of a cell holding k >= 1 neurons whose strongest |r| is m. */
export function cellLevel(t: RarityTable, m: number, k: number, th: readonly number[] = THRESHOLDS): number {
  return levelOf(Math.pow(rarityF(t, m), Math.max(1, k)), th);
}

// Splat key: quantized log|r| in the high 15 bits, neuron index in the low 17 (86,016 < 2^17),
// so an atomic max per cell keeps the strongest neuron and the winner is recoverable for picking.
export const INDEX_BITS = 17, INDEX_MASK = (1 << INDEX_BITS) - 1, MAG_MAX = (1 << (32 - INDEX_BITS)) - 1;
export const KEY_LOG_LO = Math.log(1e-4), KEY_LOG_HI = Math.log(1e4);
export const KEY_SCALE = MAG_MAX / (KEY_LOG_HI - KEY_LOG_LO);

export function packKey(r: number, index: number): number {
  const a = Math.max(Math.abs(r), 1e-20);
  const q = Math.min(MAG_MAX, Math.max(0, Math.floor((Math.log(a) - KEY_LOG_LO) * KEY_SCALE)));
  return ((q << INDEX_BITS) | index) >>> 0;
}
