// SYNTHETIC activations for benchmarks and tests: the real frame size (28 x 3,072, layer-major) with
// |r| (= activation / p99) shaped like the recorded CPU session. Anything drawn from this must be labelled SYNTHETIC.
import { NEURONS, RarityTable, rarityTableFromSamples } from "./rarity.js";

/** Recorded |r| quantiles (pooled, positions >= 1). */
export const RECORDED_QUANTILES: [number, number][] = [[0.5, 0.10], [0.9, 0.43], [0.95, 0.59], [0.99, 1.1], [0.999, 3.2]];

/** Inverse standard normal CDF (Acklam). */
export function probit(p: number): number {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const lo = 0.02425;
  if (p < lo) { const q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  if (p > 1 - lo) return -probit(1 - p);
  const q = p - 0.5, r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

/** |r| at percentile p: log|r| is piecewise linear in probit(p) through the recorded quantiles, extended past the ends. */
export function recordedQuantile(p: number): number {
  const K = RECORDED_QUANTILES.map(([q, v]) => [probit(q), Math.log(v)]), z = probit(p);
  let j = 0;
  while (j < K.length - 2 && z > K[j + 1][0]) j++;
  const [z0, y0] = K[j], [z1, y1] = K[j + 1];
  return Math.exp(y0 + ((z - z0) * (y1 - y0)) / (z1 - z0));
}

const QT_BITS = 16, QT = new Float32Array(1 << QT_BITS);
for (let i = 0; i < QT.length; i++) QT[i] = recordedQuantile((i + 0.5) / QT.length);

/** F seeded from the recorded quantiles (what the synthetic stream draws from). */
export function recordedRarityTable(bins = 512): RarityTable { return rarityTableFromSamples(QT, bins); }

/**
 * A synthetic token stream. Each token every neuron draws a fresh |r| from the recorded distribution with a random sign,
 * except a `keep` share that hold their last value, so most of the lit set changes from token to token.
 */
export function makeSynth(seed = 1, keep = 0.25) {
  let s = seed >>> 0 || 1;
  const prev = new Float32Array(NEURONS);
  let keepBelow = 0; // the first token is all fresh
  return {
    next(out: Float32Array = new Float32Array(NEURONS)) {
      for (let i = 0; i < NEURONS; i++) {
        s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0;
        if (((s >>> 1) & 1023) < keepBelow) { out[i] = prev[i]; continue; }
        const v = QT[s >>> (32 - QT_BITS)];
        out[i] = prev[i] = s & 1 ? v : -v;
      }
      keepBelow = Math.round(keep * 1024);
      return out;
    },
  };
}
