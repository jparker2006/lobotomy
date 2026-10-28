// Every number the Cinema UI and the Console show (U3), as pure functions of the session's r arrays and the live F, so a test
// can replay a recorded session and check them against numpy (web/test/fixtures/stats_fixture.py).
// "Top 1%" = |r| at or above F's 99th percentile: the same cut as the ladder's level 3 for a one-neuron cell (design.md).
import type { RarityTable } from "../render/rarity.js";
import { quantileOf } from "./rarity.js";

export const TOP1_P = 0.99;

/** |r| at which a neuron is in its top 1% under F (the pooled CDF of |r|). */
export function top1Threshold(F: RarityTable): number { return quantileOf(F, TOP1_P); }

/** Neurons with |r| >= at, in total and per layer. */
export function countTop1(r: ArrayLike<number>, at: number, layers = 28, neurons = 3072): { total: number; byLayer: Uint16Array } {
  const byLayer = new Uint16Array(layers);
  let total = 0;
  for (let l = 0, k = 0; l < layers; l++) {
    let c = 0;
    for (let n = 0; n < neurons; n++, k++) if (Math.abs(r[k]) >= at) c++;
    byLayer[l] = c;
    total += c;
  }
  return { total, byLayer };
}

/** Tape: one bar per session token, height = its top-1% count. scale = the tallest bar past position 0 (the attention sink is
 *  drawn, clipped, but never sets the scale). */
export function tape(tokens: readonly { i: number; top1: number }[]): { heights: number[]; scale: number } {
  let scale = 1;
  for (const t of tokens) if (t.i >= 1 && t.top1 > scale) scale = t.top1;
  return { heights: tokens.map((t) => t.top1), scale };
}

export interface Rare { layer: number; neuron: number; r: number }

/** The n neurons with the largest |r| on a token (the rarest under the pooled F, which is monotone in |r|). Ties: lower index first. */
export function rarest(r: ArrayLike<number>, n = 6, neurons = 3072): Rare[] {
  const best: number[] = []; // indices, sorted by |r| descending
  const a = (k: number) => Math.abs(r[k]);
  for (let k = 0; k < r.length; k++) {
    const v = a(k);
    if (!(v > 0)) continue;
    if (best.length === n && v <= a(best[n - 1])) continue;
    let j = Math.min(best.length, n - 1);
    while (j > 0 && a(best[j - 1]) < v) j--;
    best.splice(j, 0, k);
    if (best.length > n) best.pop();
  }
  return best.map((k) => ({ layer: Math.floor(k / neurons), neuron: k % neurons, r: r[k] }));
}
