// U2 data/rarity.ts: the ladder's F (baked table + live histogram), crowding correction u = F(max)^k, the ladder levels.
// Run: npm test
import { test } from "node:test";
import assert from "node:assert/strict";
import { THRESHOLDS, rarityF } from "../src/render/rarity.js";
import { makeSynth } from "../src/render/synth.js";
import { BAKED_QUANTILES, LiveRarity, levelOf, quantileF, quantileOf, quantileR, tableFromQuantiles } from "../src/data/rarity.js";

const baked = tableFromQuantiles();
const top = THRESHOLDS.map((th) => 1 - th); // single-neuron share at level >= 1..5: 10%, 3%, 1%, 0.3%, 0.1%

/** mulberry32 -> uniform (0, 1), so the test is repeatable. */
function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
  };
}

test("baked table: F hits every baked quantile", () => {
  for (const [p, v] of BAKED_QUANTILES) {
    assert.ok(Math.abs(quantileF(BAKED_QUANTILES, v) - p) < 1e-9, `knot ${p}`);
    assert.ok(Math.abs(rarityF(baked, v) - p) < 0.002 * Math.max(1, (1 - p) * 100), `F(${v}) = ${rarityF(baked, v)}, want ${p}`);
    assert.ok(Math.abs(Math.log(quantileR(BAKED_QUANTILES, p) / v)) < 1e-9);
  }
  let prev = 0;
  for (let a = 1e-4; a < 1e4; a *= 1.05) { const f = rarityF(baked, a); assert.ok(f >= prev && f <= 1); prev = f; }
});

test("ladder: each level starts at its percentile", () => {
  for (let lv = 1; lv <= 5; lv++) {
    const x = quantileOf(baked, THRESHOLDS[lv - 1]);
    assert.equal(levelOf(x * 1.002, 1, baked), lv, `just above the ${THRESHOLDS[lv - 1]} quantile (${x.toFixed(4)})`);
    assert.equal(levelOf(x * 0.998, 1, baked), lv - 1, `just below the ${THRESHOLDS[lv - 1]} quantile`);
    // and the quantile is the baked one (top 1% starts at |r| = 1.142, i.e. about the neuron's p99)
    const want = quantileR(BAKED_QUANTILES, THRESHOLDS[lv - 1]);
    assert.ok(Math.abs(Math.log(x / want)) < 0.02, `level ${lv} starts at ${x}, baked ${want}`);
  }
  assert.equal(levelOf(0, 1, baked), 0);
  assert.equal(levelOf(1e6, 64, baked), 5);
});

test("crowding: level shares at 1, 4, 16, 64 neurons per cell match the single-neuron share (±10%), i.i.d. data", () => {
  const rand = rng(42), CELLS = 1_000_000, shares: number[][] = [];
  for (const k of [1, 4, 16, 64]) {
    const hits = new Array(6).fill(0);
    for (let c = 0; c < CELLS; c++) {
      // k i.i.d. draws |r| = Q(u); Q is increasing, so the strongest is Q(max u)
      let um = 0;
      for (let j = 0; j < k; j++) { const u = rand(); if (u > um) um = u; }
      hits[levelOf(quantileR(BAKED_QUANTILES, um), k, baked)]++;
    }
    const share = [1, 2, 3, 4, 5].map((lv) => hits.slice(lv).reduce((a, b) => a + b, 0) / CELLS);
    shares.push(share);
    for (let lv = 1; lv <= 5; lv++) {
      const got = share[lv - 1], want = top[lv - 1];
      assert.ok(Math.abs(got - want) <= 0.1 * want, `k=${k} level>=${lv}: ${(100 * got).toFixed(3)}% of cells, single-neuron share ${100 * want}%`);
    }
  }
  // without the correction, crowded cells light up far more often (what the correction is for)
  const rand2 = rng(7);
  let raw = 0;
  for (let c = 0; c < 100_000; c++) { let m = 0; for (let j = 0; j < 16; j++) m = Math.max(m, quantileR(BAKED_QUANTILES, rand2())); if (levelOf(m, 1, baked) >= 3) raw++; }
  assert.ok(raw / 100_000 > 0.1, `uncorrected k=16 top-1% share ${raw / 100_000}`);
});

test("live F: seeded from the baked table, then follows the stream", () => {
  const live = new LiveRarity();
  for (const [p, v] of BAKED_QUANTILES) if (p >= 0.1) assert.ok(Math.abs(rarityF(live.table(), v) - p) < 0.003 * Math.max(1, (1 - p) * 100), `seed F(${v})`);
  // a stream twice as loud as the seed: after enough tokens the 99th percentile moves to about twice the baked one
  const s = makeSynth(5, 0), r = new Float32Array(86016);
  const p99Before = quantileOf(live.table(), 0.99);
  for (let t = 0; t < 320; t++) { s.next(r); for (let i = 0; i < r.length; i++) r[i] *= 2; live.add(r); }
  const p99After = quantileOf(live.table(), 0.99);
  assert.ok(p99After / p99Before > 1.8 && p99After / p99Before < 2.1, `p99 ${p99Before} -> ${p99After}`);
  // zeros and non-finite values are skipped
  const n = live.total;
  live.add([0, NaN, Infinity, -Infinity]);
  assert.equal(live.total, n);
});

test("crowding with the live F built from a synthetic stream (the page's path), ±10% plus sampling noise", () => {
  const live = new LiveRarity(baked, 0), s = makeSynth(3, 0), pool = new Float32Array(86016 * 24);
  for (let t = 0; t < 24; t++) { const r = s.next(); pool.set(r, t * 86016); live.add(r); }
  const F = live.table();
  for (const k of [1, 4, 16, 64]) {
    const cells = Math.floor(pool.length / k), hits = new Array(6).fill(0);
    for (let c = 0; c < cells; c++) {
      let m = 0;
      for (let j = c * k; j < c * k + k; j++) m = Math.max(m, Math.abs(pool[j]));
      hits[levelOf(m, k, F)]++;
    }
    for (let lv = 1; lv <= 5; lv++) {
      const got = hits.slice(lv).reduce((a, b) => a + b, 0) / cells, want = top[lv - 1];
      const tol = 0.1 * want + 3 * Math.sqrt(want / cells);
      assert.ok(Math.abs(got - want) <= tol, `k=${k} level>=${lv}: ${got.toFixed(5)} vs ${want}`);
    }
  }
});
