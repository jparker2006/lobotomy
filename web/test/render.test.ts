// U1 render module: rarity ladder, crowding correction, synthetic stream, splat key, CPU stages (the GPU kernels' reference).
// Run: npm test
import { test } from "node:test";
import assert from "node:assert/strict";
import { NEURONS, PER_LAYER, THRESHOLDS, rarityF, levelOf, packKey, INDEX_MASK, rarityTableFromSamples } from "../src/render/rarity.js";
import { makeSynth, recordedRarityTable, RECORDED_QUANTILES } from "../src/render/synth.js";
import { CpuStages, StageInputs } from "../src/render/cpu.js";

const table = recordedRarityTable();

test("F matches the recorded quantiles", () => {
  for (const [p, v] of RECORDED_QUANTILES) assert.ok(Math.abs(rarityF(table, v) - p) < 0.003, `F(${v}) = ${rarityF(table, v)}, want ${p}`);
  assert.equal(rarityF(table, 0), 0);
  let prev = 0;
  for (let a = 1e-3; a < 1e3; a *= 1.07) { const f = rarityF(table, a); assert.ok(f >= prev); prev = f; }
});

test("synthetic stream: distribution, signs, ~1.3% above 1, lit set changes", () => {
  const s = makeSynth(7), toks = [0, 1, 2].map(() => new Float32Array(s.next()));
  const abs = Array.from(toks[0], Math.abs).sort((a, b) => a - b);
  for (const [p, v] of RECORDED_QUANTILES) { const q = abs[Math.floor(p * abs.length)]; assert.ok(Math.abs(Math.log(q / v)) < 0.08, `p${p}: ${q} vs ${v}`); }
  const above = abs.filter((a) => a > 1).length / NEURONS;
  assert.ok(above > 0.010 && above < 0.016, `above 1: ${above}`);
  const pos = toks[0].filter((x) => x > 0).length / NEURONS;
  assert.ok(Math.abs(pos - 0.5) < 0.01);
  let lit = 0, stay = 0;
  for (let i = 0; i < NEURONS; i++) if (Math.abs(toks[1][i]) > 1) { lit++; if (Math.abs(toks[2][i]) > 1) stay++; }
  assert.ok(stay / lit < 0.5, `share of the lit set that stays lit: ${stay / lit}`);
});

test("crowding: level shares match the single-neuron shares (±10%) at 1, 4, 16, 64 neurons per cell", () => {
  const s = makeSynth(3, 0), pool = new Float32Array(NEURONS * 24);
  for (let t = 0; t < 24; t++) pool.set(s.next(), t * NEURONS);
  const F = rarityTableFromSamples(pool, 512); // the pooled F of this stream, as the page builds it
  const want = THRESHOLDS.map((th) => 1 - th);
  for (const k of [1, 4, 16, 64]) {
    const cells = Math.floor(pool.length / k), hits = new Array(6).fill(0);
    for (let c = 0; c < cells; c++) {
      let m = 0;
      for (let j = c * k; j < c * k + k; j++) m = Math.max(m, Math.abs(pool[j]));
      hits[levelOf(Math.pow(rarityF(F, m), k))]++;
    }
    for (let lv = 1; lv <= 5; lv++) {
      const got = hits.slice(lv).reduce((a, b) => a + b, 0) / cells, n = got * cells;
      const tol = 0.1 * want[lv - 1] + 3 * Math.sqrt(want[lv - 1] / cells); // 10% plus sampling noise
      assert.ok(Math.abs(got - want[lv - 1]) <= tol, `k=${k} level>=${lv}: ${got.toFixed(5)} vs ${want[lv - 1]} (${n} cells)`);
    }
  }
});

test("splat key keeps the strongest neuron and its index", () => {
  assert.ok(packKey(2, 5) > packKey(1.9, 80000));
  assert.ok(packKey(-3, 1) > packKey(2.5, 2));
  assert.equal(packKey(0.4, 86015) & INDEX_MASK, 86015);
  assert.equal(packKey(0, 7) & INDEX_MASK, 7);
});

const ortho = (sx: number, sy: number) => [sx, 0, 0, 0, 0, sy, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]; // column-major, w = 1
function inputs(o: Partial<StageInputs> = {}): StageInputs {
  return { t: 0, tPrev: 0, hlSec: 1, sweepSec: 0, t0: new Float64Array(4).fill(-1e9), vp: ortho(1 / 11, 1 / 1.25), cssW: 1100, cssH: 240, cellW: 5, cellH: 8, cols: 220, rows: 30, ...o };
}

test("trail + sweep: layer order and pace, max-hold that halves every half-life", () => {
  const cpu = new CpuStages();
  cpu.tok.fill(1, 0, NEURONS); // slot 0: every neuron at +1
  const t0 = new Float64Array(4).fill(-1e9); t0[0] = 0;
  const base = { t0, sweepSec: 0.28, hlSec: 0.5 };
  cpu.trail(inputs({ ...base, tPrev: -0.01, t: 0.005 }));
  assert.ok(cpu.glow[0] > 0.99 && cpu.glow[27 * PER_LAYER] === 0, "L0 lit first, L27 not yet");
  cpu.trail(inputs({ ...base, tPrev: 0.005, t: 0.27 }));
  assert.ok(cpu.glow[27 * PER_LAYER] > 0.99, "L27 reached by 0.27 s (27/28 of the sweep)");
  cpu.trail(inputs({ ...base, tPrev: 0.27, t: 0.77 }));
  assert.ok(Math.abs(cpu.glow[0] - Math.pow(0.5, 0.77 / 0.5)) < 1e-3, `L0 halves every half-life: ${cpu.glow[0]}`);
  cpu.tok.fill(-0.2, 0, NEURONS); t0[0] = 0.8;
  cpu.trail(inputs({ ...base, tPrev: 0.77, t: 1.2 }));
  assert.ok(cpu.glow[0] > 0, "a weaker token does not replace a stronger glow");
  cpu.tok.fill(-3, 0, NEURONS); t0[0] = 1.25;
  cpu.trail(inputs({ ...base, tPrev: 1.2, t: 1.6 }));
  assert.ok(Math.abs(cpu.glow[0] + 3 * Math.pow(0.5, 0.35 / 0.5)) < 1e-3, "a stronger token replaces it, sign included");
});

test("splat + resolve: every neuron counted, the winner is the cell's max |glow|", () => {
  const cpu = new CpuStages(), u = inputs();
  cpu.glow.set(makeSynth(11).next());
  cpu.splat(u);
  assert.equal(cpu.cnt.reduce((a, b) => a + b, 0), NEURONS);
  const best = new Float32Array(u.cols * u.rows), P = cpu.pos;
  for (let i = 0; i < NEURONS; i++) {
    const sx = (P[3 * i] * u.vp[0] * 0.5 + 0.5) * u.cssW, sy = (0.5 - P[3 * i + 1] * u.vp[5] * 0.5) * u.cssH;
    const c = Math.floor(sy / 8) * u.cols + Math.floor(sx / 5);
    best[c] = Math.max(best[c], Math.abs(cpu.glow[i]));
  }
  const cnt = cpu.cnt.slice(), out = new Float32Array(4 * u.cols * u.rows);
  cpu.resolve(u, table, out);
  let occupied = 0;
  for (let c = 0; c < cnt.length; c++) {
    if (!cnt[c]) { assert.equal(out[4 * c], -1); continue; }
    occupied++;
    const w = out[4 * c + 3], g = cpu.glow[w];
    assert.ok(Math.abs(Math.abs(g) - best[c]) < 1e-3 * best[c] + 1e-6, `cell ${c}`);
    assert.equal(out[4 * c + 1], g < 0 ? -1 : 1);
    assert.equal(out[4 * c], levelOf(Math.pow(rarityF(table, g), cnt[c])));
  }
  assert.ok(occupied > 100);
  assert.equal(cpu.cnt.reduce((a, b) => a + b, 0), 0, "resolve clears the grid");
});
