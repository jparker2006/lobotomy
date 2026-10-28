// U2's render extensions: glowAt (word replay), the per-layer visibility mask and the fixed glow state on the CPU stages.
// Run: npm test
import { test } from "node:test";
import assert from "node:assert/strict";
import { NEURONS, PER_LAYER } from "../src/render/rarity.js";
import { CpuStages, StageInputs, glowAt } from "../src/render/cpu.js";
import { outlineSegments } from "../src/render/geometry.js";

const ortho = [1 / 11, 0, 0, 0, 0, 1 / 1.25, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const inputs = (o: Partial<StageInputs> = {}): StageInputs =>
  ({ t: 0, tPrev: 0, hlSec: 1, sweepSec: 0, t0: new Float64Array(4).fill(-1e9), vp: ortho, cssW: 1100, cssH: 240, cellW: 5, cellH: 8, cols: 220, rows: 30, ...o });

test("glowAt: signed max-hold that halves every half-life (in tokens)", () => {
  const toks = [0.8, -2, 0.5].map((v) => new Float32Array(NEURONS).fill(v));
  assert.ok(Math.abs(glowAt(toks, 0, 2)[0] - 0.8) < 1e-6);
  assert.ok(Math.abs(glowAt(toks, 1, 2)[0] + 2) < 1e-6, "a stronger token replaces it, sign included");
  assert.ok(Math.abs(glowAt(toks, 2, 2)[0] + 2 * Math.SQRT1_2) < 1e-5, "one token later: x 0.5^(1/2)");
  assert.ok(Math.abs(glowAt(toks, 2, 0.5)[5] - 0.5) < 1e-6, "short half-life: the newest wins");
});

test("trail state on the CPU matches glowAt once every sweep is done", () => {
  const cpu = new CpuStages(), toks = [1, 2, 3].map((s) => { const r = new Float32Array(NEURONS); for (let i = 0; i < NEURONS; i++) r[i] = Math.sin(i * s) * s; return r; });
  const t0 = new Float64Array(4).fill(-1e9);
  let tPrev = -0.5;
  toks.forEach((r, k) => { cpu.tok.set(r, k * NEURONS); t0[k] = k * 0.5; const t = k * 0.5 + 0.001; cpu.trail(inputs({ t0, tPrev, t, hlSec: 1 })); tPrev = t; });
  cpu.trail(inputs({ t0, tPrev, t: 1.0, hlSec: 1 })); // 0.5 s/token, HL 1 s = 2 tokens
  const want = glowAt(toks, 2, 2);
  for (let i = 0; i < NEURONS; i += 997) assert.ok(Math.abs(cpu.glow[i] - want[i]) < 1e-3, `neuron ${i}: ${cpu.glow[i]} vs ${want[i]}`);
});

test("layer mask: hidden layers take no cells; a fixed glow state replaces the trail", () => {
  const cpu = new CpuStages(), mask = new Float32Array(28); mask[8] = 1;
  cpu.glow.fill(0.1);
  cpu.splat(inputs({ mask }));
  assert.equal(cpu.cnt.reduce((a, b) => a + b, 0), PER_LAYER, "only L8's 3,072 neurons");
  const out = new Float32Array(4 * 220 * 30);
  cpu.shown = new Float32Array(NEURONS).fill(-0.2);
  cpu.resolve(inputs(), { logLo: 0, logHi: 1, cdf: new Float32Array(4).fill(1) }, out);
  for (let c = 0; c < 220 * 30; c++) if (out[4 * c] >= 0) { assert.equal(out[4 * c + 1], -1); const l = Math.floor(out[4 * c + 3] / PER_LAYER); assert.equal(l, 8); }
  assert.equal(outlineSegments().length / 6, 28 + 2 * 3);
  assert.equal(outlineSegments(Array.from({ length: 28 }, (_, l) => l === 8)).length / 6, 4, "one full frame");
});
