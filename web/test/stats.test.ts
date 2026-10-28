// U3 data/stats.ts: the counter, tape and per-layer numbers, replayed from a recorded CPU stand-in session and checked
// against numpy (test/fixtures/stats_fixture.py wrote test/fixtures/stats/expected.json from the same file).
// Run: npm test
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { parseFrames } from "../src/proto/frames.js";
import { Session } from "../src/proto/session.js";
import { LiveRarity } from "../src/data/rarity.js";
import { countTop1, rarest, tape, top1Threshold } from "../src/data/stats.js";

// compiled to dist/test-build/test/: the fixture stays in the source tree
const dir = new URL("../../../test/fixtures/stats/", import.meta.url);
const bytes = (f: string) => { const b = readFileSync(new URL(f, dir)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const text = (f: string) => new TextDecoder().decode(bytes(f));

interface Expected {
  tokens: { i: number; kind: string; text: string; top1At: number; top1: number; top1ByLayer: number[]; rarest: { layer: number; neuron: number; r: number }[] }[];
  tape: { heights: number[]; scale: number };
}

test("replay a recorded session: counter, per-layer counts, rarest and tape match numpy exactly", () => {
  const want = JSON.parse(text("expected.json")) as Expected, meta = JSON.parse(text("meta.json"));
  const cal = new Float32Array(bytes("calibration.f32")), invP99 = new Float32Array(cal.length);
  for (let k = 0; k < cal.length; k++) invP99[k] = 1 / cal[k];
  const frames = parseFrames(bytes("session.bin"));
  // the page's path (app/app.ts ingest), one frame per arrival: count against F as it was, then add the token to F
  const s = new Session("fixture", "Say hi.", 0, meta.layers, meta.neurons), F = new LiveRarity();
  for (const f of frames) {
    if (!f.acts) continue;
    const at = top1Threshold(F.table()), tok = s.add(f, 0, invP99, at), w = want.tokens[s.tokens.length - 1];
    assert.equal(tok.i, w.i);
    assert.equal(tok.text, w.text);
    assert.ok(Math.abs(at / w.top1At - 1) < 1e-12, `token ${tok.i}: threshold ${at} vs ${w.top1At}`);
    const c = countTop1(tok.r, at, meta.layers, meta.neurons);
    assert.equal(c.total, w.top1, `token ${tok.i}: counter`);
    assert.deepEqual(Array.from(c.byLayer), w.top1ByLayer, `token ${tok.i}: per layer`);
    // the session's own counts (what the UI reads) agree
    assert.equal(tok.top1, c.total);
    assert.deepEqual(Array.from(tok.top1ByLayer), w.top1ByLayer);
    assert.deepEqual(rarest(tok.r, 6, meta.neurons), w.rarest, `token ${tok.i}: rarest`);
    if (tok.i >= 1) F.add(tok.r);
  }
  assert.equal(s.tokens.length, want.tokens.length);
  assert.deepEqual(tape(s.tokens), want.tape);
});

test("rarest: by |r|, sign kept, ties to the lower index, zeros skipped", () => {
  const r = new Float32Array([0, -3, 2, 3, 0.5, -0, 1]);
  assert.deepEqual(rarest(r, 3, 4), [{ layer: 0, neuron: 1, r: -3 }, { layer: 0, neuron: 3, r: 3 }, { layer: 0, neuron: 2, r: 2 }]);
  assert.deepEqual(rarest(new Float32Array([0, 0, 1]), 6, 2), [{ layer: 1, neuron: 0, r: 1 }]);
});

test("tape: scale ignores position 0 (the attention sink)", () => {
  assert.deepEqual(tape([{ i: 0, top1: 900 }, { i: 1, top1: 10 }, { i: 2, top1: 30 }]), { heights: [900, 10, 30], scale: 30 });
  assert.deepEqual(tape([]), { heights: [], scale: 1 });
});
