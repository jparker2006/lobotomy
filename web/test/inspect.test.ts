// U5 inspect/: command parsing, rarity text, the per-word sparkline, "fires most on (this answer)" and the NeuronInfo slots.
// Run: npm test
import { test } from "node:test";
import assert from "node:assert/strict";
import { firesMostOn, fmtR, nowText, parseCommand, rarityText, slotRows, sparkValues } from "../src/inspect/model.js";
import { tableFromQuantiles } from "../src/data/rarity.js";

test("parse: L8 908 and its spellings", () => {
  for (const s of ["L8 908", "8 908", "l8n908", "L08 N0908", "8:908", " l8  n908 ", "L8, 908", "L8 · N908"]) {
    assert.deepEqual(parseCommand(s), { ref: { layer: 8, neuron: 908 } }, s);
  }
  assert.deepEqual(parseCommand("L27 3071"), { ref: { layer: 27, neuron: 3071 } });
  assert.deepEqual(parseCommand("0 0"), { ref: { layer: 0, neuron: 0 } });
});

test("parse: invalid input says so", () => {
  for (const s of ["", "ocean", "L8", "908", "L8 N", "L-1 5", "8 9 0"]) assert.ok("error" in parseCommand(s), s);
  const l = parseCommand("L28 5"), n = parseCommand("L8 3072");
  assert.ok("error" in l && /LAYER 28/.test(l.error));
  assert.ok("error" in n && /NEURON 3072/.test(n.error));
});

test("rarity text from F: top 0.1% at the ladder's finest step, finer percentages below 10%", () => {
  const F = tableFromQuantiles(); // baked CPU quantiles: |r| = 1.142 at p 0.99, 3.555 at p 0.999
  assert.equal(rarityText(4.7, F), "TOP 0.1%");
  assert.equal(rarityText(-4.7, F), "TOP 0.1%");
  assert.ok(/^TOP (1|1.0|0.9)%$/.test(rarityText(1.142, F)), rarityText(1.142, F));
  assert.equal(rarityText(0.434, F), "TOP 10%");
  assert.equal(rarityText(0.09941, F), "TOP 50%");
  assert.equal(rarityText(0, F), "TOP 100%");
  assert.equal(nowText(4.71, F), "+4.7× ITS P99 · TOP 0.1%");
  assert.equal(nowText(-0.25, F), `−0.3× ITS P99 · ${rarityText(0.25, F)}`);
  assert.equal(fmtR(123.4), "+123");
});

// three neurons; tokens: 2 prompt then 4 gen
const tok = (kind: "prompt" | "gen", text: string, r: number[]) => ({ kind, text, r: Float32Array.from(r) });
const toks = [
  tok("prompt", "<|im_start|>", [9, 0, 0]), tok("prompt", " ocean", [8, 0, 0]),
  tok("gen", "The", [1, 0, 0]), tok("gen", " sea", [-4, 0, 0]), tok("gen", " glows", [2, 0, 0]), tok("gen", ".", [4, 0, 0]),
];

test("sparkline: answer words only, height = |r| / max, sign kept, now and future marked", () => {
  const s = sparkValues(toks, 0, 3);
  assert.deepEqual(s.map((x) => x.k), [2, 3, 4, 5]);
  assert.deepEqual(s.map((x) => x.h), [0.25, 1, 0.5, 1]);
  assert.deepEqual(s.map((x) => x.v), [1, -4, 2, 4]);
  assert.deepEqual(s.map((x) => x.state), ["past", "now", "future", "future"]);
  assert.deepEqual(sparkValues(toks, 0, 4, 3).map((x) => x.k), [2, 3]); // up to the token on the deck
  assert.deepEqual(sparkValues(toks, 1, 2).map((x) => x.h), [0, 0, 0, 0]);
});

test("fires most on: this answer only, by |r|, ties keep word order", () => {
  assert.deepEqual(firesMostOn(toks, 0, 3), [{ k: 3, text: " sea", v: -4 }, { k: 5, text: ".", v: 4 }, { k: 4, text: " glows", v: 2 }]);
  assert.deepEqual(firesMostOn(toks, 0, 5, 3).map((f) => f.k), [3, 2]);
  assert.equal(firesMostOn(toks, 0, 5).length, 4);
});

test("slots: grey until a NeuronInfo record fills them", () => {
  const empty = slotRows();
  assert.deepEqual(empty.map((r) => [r.key, r.filled]), [["WORDS IT PUSHES", false], ["LABEL", false], ["REAL OR NOT", false], ["STEER", false]]);
  assert.equal(empty[0].value, "COMING v0.0.3");
  assert.equal(empty[1].value, "COMING v0.1.0");
  const full = slotRows({ lensWords: [" sea", " waves"], label: "ocean", real: { survives: true, p: 0.00123 }, safeRange: { neg: 2, pos: 6 } });
  assert.ok(full.every((r) => r.filled));
  assert.equal(full[0].value, '" sea" " waves"');
  assert.equal(full[1].value, "ocean");
  assert.equal(full[2].value, "SURVIVES FDR · p = 0.0012");
  assert.equal(full[3].value, "SAFE −2.0× … +6.0× P99");
  assert.equal(slotRows({ label: "ocean" })[0].filled, false);
});
