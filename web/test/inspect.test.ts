// U5 inspect/: command parsing, rarity text, the per-word sparkline, "fires most on (this answer)" and the NeuronInfo slots.
// Run: npm test
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { firesSplit, fmtR, nowText, parseCommand, rarityText, slotRows, sparkValues } from "../src/inspect/model.js";
import { tableFromQuantiles } from "../src/data/rarity.js";
import { rarityF } from "../src/render/rarity.js";
import { cleanText, displayText, isTemplate } from "../src/data/display.js";

test("parse: L8 908 and its spellings", () => {
  for (const s of ["L8 908", "/L8 908", "/ l8n908", "8 908", "l8n908", "L08 N0908", "8:908", " l8  n908 ", "L8, 908", "L8 · N908"]) {
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

test("fires: split by sign, top-3% threshold, visible words only, hidden peak", () => {
  const hidden = (k: number) => k <= 1, display = (k: number) => toks[k].text, passes = (v: number) => Math.abs(v) >= 2;
  const fr = firesSplit(toks, 0, { passes, hidden, display });
  assert.deepEqual(fr.pos.map((f) => [f.k, f.v]), [[5, 4], [4, 2]]); // 'The' (+1) is below the threshold
  assert.deepEqual(fr.neg.map((f) => [f.k, f.v]), [[3, -4]]);
  assert.deepEqual(fr.hiddenPeak && [fr.hiddenPeak.k, fr.hiddenPeak.v], [1, 8]); // position 0 (the sink, 9) is left out
  assert.equal(fr.hiddenPeak!.text, ""); // a hidden token's markup is never carried
  const quiet = firesSplit(toks, 0, { passes: (v) => Math.abs(v) >= 5, hidden, display });
  assert.deepEqual([quiet.pos, quiet.neg], [[], []]);
  assert.equal(quiet.hiddenPeak!.v, 8); // the only big value: the readout shows just the peak line
  assert.equal(firesSplit(toks, 0, { passes, hidden: () => false, display }).hiddenPeak, null); // a visible peak is listed
  assert.deepEqual(firesSplit(toks, 0, { passes, hidden, display, upTo: 3 }).pos, []);
  assert.equal(firesSplit(toks, 0, { passes: () => true, hidden: () => false, display, n: 2 }).pos.length, 2);
});

// The real Qwen3 chat template, from a recorded CPU session (fixtures/inspect/ocean.json: every token's text plus r for two neurons)
const rec = JSON.parse(new TextDecoder().decode(readFileSync(new URL("../../../test/fixtures/inspect/ocean.json", import.meta.url)) as any)) as
  { tokens: { i: number; kind: "prompt" | "gen"; text: string; L8N908: number; L6N194: number }[] };
const sess = { tokens: rec.tokens };

test("display rule: the chat template is never printed, the question and the answer are", () => {
  const hidden = rec.tokens.map((_, k) => k).filter((k) => isTemplate(sess, k));
  assert.deepEqual(hidden.map((k) => rec.tokens[k].text), ["<|im_start|>", "user", "\n", "<|im_end|>", "\n", "<|im_start|>", "assistant", "\n", "<think>", "\n\n", "</think>", "\n\n"]);
  const text = (kind: string) => rec.tokens.map((t, k) => (t.kind === kind ? displayText(sess, k) : "")).join("");
  assert.equal(text("prompt"), "Describe the ocean at night in three sentences.");
  assert.ok(text("gen").startsWith("The ocean at night is a vast, shimmering expanse of water"));
  assert.ok(!/<\||\|>|think>|assistant/.test(text("prompt") + text("gen")));
  assert.equal(cleanText("bye<|im_end|>"), "bye");
  // any special token, mid-answer too, and a lone role-less newline stays when it is not next to the template
  const s2 = { tokens: [{ text: "Hi" }, { text: "\n" }, { text: "there" }, { text: "<|endoftext|>" }] };
  assert.deepEqual(s2.tokens.map((_, k) => isTemplate(s2, k)), [false, false, false, true]);
});

test("fires on the recorded session: L8 N908 fires on ocean words; L6 N194 peaks on a hidden setup token", () => {
  const F = tableFromQuantiles(), th = 0.97, passes = (v: number) => rarityF(F, Math.abs(v)) >= th;
  const run = (key: "L8N908" | "L6N194") => firesSplit(rec.tokens.map((t) => ({ ...t, r: Float32Array.of(t[key]) })), 0,
    { passes, hidden: (k) => isTemplate(sess, k), display: (k) => displayText(sess, k) });
  const ocean = run("L8N908");
  assert.deepEqual(ocean.pos.map((f) => f.text), [" ocean", " ocean", " sea", " water", " waters"]);
  assert.deepEqual(ocean.pos.map((f) => f.kind), ["gen", "prompt", "gen", "gen", "gen"]);
  assert.equal(ocean.hiddenPeak, null);
  const n194 = run("L6N194");
  assert.equal(n194.hiddenPeak!.text, "");
  assert.ok(n194.hiddenPeak!.v > 8 && isTemplate(sess, n194.hiddenPeak!.k));
  assert.ok(n194.pos.every((f) => !isTemplate(sess, f.k)) && n194.neg.every((f) => f.v < 0 && passes(f.v)));
});

test("sparkline skips template tokens", () => {
  const s = sparkValues(toks, 0, 3, 5, (k) => k === 4);
  assert.deepEqual(s.map((x) => x.k), [2, 3, 5]);
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
