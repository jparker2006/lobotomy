// U2 proto/ + data/f16.ts: frame parser, float16 decode, r = activation / p99, session bookkeeping.
// Run: npm test
import { test } from "node:test";
import assert from "node:assert/strict";
import { parseFrames } from "../src/proto/frames.js";
import { F16, toR } from "../src/data/f16.js";
import { Session } from "../src/proto/session.js";

const enc = new TextEncoder();
function frame(h: object, payload = new Uint8Array(0)) {
  const hj = enc.encode(JSON.stringify({ ...h, n: payload.length })), out = new Uint8Array(4 + hj.length + payload.length);
  new DataView(out.buffer).setUint32(0, hj.length, true);
  out.set(hj, 4); out.set(payload, 4 + hj.length);
  return out;
}
const cat = (...a: Uint8Array[]) => { const o = new Uint8Array(a.reduce((n, x) => n + x.length, 0)); let k = 0; for (const x of a) { o.set(x, k); k += x.length; } return o.buffer; };

test("float16 table", () => {
  assert.equal(F16[0x3c00], 1); assert.equal(F16[0xc000], -2); assert.equal(F16[0x0000], 0);
  assert.equal(F16[0x7bff], 65504); assert.equal(F16[0x0001], 2 ** -24); assert.equal(F16[0x7c00], Infinity); assert.ok(Number.isNaN(F16[0x7e00]));
});

test("parser: header + float16 payload, done frame, multi-byte text, odd header length", () => {
  const acts = new Uint16Array([0x3c00, 0xc000, 0x3800]);
  const buf = cat(frame({ i: 0, kind: "prompt", text: "é" }, new Uint8Array(acts.buffer)), frame({ i: 1, kind: "gen", text: "x" }, new Uint8Array(acts.buffer)),
    frame({ i: 2, kind: "done", text: "", tok_s: 12.5, n_prompt: 1, n_gen: 1 }));
  const f = parseFrames(buf);
  assert.equal(f.length, 3);
  assert.equal(f[0].h.text, "é");
  assert.deepEqual(Array.from(f[1].acts!), Array.from(acts));
  assert.equal(f[2].acts, null);
  assert.equal((f[2].h as any).tok_s, 12.5);
  assert.deepEqual(Array.from(toR(f[0].acts!, new Float32Array([2, 0.5, 4]))), [2, -1, 2]);
  assert.throws(() => parseFrames(buf.slice(0, buf.byteLength - 20)));
});

test("session: tokens, top-1% counts per layer, answer text", () => {
  const L = 2, N = 4, s = new Session("x", "q", 0, L, N);
  const acts = new Uint16Array(L * N).fill(0x3800); // 0.5
  acts[1] = 0x4000; acts[6] = 0xc200; // 2, -3
  const p = new Uint8Array(acts.buffer);
  for (const h of [{ i: 0, kind: "prompt", text: "Q" }, { i: 1, kind: "gen", text: "Hi" }]) s.add(parseFrames(cat(frame(h, p)))[0], h.i, new Float32Array(L * N).fill(1), 1.5);
  s.done = { i: 2, kind: "done", text: "!", n: 0, tok_s: 1, n_prompt: 1, n_gen: 1 };
  assert.equal(s.tokens[1].top1, 2);
  assert.deepEqual(Array.from(s.tokens[1].top1ByLayer), [1, 1]);
  assert.deepEqual(s.text(), { prompt: "Q", answer: "Hi!" });
  assert.deepEqual(s.text(0), { prompt: "Q", answer: "" });
  assert.equal(s.nPrompt, 1);
});
