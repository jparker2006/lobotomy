// U3 ui/: subtitle wrapping keeps every character's session token (for data-token) and drops whitespace at line ends.
// Run: npm test
import { test } from "node:test";
import assert from "node:assert/strict";
import { wrapChars } from "../src/ui/index.js";

const chars = (toks: string[]) => toks.flatMap((t, k) => Array.from(t, (c) => ({ c, k })));
const show = (lines: { c: string; k: number }[][]) => lines.map((l) => l.map((x) => x.c).join(""));

test("wrap: greedy by words, whitespace collapsed, tokens kept", () => {
  const lines = wrapChars(chars(["The", " deep", "\n\n", "blue", " waters", " glow"]), 10);
  assert.deepEqual(show(lines), ["The deep", "blue", "waters", "glow"]);
  assert.deepEqual(lines[3].map((x) => x.k), [5, 5, 5, 5]);
  assert.deepEqual(lines[0].map((x) => x.k), [0, 0, 0, 1, 1, 1, 1, 1]);
});

test("wrap: a word longer than the line breaks hard", () => {
  assert.deepEqual(show(wrapChars(chars(["a", " abcdefghij"]), 4)), ["a", "abcd", "efgh", "ij"]);
  assert.deepEqual(wrapChars([], 10), []);
});
