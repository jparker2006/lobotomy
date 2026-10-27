// Shared loader for mockups: reads a recorded T2 frame stream (see server/server.py) from data/.
// Prototype code only. Not the product.

const F16 = new Float32Array(65536);
for (let h = 0; h < 65536; h++) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
  F16[h] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
}

export const L = 28, N = 3072;

export function parseFrames(buf) {
  const dv = new DataView(buf), dec = new TextDecoder(), out = [];
  for (let o = 0; o < buf.byteLength;) {
    const hl = dv.getUint32(o, true);
    const h = JSON.parse(dec.decode(new Uint8Array(buf, o + 4, hl)));
    const start = o + 4 + hl;
    out.push({ h, acts: h.n ? new Uint16Array(buf.slice(start, start + h.n)) : null });
    o = start + h.n;
  }
  return out;
}

// Loads a session. Each token gets `r`: Float32Array [L*N] of activation / p99 (signed).
export async function loadSession(name, base = "../data/") {
  const [cal, buf] = await Promise.all([
    fetch(base + "calibration.f32").then((r) => r.arrayBuffer()),
    fetch(base + name + ".bin").then((r) => r.arrayBuffer()),
  ]);
  const p99 = new Float32Array(cal);
  const frames = parseFrames(buf);
  const tokens = [];
  for (const f of frames) {
    if (!f.acts) continue;
    const r = new Float32Array(L * N);
    for (let k = 0; k < r.length; k++) r[k] = F16[f.acts[k]] / p99[k];
    let lit = 0;
    for (let k = 0; k < r.length; k++) if (Math.abs(r[k]) > 1) lit++;
    tokens.push({ i: f.h.i, kind: f.h.kind, text: f.h.text, r, lit });
  }
  const done = frames.find((f) => f.h.kind === "done")?.h;
  return { tokens, done, p99 };
}

// Glow with afterglow: max over recent tokens of |r| * 0.5^(age / halfLife). Keeps the sign of the winner.
export function glow(tokens, t, halfLife = 1.5, depth = 6) {
  const out = new Float32Array(L * N);
  for (let age = depth; age >= 0; age--) {
    const tok = tokens[t - age];
    if (!tok) continue;
    const w = Math.pow(0.5, age / halfLife);
    const r = tok.r;
    for (let k = 0; k < out.length; k++) {
      const v = r[k] * w;
      if (Math.abs(v) >= Math.abs(out[k])) out[k] = v;
    }
  }
  return out;
}

// Half-octave log ramp: 0 below 0.25 x p99, then one step per half octave, 9 at >= 4 x p99.
export const RAMP = " .:-=+*#%@";
export function level(x) {
  const a = Math.abs(x);
  if (!(a >= 0.25)) return 0;
  return Math.min(9, 1 + Math.floor(2 * Math.log2(a / 0.25)));
}
// Brightness 0..1 on the same log scale (0.25 -> 0, 4 -> 1).
export function bright(x) {
  const a = Math.abs(x);
  if (!(a > 0.25)) return 0;
  return Math.min(1, Math.log2(a / 0.25) / 4);
}

export function answerText(tokens, t) {
  let prompt = "", answer = "";
  for (const tok of tokens.slice(0, t + 1)) {
    if (tok.kind === "prompt") prompt += tok.text;
    else answer += tok.text;
  }
  return { prompt, answer };
}

// The user's prompt without the chat template.
export function userPrompt(tokens) {
  const p = tokens.filter((x) => x.kind === "prompt").map((x) => x.text).join("");
  const m = p.match(/<\|im_start\|>user\n([\s\S]*?)<\|im_end\|>/);
  return m ? m[1] : p;
}

export const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
export const rgb = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
