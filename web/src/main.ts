// lobotomy v0.0.1 page: ask a question, stream the answer, draw every neuron of the latest frame.
// Protocol: see server/server.py.

type Meta = { layers: number; neurons: number; backend: string; model: string; max_tokens: number };
type Header = {
  i: number; kind: "prompt" | "gen" | "done"; text: string; n: number;
  tok_s?: number; n_prompt?: number; n_gen?: number; cancelled?: boolean; error?: string;
};
type Frame = { h: Header; acts: Uint16Array | null; bytes: number };

const TW = 64, TH = 48; // one tile per layer: 64 x 48 = 3,072 neurons

const el = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const statusEl = el<HTMLDivElement>("status");
const answerEl = el<HTMLDivElement>("answer");
const promptEl = el<HTMLInputElement>("prompt");

// float16 bits -> float32
const F16 = new Float32Array(65536);
for (let h = 0; h < 65536; h++) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
  F16[h] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
}

// brightness 0..255 -> RGBA (little-endian Uint32): warm for positive, cool for negative
const ramp = (x: number) => Math.round(255 * Math.min(1, Math.max(0, x)));
const rgba = (r: number, g: number, b: number) => (255 << 24) | (b << 16) | (g << 8) | r;
const WARM = new Uint32Array(256), COOL = new Uint32Array(256);
for (let i = 0; i < 256; i++) {
  const b = i / 255;
  WARM[i] = rgba(ramp(2 * b), ramp(0.9 * (2 * b - 0.8)), ramp(0.7 * (3 * b - 2)));
  COOL[i] = rgba(ramp(0.7 * (3 * b - 2)), ramp(0.9 * (2 * b - 0.8)), ramp(2 * b));
}

function parseFrames(buf: ArrayBuffer): Frame[] {
  const dv = new DataView(buf), dec = new TextDecoder(), out: Frame[] = [];
  for (let o = 0; o < buf.byteLength;) {
    const hl = dv.getUint32(o, true);
    const h: Header = JSON.parse(dec.decode(new Uint8Array(buf, o + 4, hl)));
    const start = o + 4 + hl;
    out.push({ h, acts: h.n ? new Uint16Array(buf.slice(start, start + h.n)) : null, bytes: 4 + hl + h.n });
    o = start + h.n;
  }
  return out;
}

async function main() {
  const meta: Meta = await (await fetch("/api/meta")).json();
  const cal = new Float32Array(await (await fetch("/api/calibration")).arrayBuffer());
  const inv = cal.map((c) => 255 / c);
  const N = meta.neurons;

  const tiles: { ctx: CanvasRenderingContext2D; img: ImageData; px: Uint32Array }[] = [];
  for (let l = 0; l < meta.layers; l++) {
    const div = document.createElement("div");
    div.className = "tile";
    const label = document.createElement("span");
    label.textContent = `L${l}`;
    const canvas = document.createElement("canvas");
    canvas.width = TW;
    canvas.height = TH;
    div.append(label, canvas);
    el("grid").append(div);
    const ctx = canvas.getContext("2d")!;
    const img = ctx.createImageData(TW, TH);
    tiles.push({ ctx, img, px: new Uint32Array(img.data.buffer) });
  }

  let pending: Uint16Array | null = null;
  function draw() {
    if (pending) {
      const a = pending;
      pending = null;
      for (let l = 0; l < meta.layers; l++) {
        const { ctx, img, px } = tiles[l];
        for (let n = 0, k = l * N; n < N; n++, k++) {
          const v = F16[a[k]];
          const b = Math.min(255, Math.abs(v) * inv[k]) | 0;
          px[n] = v >= 0 ? WARM[b] : COOL[b];
        }
        ctx.putImageData(img, 0, 0);
      }
    }
    requestAnimationFrame(draw);
  }
  requestAnimationFrame(draw);

  const head = `${meta.backend} · ${meta.model} · ${meta.layers}×${meta.neurons} neurons · ${meta.max_tokens} tokens max`;
  statusEl.textContent = head;
  let current = "";

  async function ask(prompt: string) {
    answerEl.textContent = "";
    const r = await fetch("/api/generate", { method: "POST", body: JSON.stringify({ prompt }) });
    const body = await r.json();
    if (!r.ok) {
      statusEl.textContent = `${head}\nerror: ${body.error}`;
      return;
    }
    const id: string = (current = body.id);
    let from = 0, nGen = 0, firstGen = 0, tokS = "–", bytes = 0, last = "";
    while (id === current) {
      let frames: Frame[];
      try {
        const res = await fetch(`/api/frames?id=${id}&from=${from}`);
        if (res.status === 404) return;
        if (!res.ok) throw new Error(String(res.status));
        frames = parseFrames(await res.arrayBuffer());
      } catch {
        await new Promise((ok) => setTimeout(ok, 1000));
        continue;
      }
      if (id !== current) return;
      for (const f of frames) {
        from = f.h.i + 1;
        if (f.acts) {
          pending = f.acts;
          bytes = f.bytes;
          last = `${f.h.kind} #${f.h.i} ${JSON.stringify(f.h.text)}`;
        }
        if (f.h.kind === "gen") {
          answerEl.textContent += f.h.text;
          if (nGen++ === 0) firstGen = performance.now();
          else tokS = ((nGen - 1) / ((performance.now() - firstGen) / 1000)).toFixed(1);
        }
        if (f.h.kind === "done") {
          answerEl.textContent += f.h.text;
          tokS = String(f.h.tok_s);
          last = `done: ${f.h.n_prompt} prompt + ${f.h.n_gen} generated tokens` +
            (f.h.cancelled ? " (cancelled)" : "") + (f.h.error ? ` error: ${f.h.error}` : "");
        }
      }
      statusEl.textContent = `${head}\n${tokS} tok/s · ${bytes.toLocaleString()} bytes/frame · ${last}`;
      if (frames.some((f) => f.h.kind === "done")) return;
    }
  }

  el<HTMLFormElement>("ask").addEventListener("submit", (e) => {
    e.preventDefault();
    if (promptEl.value.trim()) ask(promptEl.value);
  });
}

main().catch((e) => (statusEl.textContent = `error: ${e}`));
