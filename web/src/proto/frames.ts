// The T2 frame protocol (server/server.py docstring): u32 LE header length, UTF-8 JSON header, then n payload bytes
// (float16 LE [layers, neurons] for prompt/gen frames, nothing for done).

export interface Meta { layers: number; neurons: number; backend: string; model: string; max_tokens: number }

export interface TokenHeader { i: number; kind: "prompt" | "gen"; text: string; n: number }
export interface DoneHeader {
  i: number; kind: "done"; text: string; n: 0;
  tok_s: number; n_prompt: number; n_gen: number; cancelled?: boolean; error?: string;
}
export type FrameHeader = TokenHeader | DoneHeader;

/** acts: raw float16 bits, layer-major (null for done). bytes: the frame's size on the wire. */
export interface Frame { h: FrameHeader; acts: Uint16Array | null; bytes: number }

const dec = new TextDecoder();

/** Parses a buffer of whole frames (one long-poll response or a recorded file). */
export function parseFrames(buf: ArrayBuffer): Frame[] {
  const dv = new DataView(buf), out: Frame[] = [];
  for (let o = 0; o + 4 <= buf.byteLength;) {
    const hl = dv.getUint32(o, true);
    const h = JSON.parse(dec.decode(new Uint8Array(buf, o + 4, hl))) as FrameHeader;
    const start = o + 4 + hl;
    if (start + h.n > buf.byteLength) throw new Error(`frame ${h.i}: truncated (${buf.byteLength - start} of ${h.n} bytes)`);
    out.push({ h, acts: h.n ? new Uint16Array(buf.slice(start, start + h.n)) : null, bytes: 4 + hl + h.n });
    o = start + h.n;
  }
  return out;
}
