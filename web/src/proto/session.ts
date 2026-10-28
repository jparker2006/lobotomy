// One generation in memory: every token frame (kind, text, r = activation / p99, arrival time) plus per-token derived
// arrays, and the done frame. Live answers and replays (U6) fill it the same way.
import type { DoneHeader, Frame, TokenHeader } from "./frames.js";
import { toR } from "../data/f16.js";

export interface Token {
  /** Position in the context (0 = the attention sink, drawn but excluded from F). */
  i: number;
  kind: "prompt" | "gen";
  text: string;
  /** Signed activation / p99, 28 x 3,072 layer-major. */
  r: Float32Array;
  /** Arrival time on the app clock (seconds). */
  t: number;
  /** Neurons in their top 1% (|r| at or above F's 99th percentile when the token arrived). */
  top1: number;
  /** The same count per layer. */
  top1ByLayer: Uint16Array;
}

export class Session {
  readonly tokens: Token[] = [];
  done: DoneHeader | null = null;
  constructor(readonly id: string, readonly prompt: string, readonly startedAt: number, readonly layers = 28, readonly neurons = 3072) {}

  /** Adds a prompt/gen frame. top1At: the |r| of F's 99th percentile right now. */
  add(f: Frame, t: number, invP99: Float32Array, top1At: number): Token {
    const h = f.h as TokenHeader;
    const r = toR(f.acts!, invP99);
    const byLayer = new Uint16Array(this.layers);
    let top1 = 0;
    for (let l = 0, k = 0; l < this.layers; l++) {
      let c = 0;
      for (let n = 0; n < this.neurons; n++, k++) if (Math.abs(r[k]) >= top1At) c++;
      byLayer[l] = c;
      top1 += c;
    }
    const tok: Token = { i: h.i, kind: h.kind, text: h.text, r, t, top1, top1ByLayer: byLayer };
    this.tokens.push(tok);
    return tok;
  }

  /** Prompt text (chat template included) and answer text up to token index `upTo` (inclusive). */
  text(upTo = this.tokens.length - 1): { prompt: string; answer: string } {
    let prompt = "", answer = "";
    for (let j = 0; j <= upTo && j < this.tokens.length; j++) {
      const tok = this.tokens[j];
      if (tok.kind === "prompt") prompt += tok.text; else answer += tok.text;
    }
    if (upTo >= this.tokens.length - 1 && this.done) answer += this.done.text;
    return { prompt, answer };
  }

  get nPrompt() { let n = 0; for (const t of this.tokens) if (t.kind === "prompt") n++; return n; }
}
