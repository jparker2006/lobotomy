// Inspect (U5): the pure parts, so tests can check them without a DOM. Command parsing for `/`, rarity text, the per-word
// sparkline, "fires most on (this answer)" and the slots that later versions fill through a NeuronInfo record.
// Every value comes from the session's r = activation / p99 and the live pooled F (design.md "Rarity").
import { rarityF, type RarityTable } from "../render/rarity.js";
import type { NeuronRef } from "../app/state.js";

/**
 * Optional per-neuron data from later versions (design.md "Slots for later data"). Missing fields show as grey slots, so
 * v0.0.3 and v0.1.0 fill them by calling setNeuronInfo() (inspect/index.ts) without UI changes.
 */
export interface NeuronInfo {
  /** v0.0.3: logit-lens words this neuron pushes (trimmed vocab), strongest first. */
  lensWords?: string[];
  /** v0.1.0: a short label, plus an optional one-line description. */
  label?: string;
  description?: string;
  /** v0.1.0: does it survive the FDR test, with its p-value. */
  real?: { survives: boolean; p?: number };
  /** v0.1.0: safe steering range per sign, in × p99. */
  safeRange?: { neg: number; pos: number };
}

export const pad = (n: number, w: number) => String(n).padStart(w, "0");
export const refName = (r: NeuronRef) => `L${pad(r.layer, 2)} · N${pad(r.neuron, 4)}`;

/** Signed × p99 as the page prints it: "+4.7", "−0.3" (a real minus sign). */
export function fmtR(r: number): string {
  const a = Math.abs(r);
  return (r < 0 ? "−" : "+") + (a < 100 ? a.toFixed(1) : a < 1e4 ? a.toFixed(0) : a.toExponential(1));
}

/** A token's text as the readout shows it: quoted, with whitespace made visible (JSON escapes). */
export const quoteWord = (text: string) => JSON.stringify(text);

/** `/` command: "L8 908", "8 908", "l8n908", "L08 N0908", "8:908" (a typed leading "/" is fine). */
export function parseCommand(s: string, layers = 28, neurons = 3072): { ref: NeuronRef } | { error: string } {
  const m = /^\s*\/?\s*l?\s*(\d{1,3})\s*(?:[\s:,.·/]+\s*n?|n)\s*(\d{1,5})\s*$/i.exec(s);
  if (!m) return { error: `NOT A NEURON: TYPE L<LAYER> <NEURON>, E.G. L8 908` };
  const layer = Number(m[1]), neuron = Number(m[2]);
  if (layer >= layers) return { error: `NO LAYER ${layer}: LAYERS ARE 0–${layers - 1}` };
  if (neuron >= neurons) return { error: `NO NEURON ${neuron}: NEURONS ARE 0–${neurons - 1}` };
  return { ref: { layer, neuron } };
}

/** Rarity of one value under F: "TOP 0.1%" (rarer than the ladder's finest step reads as top 0.1%), "TOP 2.4%", "TOP 37%". */
export function rarityText(r: number, F: RarityTable): string {
  const top = (1 - rarityF(F, Math.abs(r))) * 100;
  if (top <= 0.1) return "TOP 0.1%";
  if (top < 1) return `TOP ${top.toFixed(1)}%`;
  if (top < 10) return `TOP ${top.toFixed(1).replace(/\.0$/, "")}%`;
  return `TOP ${Math.min(100, Math.round(top))}%`;
}

/** "+4.7× ITS P99 · TOP 0.1%" */
export const nowText = (r: number, F: RarityTable) => `${fmtR(r)}× ITS P99 · ${rarityText(r, F)}`;

interface Tok { kind: "prompt" | "gen"; text: string; r: Float32Array }

export interface Spark { k: number; v: number; h: number; state: "past" | "now" | "future" }

/**
 * The per-word values over this answer (generated tokens up to `upTo`, minus `skip`ped template tokens): one entry per word,
 * h = |v| / max |v| (0..1), sign in v. Words after `now` (a pinned word) are "future". Feeds the hover card's sparkline and
 * the readout's highlighted words.
 */
export function sparkValues(tokens: readonly Tok[], index: number, now: number, upTo = tokens.length - 1, skip: (k: number) => boolean = () => false): Spark[] {
  const out: Spark[] = [];
  let mx = 0;
  for (let k = 0; k <= upTo && k < tokens.length; k++) {
    if (tokens[k].kind !== "gen" || skip(k)) continue;
    const v = tokens[k].r[index];
    mx = Math.max(mx, Math.abs(v));
    out.push({ k, v, h: 0, state: k === now ? "now" : k > now ? "future" : "past" });
  }
  for (const s of out) s.h = mx > 0 ? Math.abs(s.v) / mx : 0;
  return out;
}

export interface Fire { k: number; text: string; v: number; kind: "prompt" | "gen" }
export interface Fires { pos: Fire[]; neg: Fire[]; hiddenPeak: Fire | null }

/**
 * What the neuron fires on in this session, split by sign (design.md Q8): up to n visible words (question + answer, by the
 * shared display rule) where it is positive / negative, strongest first, and only where `passes(v)` (the caller: top 3%
 * under the live F, ladder level >= 2). hiddenPeak: the session's strongest |v| when it sits on a hidden template token and
 * passes too. Position 0 (the attention sink) is left out, as it is from F. Ties keep token order.
 */
export function firesSplit(tokens: readonly Tok[], index: number, opts: {
  passes: (v: number) => boolean; hidden: (k: number) => boolean; display: (k: number) => string; n?: number; upTo?: number;
}): Fires {
  const n = opts.n ?? 5, upTo = Math.min(opts.upTo ?? tokens.length - 1, tokens.length - 1);
  const pos: Fire[] = [], neg: Fire[] = [];
  let peak: Fire | null = null, peakHidden = false;
  for (let k = 1; k <= upTo; k++) {
    const v = tokens[k].r[index], hidden = opts.hidden(k);
    const f: Fire = { k, text: hidden ? "" : opts.display(k), v, kind: tokens[k].kind };
    if (!peak || Math.abs(v) > Math.abs(peak.v)) { peak = f; peakHidden = hidden; }
    if (hidden || !f.text.trim() || !opts.passes(v)) continue;
    (v > 0 ? pos : neg).push(f);
  }
  const order = (a: Fire, b: Fire) => Math.abs(b.v) - Math.abs(a.v) || a.k - b.k;
  return { pos: pos.sort(order).slice(0, n), neg: neg.sort(order).slice(0, n), hiddenPeak: peak && peakHidden && opts.passes(peak.v) ? peak : null };
}

/** The readout's slot rows: real values where the record has them, else grey "COMING vX" (filled = false). */
export function slotRows(info?: NeuronInfo | null): { key: string; value: string; filled: boolean }[] {
  const i = info ?? {};
  return [
    i.lensWords?.length ? { key: "WORDS IT PUSHES", value: i.lensWords.slice(0, 6).map(quoteWord).join(" "), filled: true }
      : { key: "WORDS IT PUSHES", value: "COMING v0.0.3", filled: false },
    i.label ? { key: "LABEL", value: i.description ? `${i.label} · ${i.description}` : i.label, filled: true }
      : { key: "LABEL", value: "COMING v0.1.0", filled: false },
    i.real ? { key: "REAL OR NOT", value: `${i.real.survives ? "SURVIVES FDR" : "DOES NOT SURVIVE FDR"}${i.real.p !== undefined ? ` · p = ${i.real.p.toPrecision(2)}` : ""}`, filled: true }
      : { key: "REAL OR NOT", value: "COMING v0.1.0", filled: false },
    i.safeRange ? { key: "STEER", value: `SAFE ${fmtR(-Math.abs(i.safeRange.neg))}× … ${fmtR(Math.abs(i.safeRange.pos))}× P99`, filled: true }
      : { key: "STEER", value: "━━━━○━━━━  v0.1.0", filled: false },
  ];
}
