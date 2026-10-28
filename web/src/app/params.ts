// The one place app state maps onto deck.params (render/deck.ts DeckParams).
import type { DeckParams, RGB } from "../render/deck.js";
import type { AppState } from "./state.js";

/** HSL (h degrees, s and l 0..1) -> display RGB 0..1. */
export function hsl(h: number, s: number, l: number): RGB {
  const k = (n: number) => (n + h / 30) % 12, a = s * Math.min(l, 1 - l);
  const f = (n: number) => l - a * Math.max(-1, Math.min(k(n) - 3, 9 - k(n), 1));
  return [f(0), f(8), f(4)];
}

/** Ink and hot ink for a hue: at 41.4 degrees these are design.md's #FFB000 and #FFE4A0. */
export const inkOf = (hue: number): RGB => hsl(hue, 1, 0.5);
export const hotOf = (hue: number): RGB => hsl(hue + 1.6, 1, 0.814);

export function applyState(p: DeckParams, s: AppState) {
  const k = s.settings;
  p.halfLifeTokens = k.halfLifeTokens;
  p.bloomStrength = k.bloomStrength;
  p.bloomScale = k.bloomScale;
  p.hazeStrength = k.hazeStrength;
  p.cellW = k.cellW;
  p.cellH = k.cellH;
  p.sweepSec = k.sweepOn ? null : 0;
  p.thresholds = [...k.thresholds];
  p.ink = inkOf(k.inkHue);
  p.hot = hotOf(k.inkHue);
  // lock-on draws only the locked layer
  p.visibleLayers = s.locked ? Array.from({ length: 28 }, (_, l) => l === s.locked!.layer) : null;
  // the measured pace sets the sweep (0.9 / tok_s) and the half-life in seconds; clamp so a stall never freezes trails
  if (s.tokS > 0) p.tokPerSec = Math.min(60, Math.max(2, s.tokS));
}
