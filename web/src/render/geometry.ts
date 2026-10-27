// Deck geometry (design.md Q2): 28 sheets of 64 x 48 neurons standing side by side along x,
// L0 left -> L27 right; neuron n sits at row floor(n / 64), col n % 64 (index order until R9).
import { LAYERS, PER_LAYER } from "./rarity.js";

export const SHEET_COLS = 64, SHEET_ROWS = 48, GAP = 0.8, PITCH = 0.05;
export const HALF_Y = 1.2, HALF_Z = 1.6;

export function neuronPos(i: number, out: number[] | Float32Array = [0, 0, 0], o = 0) {
  const l = Math.floor(i / PER_LAYER), n = i % PER_LAYER, row = Math.floor(n / SHEET_COLS), col = n % SHEET_COLS;
  out[o] = (l - (LAYERS - 1) / 2) * GAP;
  out[o + 1] = ((SHEET_ROWS - 1) / 2 - row) * PITCH;
  out[o + 2] = (col - (SHEET_COLS - 1) / 2) * PITCH;
  return out;
}

export const layerX = (l: number) => (l - (LAYERS - 1) / 2) * GAP;

/** Line segments (pairs of xyz): a full frame for L0 and L27, a top rule for the rest. */
export function outlineSegments(): Float32Array {
  const s: number[] = [];
  for (let l = 0; l < LAYERS; l++) {
    const x = layerX(l), a = [x, HALF_Y, -HALF_Z], b = [x, HALF_Y, HALF_Z], c = [x, -HALF_Y, HALF_Z], d = [x, -HALF_Y, -HALF_Z];
    s.push(...a, ...b);
    if (l === 0 || l === LAYERS - 1) s.push(...b, ...c, ...c, ...d, ...d, ...a);
  }
  return new Float32Array(s);
}

/** Anchor for a layer's label (above the sheet's far top corner). */
export const labelPos = (l: number): [number, number, number] => [layerX(l), HALF_Y + 0.25, -HALF_Z];

/** Deck bounding box corners, for the view-only depth fade range. */
export function boxCorners(): number[][] {
  const X = layerX(LAYERS - 1), out: number[][] = [];
  for (const x of [-X, X]) for (const y of [-HALF_Y, HALF_Y]) for (const z of [-HALF_Z, HALF_Z]) out.push([x, y, z]);
  return out;
}

/** The overview of final-cinema-astrophage.png: front left, looking down the deck. center = principal point (fraction of the canvas). */
export interface View { eye: [number, number, number]; target: [number, number, number]; fov: number; center: [number, number] }
export const DEFAULT_VIEW: View = { eye: [-22.4, 8.6, 31.2], target: [0, -0.5, 0], fov: 21, center: [0.56, 0.43] };
