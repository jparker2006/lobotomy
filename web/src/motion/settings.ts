// Settings persistence (localStorage): load on init, save on change, reset. Pure functions over a Storage-like object so the
// tests can shim it. Unknown keys and bad values fall back to the defaults, so an old or hand-edited entry never breaks the page.
import { DEFAULT_SETTINGS, Settings } from "../app/state.js";

export const STORAGE_KEY = "lobotomy.settings.v1";
export type StorageLike = Pick<Storage, "getItem" | "setItem" | "removeItem">;

const copy = (s: Settings): Settings => ({ ...s, thresholds: [...s.thresholds] });

/** Merge a stored JSON string over the defaults, keeping only well-typed values. */
export function parseSettings(raw: string | null, defaults: Settings = DEFAULT_SETTINGS): Settings {
  const out = copy(defaults);
  let o: any;
  try { o = raw ? JSON.parse(raw) : null; } catch { o = null; }
  if (!o || typeof o !== "object") return out;
  for (const k of Object.keys(defaults) as (keyof Settings)[]) {
    const v = o[k], d = defaults[k];
    if (typeof d === "number" && typeof v === "number" && Number.isFinite(v)) (out as any)[k] = v;
    else if (typeof d === "boolean" && typeof v === "boolean") (out as any)[k] = v;
    else if (k === "thresholds" && validThresholds(v)) out.thresholds = [...v];
  }
  return out;
}

/** Five rarity cut-offs in (0, 1), strictly increasing. */
export function validThresholds(v: unknown): v is number[] {
  return Array.isArray(v) && v.length === DEFAULT_SETTINGS.thresholds.length &&
    v.every((x, i) => typeof x === "number" && x > 0 && x < 1 && (i === 0 || x > v[i - 1]));
}

export function loadSettings(storage: StorageLike | null, defaults: Settings = DEFAULT_SETTINGS): Settings {
  let raw: string | null = null;
  try { raw = storage?.getItem(STORAGE_KEY) ?? null; } catch { /* storage blocked */ }
  return parseSettings(raw, defaults);
}

export function saveSettings(storage: StorageLike | null, s: Settings) {
  try { storage?.setItem(STORAGE_KEY, JSON.stringify(s)); } catch { /* storage full or blocked */ }
}

export function clearSettings(storage: StorageLike | null) {
  try { storage?.removeItem(STORAGE_KEY); } catch { /* blocked */ }
}

/** The browser's localStorage, or null where it is missing or blocked. */
export function browserStorage(): StorageLike | null {
  try { return (globalThis as any).localStorage ?? null; } catch { return null; }
}
