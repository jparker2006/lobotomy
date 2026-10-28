// The app store: every field the page and later tickets use, plus typed events. See web/ARCHITECTURE.md.
// Change a field with store.set({...}); every changed key fires `change:<key>`, then one `change` with all keys.
import type { Meta, DoneHeader } from "../proto/frames.js";
import type { Session } from "../proto/session.js";
import { THRESHOLDS } from "../render/rarity.js";

export interface NeuronRef { layer: number; neuron: number } // neuron = index within the layer, 0..3071

/** Every taste knob (design.md Principle 8). U3's tuning console edits these; app/params.ts maps them onto the deck. */
export interface Settings {
  halfLifeTokens: number;
  bloomStrength: number;
  /** Bloom target size / canvas (0.5 = half resolution). */
  bloomScale: number;
  hazeStrength: number;
  /** Map dot cell in CSS px. */
  cellW: number;
  cellH: number;
  sweepOn: boolean;
  driftOn: boolean;
  /** Rarity ladder: u thresholds for levels 1..5 (top 10 / 3 / 1 / 0.3 / 0.1%). */
  thresholds: number[];
  /** Ink hue in degrees; amber #FFB000 = 41.4. Hot ink keeps the same hue. */
  inkHue: number;
}

export const DEFAULT_SETTINGS: Settings = {
  halfLifeTokens: 2, bloomStrength: 1, bloomScale: 0.5, hazeStrength: 1.6, cellW: 5, cellH: 8,
  sweepOn: true, driftOn: true, thresholds: [...THRESHOLDS], inkHue: 41.4,
};

/** A measured power record (R12). The page shows a power figure only when one is attached. */
export interface PowerRecord { watts: number; joulesPerToken?: number; method: string; date: string }

export interface AppState {
  meta: Meta | null;
  /** The real backend, from /api/meta (or the replay file): "cpu", "fake", later "torq" / "npu". */
  backend: string;
  /** Per-neuron p99 of |activation| (/api/calibration), layer-major. */
  calibration: Float32Array | null;
  /** The current generation: every token's frame plus derived arrays (proto/session.ts). */
  session: Session | null;
  /** Index into session.tokens of the newest token on the deck (-1 = none yet). Lags arrival by the display pacing. */
  liveIndex: number;
  /** Word replay (U3/U5): the token the map is pinned to, or null for live. Set it through app.pin() / app.unpin(). */
  pinnedIndex: number | null;
  hover: NeuronRef | null;
  /** Lock-on (U5). While set, only its layer is drawn. */
  locked: NeuronRef | null;
  consoleOpen: boolean;
  settings: Settings;
  /** Measured generation pace (tokens/s) from arrival times; the server's figure once the generation is done. */
  tokS: number;
  /** Seconds since the current question was asked (stops when it is done). */
  elapsedSec: number;
  /** Non-null while replaying a recording (U6): what it says about itself. */
  replay: { backend: string; recordedAt: string; power?: PowerRecord } | null;
  /** Connection / generation status for the HUD. */
  status: "connecting" | "ready" | "asking" | "generating" | "done" | "error";
  error: string | null;
}

export const initialState = (): AppState => ({
  meta: null, backend: "", calibration: null, session: null, liveIndex: -1, pinnedIndex: null,
  hover: null, locked: null, consoleOpen: false, settings: { ...DEFAULT_SETTINGS, thresholds: [...DEFAULT_SETTINGS.thresholds] },
  tokS: 0, elapsedSec: 0, replay: null, status: "connecting", error: null,
});

/** Events besides field changes. */
export interface AppEvents {
  /** A new generation started (live or replay); session.tokens is empty. */
  session: Session;
  /** A token's frame arrived and is in session.tokens (not necessarily on the deck yet). */
  arrive: { index: number };
  /** A token reached the deck (liveIndex moved to it). */
  token: { index: number };
  /** The generation ended (done frame). */
  done: DoneHeader;
  /** After every drawn frame: t = clock seconds. DOM overlays that follow the deck (labels, reticles) update here. */
  frame: { t: number; dt: number };
  /** The rarity table changed (live F update). */
  rarity: void;
}

export type StoreEvents = AppEvents & { [K in keyof AppState as `change:${K}`]: AppState[K] } & { change: (keyof AppState)[] };

type Handler<T> = (payload: T) => void;

export class Store {
  readonly state: AppState = initialState();
  private handlers = new Map<string, Set<Handler<any>>>();

  get<K extends keyof AppState>(k: K): AppState[K] { return this.state[k]; }

  /** Shallow update; fires change events for keys whose value changed (by identity). */
  set(patch: Partial<AppState>) {
    const changed: (keyof AppState)[] = [];
    for (const k of Object.keys(patch) as (keyof AppState)[]) {
      if (this.state[k] === patch[k]) continue;
      (this.state as any)[k] = patch[k];
      changed.push(k);
    }
    for (const k of changed) this.emit(`change:${k}` as keyof StoreEvents, this.state[k] as any);
    if (changed.length) this.emit("change", changed);
  }

  /** Merges into settings (fires change:settings with a new object). */
  setSettings(patch: Partial<Settings>) { this.set({ settings: { ...this.state.settings, ...patch } }); }

  on<E extends keyof StoreEvents>(event: E, fn: Handler<StoreEvents[E]>): () => void {
    let s = this.handlers.get(event as string);
    if (!s) this.handlers.set(event as string, (s = new Set()));
    s.add(fn);
    return () => s!.delete(fn);
  }

  emit<E extends keyof StoreEvents>(event: E, payload: StoreEvents[E]) {
    for (const fn of this.handlers.get(event as string) ?? []) {
      try { fn(payload); } catch (e) { console.error(`handler for ${String(event)} failed`, e); }
    }
  }
}
