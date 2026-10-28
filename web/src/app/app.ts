// The app context: wires protocol -> session -> rarity -> deck, paces tokens onto the deck, runs the frame loop.
// Modules receive `App` in their init function and talk to each other only through app.store (see ARCHITECTURE.md).
import { createDeck, Deck } from "../render/deck.js";
import { glowAt } from "../render/cpu.js";
import { NEURONS } from "../render/rarity.js";
import { createClient, Client } from "../proto/client.js";
import type { Frame, DoneHeader } from "../proto/frames.js";
import { Session } from "../proto/session.js";
import { LiveRarity, quantileOf } from "../data/rarity.js";
import { Store } from "./state.js";
import { createClock, Clock } from "./clock.js";
import { createCameraRig, CameraRig } from "./camera.js";
import { applyState } from "./params.js";

/** Seconds between prompt tokens on the deck. The prompt runs as one forward pass, so it has no measured pace of its own. */
export const PROMPT_GAP = 1 / 30;

export interface App {
  readonly store: Store;
  readonly deck: Deck;
  readonly clock: Clock;
  readonly camera: CameraRig;
  readonly client: Client;
  readonly canvas: HTMLCanvasElement;
  /** The overlay root (#ui): modules add their DOM here. */
  readonly ui: HTMLElement;
  /** Live pooled F of |r| (seeded from the baked table). */
  readonly rarity: LiveRarity;
  /** Ask the server a question (replaces the current one). */
  ask(prompt: string): Promise<void>;
  /** Stop following the current generation. */
  cancel(): void;
  /** Start a session fed by the caller (replay, U6): then call ingest() with its frames. */
  beginSession(prompt: string, id?: string): Session;
  /** Feed frames into the current session; t = arrival time on the app clock (default now). */
  ingest(frames: Frame[], t?: number): void;
  /** Show the map at session token k (word replay); unpin() returns to live (or to the held final state once the answer is done). */
  pin(k: number): void;
  unpin(): void;
  /** Deterministic render: freeze the clock at t seconds and draw one frame (U6 clips). */
  renderAt(t: number): void;
  /** Start the animation loop (main.ts, after every module's init). */
  start(): void;
}

export async function createApp(canvas: HTMLCanvasElement, ui: HTMLElement, opts: { forceWebGL?: boolean } = {}): Promise<App> {
  const store = new Store(), clock = createClock(), client = createClient(), camera = createCameraRig(clock);
  const deck = await createDeck(canvas, {}, { forceWebGL: opts.forceWebGL });
  const rarity = new LiveRarity();
  deck.setRarityTable(rarity.table());

  let invP99 = new Float32Array(0), top1At = quantileOf(rarity.table(), 0.99);
  let shown = -1, lastDue = -Infinity, genTimes: number[] = [];
  // The trail's half-life is in tokens: once the answer is done and its last sweep is over, no more tokens pass, so the map
  // holds the state right after the last token (render/cpu.ts glowAt) instead of fading out in wall-clock time.
  let hold: Float32Array | null = null;

  // ---- state -> deck params, in one place ----
  const sync = () => applyState(deck.params, store.state);
  store.on("change", sync);
  sync();

  function beginSession(prompt: string, id = "local"): Session {
    const meta = store.state.meta;
    const s = new Session(id, prompt, clock.now(), meta?.layers, meta?.neurons);
    shown = -1; lastDue = -Infinity; genTimes = [];
    hold = null;
    deck.setGlow(null);
    store.set({ session: s, liveIndex: -1, pinnedIndex: null, elapsedSec: 0, status: "generating", error: null });
    store.emit("session", s);
    return s;
  }

  function ingest(frames: Frame[], t = clock.now()) {
    const s = store.state.session;
    if (!s) return;
    let added = false;
    for (const f of frames) {
      if (f.h.kind === "done") {
        const h = f.h as DoneHeader;
        s.done = h;
        store.set({ status: h.error ? "error" : "done", error: h.error ?? null, tokS: h.tok_s > 0 ? h.tok_s : store.state.tokS });
        store.emit("done", h);
        continue;
      }
      if (!f.acts) continue;
      const tok = s.add(f, t, invP99, top1At);
      if (tok.i >= 1) { rarity.add(tok.r); added = true; } // position 0 (attention sink) is drawn, never counted in F
      if (tok.kind === "gen") {
        genTimes.push(t);
        if (genTimes.length > 17) genTimes.shift();
        const n = genTimes.length, span = genTimes[n - 1] - genTimes[0];
        if (n >= 3 && span > 0) store.set({ tokS: (n - 1) / span });
      }
      store.emit("arrive", { index: s.tokens.length - 1 });
    }
    if (added) {
      const F = rarity.table();
      deck.setRarityTable(F);
      top1At = quantileOf(F, 0.99);
      store.emit("rarity", undefined);
    }
  }

  async function ask(prompt: string) {
    store.set({ status: "asking", error: null });
    let s: Session | null = null;
    try {
      await client.generate(prompt, (frames, id) => {
        if (!s) s = beginSession(prompt, id);
        ingest(frames);
      });
    } catch (e) {
      store.set({ status: "error", error: String((e as Error)?.message ?? e) });
    }
  }

  // ---- pacing: tokens go onto the deck in order, at most one per measured gap, never before they arrive ----
  function release(t: number) {
    const s = store.state.session;
    if (!s) return;
    let moved = false;
    while (shown + 1 < s.tokens.length) {
      const tok = s.tokens[shown + 1];
      const gap = tok.kind === "prompt" ? PROMPT_GAP : 0.95 / Math.max(1, store.state.tokS || 6);
      const due = Math.max(tok.t, lastDue + gap);
      if (due > t) break;
      deck.pushToken(tok.r, due);
      lastDue = due;
      shown++;
      moved = true;
      store.emit("token", { index: shown });
    }
    if (moved) store.set({ liveIndex: shown });
    const sweepSec = deck.params.sweepSec ?? 0.9 / deck.params.tokPerSec;
    if (!hold && s.done && shown >= 0 && shown === s.tokens.length - 1 && t >= lastDue + sweepSec) {
      hold = glowAt(s.tokens.map((x) => x.r), shown, store.state.settings.halfLifeTokens);
      if (store.state.pinnedIndex === null) deck.setGlow(hold);
    }
  }

  function pin(k: number) {
    const s = store.state.session;
    if (!s || k < 0 || k >= s.tokens.length) return;
    deck.setGlow(glowAt(s.tokens.map((x) => x.r), k, store.state.settings.halfLifeTokens));
    store.set({ pinnedIndex: k });
  }
  function unpin() { deck.setGlow(hold); store.set({ pinnedIndex: null }); }

  let lastT = 0;
  function step(t: number) {
    release(t);
    const st = store.state;
    if (st.session && st.status === "generating") {
      const e = Math.round((t - st.session.startedAt) * 10) / 10;
      if (e !== st.elapsedSec) store.set({ elapsedSec: e });
    }
    camera.update(t);
    deck.setView(camera.pose);
    deck.frame(t);
    store.emit("frame", { t, dt: t - lastT });
    lastT = t;
  }

  let running = false;
  function tick(ts: number) {
    requestAnimationFrame(tick);
    if (clock.manual) return;
    step(clock.tick(ts));
  }

  const app: App = {
    store, deck, clock, camera, client, canvas, ui, rarity,
    ask, cancel: () => client.abort(), beginSession, ingest, pin, unpin,
    renderAt(t) { clock.setManual(t); step(t); },
    start() { if (!running) { running = true; requestAnimationFrame(tick); } },
  };

  // meta + calibration: the HUD shows the real backend from /api/meta
  const [meta, cal] = await Promise.all([client.meta(), client.calibration()]);
  if (cal.length !== NEURONS || meta.layers * meta.neurons !== NEURONS) {
    throw new Error(`calibration has ${cal.length} values; the deck draws 28 x 3,072 = 86,016`);
  }
  invP99 = new Float32Array(cal.length);
  for (let k = 0; k < cal.length; k++) invP99[k] = 1 / cal[k];
  store.set({ meta, backend: meta.backend, calibration: cal, status: "ready" });
  return app;
}
