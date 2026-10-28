// Inspect (U5, #23): hover card, lock-on, `/` jump, word replay clicks (design.md Q8, mockups q8-hover / q8-lock).
// Hover: deck.pick() returns the neuron that set the dot under the pointer (store.hover); a light card follows it.
// Click: lock on (store.locked -> app/params.ts draws only that layer; camera.faceLayer), a reticle, a leader line and a docked
// readout. `/` opens a command line (L8 908). Clicking a [data-token] word or tape bar pins the map to that token (app.pin),
// `space` returns to live, `Esc` unlocks. Every value is the session's activation / p99 and the live pooled F.
import type { App } from "../app/app.js";
import type { NeuronRef } from "../app/state.js";
import { loadCSS } from "../app/css.js";
import { inkOf, hotOf } from "../app/params.js";
import { typing } from "../ui/index.js";
import { neuronPos } from "../render/geometry.js";
import { PER_LAYER } from "../render/rarity.js";
import {
  type NeuronInfo, type Spark, firesMostOn, fmtR, nowText, parseCommand, quoteWord, refName, slotRows, sparkValues,
} from "./model.js";

export type { NeuronInfo } from "./model.js";

const CW = 9, CH = 16, READOUT_COLS = 48;

// ---- later data (v0.0.3 lens words, v0.1.0 labels / FDR / safe range): a provider fills the slots, no UI change ----
let infoSource: ((ref: NeuronRef) => NeuronInfo | null | undefined) | null = null;
let refresh: (() => void) | null = null;
/** Registers the per-neuron record source (null = none: the slots stay grey). */
export function setNeuronInfo(fn: ((ref: NeuronRef) => NeuronInfo | null | undefined) | null) { infoSource = fn; refresh?.(); }

const same = (a: NeuronRef | null, b: NeuronRef | null) => a === b || (!!a && !!b && a.layer === b.layer && a.neuron === b.neuron);
const rgb = (c: number[]) => `rgb(${c.map((x) => Math.round(255 * Math.max(0, Math.min(1, x)))).join(",")})`;

export function initInspect(app: App): void {
  loadCSS(new URL("./inspect.css", import.meta.url));
  const { store, deck, camera, canvas } = app;
  const el = <K extends keyof HTMLElementTagNameMap = "div">(cls: string, parent: HTMLElement, tag?: K, text = "") => {
    const e = document.createElement(tag ?? "div") as HTMLElementTagNameMap[K];
    e.className = cls;
    if (text) e.textContent = text;
    parent.append(e);
    return e;
  };
  const root = el("insp-root", app.ui);
  const reticle = (cls: string) => { const r = el(`insp-ret ${cls}`, root); for (const c of ["tl", "tr", "bl", "br"]) el(`insp-c insp-${c}`, r, "i"); return r; };
  const hoverRet = reticle("insp-ret-hover"), lockRet = reticle("insp-ret-lock");
  const leader = el("insp-leader", root);

  // a neuron panel: the hover card and the docked readout share their rows
  function panel(cls: string, full: boolean) {
    const box = el(cls, root);
    const head = el("insp-head", box);
    const title = el("insp-title", head, "b");
    if (full) el("insp-esc", head, "span", "[ESC] UNLOCK");
    const row = (k: string) => { const r = el("insp-row", box); el("insp-k", r, "span", k); return el("insp-v", r, "span"); };
    const now = row("NOW"), word = row("WORD");
    now.classList.add("insp-now");
    el("insp-sub", box, "div", "THIS ANSWER, WORD BY WORD");
    const spark = el("insp-spark", box, "canvas");
    const p = { box, title, now, word, spark, fires: null as HTMLElement | null, slots: null as HTMLElement | null };
    if (full) {
      el("insp-sub", box, "div", "FIRES MOST ON (THIS ANSWER)");
      p.fires = el("insp-fires", box);
      const pos = el("insp-row insp-pos", box);
      el("insp-k", pos, "span", "POSITION");
      el("insp-v", pos, "span", "INDEX ORDER (ARBITRARY UNTIL THE v0.1.0 LAYOUT)");
      p.slots = el("insp-slots", box);
    } else el("insp-foot", box, "div", "CLICK TO LOCK ON");
    return p;
  }
  const card = panel("insp-card", false), readout = panel("insp-readout", true);
  const caption = el("insp-caption", root);
  const cmd = el("insp-cmd", root);
  el("insp-cmd-mark", cmd, "span", "/");
  const cmdInput = el("insp-cmd-input interactive", cmd, "input");
  Object.assign(cmdInput, { autocomplete: "off", spellcheck: false, placeholder: "L8 908" });
  const cmdMsg = el("insp-cmd-msg", cmd, "span");
  const HINT = "— TYPE A NEURON TO JUMP TO IT · LATER: SEARCH LABELS";

  // ---- what is "now": the pinned word, else the newest word on the deck (as U3's subtitles) ----
  const nowIndex = () => store.state.pinnedIndex ?? store.state.liveIndex;
  const neurons = () => store.state.meta?.neurons ?? PER_LAYER;
  const globalIndex = (r: NeuronRef) => r.layer * neurons() + r.neuron;

  function drawSpark(cv: HTMLCanvasElement, sparks: Spark[]) {
    const dpr = devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
    if (!w || !h) return;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    const g = cv.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const st = store.state.settings, ink = rgb(inkOf(st.inkHue)), hot = rgb(hotOf(st.inkHue));
    g.fillStyle = ink; g.globalAlpha = 0.18; g.fillRect(0, h - 1, w, 1); // baseline
    const n = sparks.length;
    if (!n) return;
    const bw = Math.min(9, w / n), gap = bw >= 3 ? 1 : 0;
    sparks.forEach((s, j) => {
      const x = j * bw, bh = Math.max(1, Math.round(s.h * (h - 1))), y = h - bh;
      g.globalAlpha = s.state === "now" ? 1 : s.state === "future" ? 0.15 : s.v >= 0 ? 0.75 : 0.5;
      const c = s.state === "now" ? hot : ink;
      if (s.v >= 0 || bw - gap < 3) { g.fillStyle = c; g.fillRect(x, y, Math.max(1, bw - gap), bh); }
      else { g.strokeStyle = c; g.lineWidth = 1; g.strokeRect(x + 0.5, y + 0.5, bw - gap - 1, Math.max(0, bh - 1)); } // hollow = negative
    });
  }

  function fill(p: ReturnType<typeof panel>, ref: NeuronRef) {
    const s = store.state, sess = s.session, k = nowIndex(), K = globalIndex(ref), info = infoSource?.(ref) ?? null;
    const tok = sess && k >= 0 ? sess.tokens[k] : null;
    p.title.textContent = refName(ref) + (info?.label ? ` · ${info.label.toUpperCase()}` : "");
    if (tok) {
      const v = tok.r[K];
      p.now.textContent = nowText(v, app.rarity.table());
      p.now.dataset.r = String(v);
      p.word.textContent = quoteWord(tok.text);
    } else { p.now.textContent = "—"; delete p.now.dataset.r; p.word.textContent = "—"; }
    const toks = sess ? sess.tokens : [], upTo = Math.max(s.liveIndex, k);
    drawSpark(p.spark, sparkValues(toks, K, k, upTo));
    if (p.fires) {
      p.fires.textContent = "";
      for (const f of firesMostOn(toks, K, 5, upTo)) {
        const e = el("insp-fire interactive", p.fires, "span");
        e.dataset.token = String(f.k); // click = pin that word (word replay)
        e.dataset.r = String(f.v);
        el("insp-fw", e, "span", quoteWord(f.text));
        el("insp-fv", e, "span", `${fmtR(f.v)}×`);
      }
      if (!p.fires.childElementCount) el("insp-dim", p.fires, "span", "NO WORDS YET");
    }
    if (p.slots) {
      p.slots.textContent = "";
      for (const r of slotRows(info)) {
        const e = el(`insp-row${r.filled ? "" : " insp-slot"}`, p.slots);
        el("insp-k", e, "span", r.key);
        el("insp-v", e, "span", r.value);
      }
    }
  }

  // ---- render: content on state changes, positions on every frame ----
  function render() {
    const s = store.state, hover = s.hover, locked = s.locked;
    const showCard = !!hover && !same(hover, locked);
    card.box.style.display = showCard ? "" : "none";
    if (showCard) fill(card, hover!);
    readout.box.style.display = locked ? "" : "none";
    if (locked) fill(readout, locked);
    root.dataset.locked = locked ? `${locked.layer}:${locked.neuron}` : "";
    caption.textContent = locked ? `LAYER ${locked.layer} · FACE-ON · OTHER LAYERS HIDDEN` : "";
    root.classList.toggle("insp-locked", !!locked);
    root.classList.toggle("insp-console", s.consoleOpen);
    renderCmd();
    place();
  }
  refresh = render;

  function renderCmd() {
    const open = cmd.classList.contains("insp-open");
    cmd.style.display = open || store.state.locked ? "" : "none";
    if (!open) { cmdMsg.textContent = `L8 908  ${HINT}`; cmdMsg.classList.remove("insp-err"); }
  }

  function at(ref: NeuronRef) { return deck.project(neuronPos(globalIndex(ref)) as [number, number, number]); }
  function put(e: HTMLElement, p: { x: number; y: number } | null) {
    e.style.display = p ? "" : "none";
    if (p) e.style.transform = `translate(${Math.round(p.x)}px, ${Math.round(p.y)}px)`;
  }

  function place() {
    const s = store.state, W = innerWidth, H = innerHeight;
    const hp = s.hover ? at(s.hover) : null, lp = s.locked ? at(s.locked) : null;
    put(hoverRet, s.hover && !same(s.hover, s.locked) ? hp : null);
    put(lockRet, lp);
    if (card.box.style.display !== "none" && hp) {
      const w = card.box.offsetWidth, h = card.box.offsetHeight;
      let x = hp.x + 3 * CW, y = hp.y - h - CH;
      if (x + w > W - 2 * CW) x = hp.x - 3 * CW - w;
      x = Math.max(2 * CW, x); y = Math.max(4 * CH, Math.min(H - h - 2 * CH, y));
      card.box.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
    }
    leader.style.display = "none";
    if (s.locked && !s.consoleOpen) {
      // the readout docks on the right, its title row level with the reticle (kept clear of the HUD and the subtitles)
      const h = readout.box.offsetHeight, y0 = lp ? lp.y - 2.5 * CH : H * 0.3;
      const top = Math.max(5 * CH, Math.min(H - h - 11 * CH, y0));
      readout.box.style.top = `${Math.round(top)}px`;
      const left = readout.box.getBoundingClientRect().left;
      if (lp && lp.x + 4 * CW < left - CW) {
        leader.style.display = "";
        leader.style.transform = `translate(${Math.round(lp.x + 4 * CW)}px, ${Math.round(lp.y)}px)`;
        leader.style.width = `${Math.round(left - CW - (lp.x + 4 * CW))}px`;
      }
    } else readout.box.style.top = "";
  }

  // ---- camera framing: the readout is a side panel; the rig combines it with other owners' panels (U3's Console) ----
  function frameShare() {
    const W = Math.max(1, innerWidth);
    return { l: 0, r: store.state.locked ? ((READOUT_COLS + 6) * CW) / W : 0 };
  }

  function lock(ref: NeuronRef) {
    const was = store.state.locked;
    store.set({ locked: ref });
    if (!was || was.layer !== ref.layer) { void camera.setFrame(frameShare(), 0, "inspect"); void camera.faceLayer(ref.layer); }
  }
  function unlock() {
    if (!store.state.locked) return;
    store.set({ locked: null });
    void camera.setFrame(frameShare(), 0, "inspect");
    void camera.overview();
  }

  // ---- hover: one pick in flight at a time, always for the newest pointer position ----
  let ptr: { x: number; y: number } | null = null, picking = false, again = false, hoverSeq = 0;
  const accept = (p: { level: number; layer: number } | null) => !!p && (p.level >= 1 || (!!store.state.locked && store.state.locked.layer === p.layer));
  async function pickLoop() {
    if (picking) { again = true; return; }
    picking = true;
    try {
      do {
        again = false;
        const p0 = ptr, seq = hoverSeq;
        if (!p0) break;
        const p = await deck.pick(p0.x, p0.y);
        if (seq !== hoverSeq) continue;
        const ref = accept(p) ? { layer: p!.layer, neuron: p!.index } : null;
        if (!same(ref, store.state.hover)) store.set({ hover: ref });
      } while (again);
    } finally { picking = false; }
  }
  const local = (e: MouseEvent) => { const r = canvas.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
  canvas.addEventListener("pointermove", (e) => {
    if (e.buttons) { ptr = null; hoverSeq++; if (store.state.hover) store.set({ hover: null }); return; } // dragging: no card
    ptr = local(e);
    void pickLoop();
  });
  canvas.addEventListener("pointerleave", () => { ptr = null; hoverSeq++; if (store.state.hover) store.set({ hover: null }); });

  // ---- click = lock on (U4 swallows the click that ends a drag); double-click (U4: overview) also unlocks ----
  let clickSeq = 0;
  canvas.addEventListener("click", async (e) => {
    const seq = ++clickSeq, q = local(e), p = await deck.pick(q.x, q.y);
    if (seq !== clickSeq || !accept(p)) return;
    lock({ layer: p!.layer, neuron: p!.index });
  });
  canvas.addEventListener("dblclick", () => { clickSeq++; unlock(); });

  // ---- word replay: any [data-token] in the overlay (U3's subtitles, transcript, tape; our fires-most-on words) ----
  app.ui.addEventListener("click", (e) => {
    const t = (e.target as Element | null)?.closest?.("[data-token]") as HTMLElement | null;
    if (!t || !app.ui.contains(t)) return;
    const k = Number(t.dataset.token);
    if (Number.isInteger(k)) app.pin(k);
  });

  // ---- `/` command line ----
  function openCmd() {
    cmd.classList.add("insp-open");
    cmdInput.value = "";
    cmdMsg.textContent = "";
    renderCmd();
    cmdInput.focus();
  }
  function closeCmd() { cmd.classList.remove("insp-open"); cmdInput.blur(); renderCmd(); }
  cmdInput.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { e.preventDefault(); closeCmd(); return; }
    if (e.key !== "Enter") return;
    e.preventDefault();
    const m = store.state.meta, r = parseCommand(cmdInput.value, m?.layers ?? 28, m?.neurons ?? PER_LAYER);
    if ("error" in r) { cmdMsg.textContent = r.error; cmdMsg.classList.add("insp-err"); return; }
    closeCmd();
    lock(r.ref);
  });
  cmdInput.addEventListener("input", () => { cmdMsg.textContent = ""; cmdMsg.classList.remove("insp-err"); });
  cmdInput.addEventListener("blur", () => { if (cmd.classList.contains("insp-open")) closeCmd(); });

  addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey || typing(e)) return;
    if (e.key === "/") { e.preventDefault(); openCmd(); }
    else if (e.key === "Escape" && store.state.locked) { e.preventDefault(); unlock(); }
    else if (e.key === " " && store.state.pinnedIndex !== null) { e.preventDefault(); app.unpin(); }
  });

  // ---- wiring ----
  const syncInk = () => {
    const h = store.state.settings.inkHue;
    root.style.setProperty("--insp-ink", rgb(inkOf(h)));
    root.style.setProperty("--insp-hot", rgb(hotOf(h)));
  };
  store.on("change", (keys) => {
    if (keys.includes("settings")) syncInk();
    if (keys.some((k) => ["hover", "locked", "liveIndex", "pinnedIndex", "session", "consoleOpen", "settings", "meta"].includes(k))) render();
  });
  store.on("rarity", render);
  store.on("session", () => store.set({ hover: null }));
  store.on("frame", place);
  addEventListener("resize", render);
  syncInk();
  render();
}
