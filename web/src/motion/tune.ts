// The tuning console (`~`, design.md Principle 8): every taste knob in store.settings, changed live through store.setSettings,
// plus a live fps / p95 readout from the `frame` events, measured the way the bench does (motion/fps.ts).
import type { App } from "../app/app.js";
import { DEFAULT_SETTINGS, Settings } from "../app/state.js";
import { inkOf, hotOf } from "../app/params.js";
import { FrameMeter } from "./fps.js";
import { StorageLike, saveSettings } from "./settings.js";

type NumKey = "halfLifeTokens" | "bloomStrength" | "bloomScale" | "hazeStrength" | "cellW" | "cellH" | "inkHue";
interface Knob { key: NumKey; label: string; min: number; max: number; step: number; fmt: (v: number) => string }

const KNOBS: Knob[] = [
  { key: "halfLifeTokens", label: "trail half-life", min: 0.25, max: 8, step: 0.25, fmt: (v) => `${v.toFixed(2)} tok` },
  { key: "bloomStrength", label: "bloom strength", min: 0, max: 3, step: 0.05, fmt: (v) => v.toFixed(2) },
  { key: "bloomScale", label: "bloom resolution", min: 0.25, max: 1, step: 0.05, fmt: (v) => `${v.toFixed(2)}×` },
  { key: "hazeStrength", label: "haze strength", min: 0, max: 4, step: 0.05, fmt: (v) => v.toFixed(2) },
  { key: "cellW", label: "cell width", min: 3, max: 12, step: 1, fmt: (v) => `${v} px` },
  { key: "cellH", label: "cell height", min: 4, max: 20, step: 1, fmt: (v) => `${v} px` },
  { key: "inkHue", label: "ink hue", min: 0, max: 360, step: 0.1, fmt: (v) => `${v.toFixed(1)}°` },
];
/** Ink presets: design.md's amber (#FFB000) and "astrophage red". */
export const INK_PRESETS: { name: string; hue: number }[] = [{ name: "amber", hue: 41.4 }, { name: "astrophage red", hue: 8 }];
const TOGGLES: { key: "sweepOn" | "driftOn"; label: string }[] = [{ key: "sweepOn", label: "sweep" }, { key: "driftOn", label: "drift" }];

/** Key handlers ignore typing (ARCHITECTURE.md shared rules). */
export function isTyping(e: Event): boolean {
  const typing = (n: unknown) => n instanceof HTMLElement && (n.matches("input, textarea") || n.isContentEditable);
  return typing(e.target) || typing(document.activeElement);
}

const topPct = (u: number) => +((1 - u) * 100).toPrecision(3);
const css = (c: number[]) => `rgb(${c.map((x) => Math.round(255 * Math.min(1, Math.max(0, x)))).join(", ")})`;

export function initTuning(app: App, storage: StorageLike | null) {
  const { store, deck } = app;
  const panel = document.createElement("div");
  panel.className = "mot-tune interactive";
  panel.hidden = true;
  panel.innerHTML =
    `<div class="mot-head"><b>TUNING</b><span class="mot-dim">~ to close</span><button class="mot-btn" data-reset>reset</button></div>` +
    `<div class="mot-fps">fps —</div><div class="mot-fps2 mot-dim">&nbsp;</div><div class="mot-rule"></div>` +
    KNOBS.filter((k) => k.key !== "inkHue").map(row).join("") +
    TOGGLES.map((t) => `<div class="mot-row"><span class="mot-label">${t.label}</span><button class="mot-btn mot-toggle" data-toggle="${t.key}"></button><span class="mot-val mot-dim" data-info="${t.key}"></span></div>`).join("") +
    `<div class="mot-row"><span class="mot-label">ladder top %</span><span class="mot-ladder">${DEFAULT_SETTINGS.thresholds.map((_, i) => `<input data-th="${i}" inputmode="decimal" spellcheck="false">`).join("")}</span></div>` +
    row(KNOBS.find((k) => k.key === "inkHue")!) +
    `<div class="mot-row"><span class="mot-label"></span><span class="mot-presets">${INK_PRESETS.map((p) => `<button class="mot-btn" data-hue="${p.hue}">${p.name}</button>`).join("")}</span></div>` +
    `<div class="mot-foot mot-dim">saved in this browser (localStorage)</div>`;
  app.ui.append(panel);

  function row(k: Knob) {
    return `<label class="mot-row"><span class="mot-label">${k.label}</span><input type="range" data-knob="${k.key}" min="${k.min}" max="${k.max}" step="${k.step}"><span class="mot-val" data-val="${k.key}"></span></label>`;
  }
  const q = <T extends Element>(s: string) => panel.querySelector(s) as T;
  const qa = <T extends Element>(s: string) => [...panel.querySelectorAll(s)] as T[];
  const sliders = qa<HTMLInputElement>("input[data-knob]"), ladder = qa<HTMLInputElement>("input[data-th]");
  const fps = q<HTMLElement>(".mot-fps"), fps2 = q<HTMLElement>(".mot-fps2");

  // ---- controls -> store ----
  for (const s of sliders) {
    s.addEventListener("input", () => store.setSettings({ [s.dataset.knob as NumKey]: +s.value } as Partial<Settings>));
    s.addEventListener("change", () => s.blur()); // so `~` works again right after a drag
  }
  for (const b of qa<HTMLButtonElement>("[data-toggle]")) {
    const k = b.dataset.toggle as "sweepOn" | "driftOn";
    b.addEventListener("click", (e) => { e.preventDefault(); store.setSettings({ [k]: !store.state.settings[k] } as Partial<Settings>); b.blur(); });
  }
  for (const b of qa<HTMLButtonElement>("[data-hue]")) b.addEventListener("click", () => { store.setSettings({ inkHue: +b.dataset.hue! }); b.blur(); });
  q<HTMLButtonElement>("[data-reset]").addEventListener("click", (e) => {
    store.setSettings({ ...DEFAULT_SETTINGS, thresholds: [...DEFAULT_SETTINGS.thresholds] });
    saveSettings(storage, store.state.settings);
    (e.target as HTMLElement).blur();
  });
  const applyLadder = () => {
    const u = ladder.map((i) => 1 - parseFloat(i.value) / 100);
    const ok = u.every((x, i) => Number.isFinite(x) && x > 0 && x < 1 && (i === 0 || x > u[i - 1]));
    ladder.forEach((i) => i.classList.toggle("mot-bad", !ok));
    if (ok) store.setSettings({ thresholds: u });
  };
  for (const i of ladder) {
    i.addEventListener("change", applyLadder);
    i.addEventListener("keydown", (e) => { if (e.key === "Enter") { applyLadder(); i.blur(); } else if (e.key === "Escape") i.blur(); });
  }

  // ---- store -> controls, and the UI ink follows the deck's ink ----
  function render() {
    const s = store.state.settings;
    for (const sl of sliders) {
      const k = sl.dataset.knob as NumKey;
      if (document.activeElement !== sl) sl.value = String(s[k]);
      q<HTMLElement>(`[data-val="${k}"]`).textContent = KNOBS.find((x) => x.key === k)!.fmt(s[k]);
    }
    for (const b of qa<HTMLButtonElement>("[data-toggle]")) {
      const on = s[b.dataset.toggle as "sweepOn" | "driftOn"];
      b.textContent = on ? "[x] on " : "[ ] off";
      b.classList.toggle("mot-on", on);
    }
    s.thresholds.forEach((u, i) => { if (document.activeElement !== ladder[i]) { ladder[i].value = String(topPct(u)); ladder[i].classList.remove("mot-bad"); } });
    for (const b of qa<HTMLButtonElement>("[data-hue]")) b.classList.toggle("mot-on", Math.abs(+b.dataset.hue! - s.inkHue) < 0.05);
    const root = document.documentElement.style;
    root.setProperty("--ink", css(inkOf(s.inkHue)));
    root.setProperty("--hot", css(hotOf(s.inkHue)));
    info();
  }
  /** What the toggles do right now: the sweep time from the measured pace (deck.params, set by app/params.ts), the drift state. */
  function info() {
    const s = store.state.settings, p = deck.params;
    q<HTMLElement>(`[data-info="sweepOn"]`).textContent = s.sweepOn ? `${(p.sweepSec ?? 0.9 / p.tokPerSec).toFixed(2)} s · ${p.tokPerSec.toFixed(1)} tok/s` : "instant";
    q<HTMLElement>(`[data-info="driftOn"]`).textContent = !s.driftOn ? "" : store.state.locked ? "paused (locked)" : app.camera.drifting ? "drifting" : "after 8 s idle";
  }
  store.on("change:settings", render);
  render();

  // ---- fps / p95 from the frame events (rAF intervals; the deterministic clip mode is not measured) ----
  const meter = new FrameMeter();
  let lastShow = 0, wasManual = false;
  store.on("frame", ({ t, dt }) => {
    if (app.clock.manual) { wasManual = true; return; }
    if (wasManual) { wasManual = false; meter.reset(); return; }
    meter.push(dt * 1000);
    if (panel.hidden || t - lastShow < 0.25) return;
    lastShow = t;
    info();
    const st = meter.stats();
    if (st.frames < 10) { fps.textContent = "fps —"; fps2.textContent = "measuring…"; return; }
    fps.innerHTML = `<b>${st.fps.toFixed(1)}</b> fps · p95 <b>${st.p95Vsync.toFixed(1)}</b> ms`;
    fps2.textContent = `raw p95 ${st.p95.toFixed(1)} · dropped ${st.droppedPct.toFixed(1)}% · >33 ms ${st.over33Pct.toFixed(1)}% · ${st.frames} frames`;
  });
  // a hidden tab gets no frames (the bench flags such runs): start the window again when it comes back
  document.addEventListener("visibilitychange", () => meter.reset());

  // ---- `~` toggles the panel ----
  window.addEventListener("keydown", (e) => {
    if (e.key !== "~" && e.key !== "`") return;
    if (isTyping(e) || e.metaKey || e.ctrlKey || e.altKey) return;
    e.preventDefault();
    panel.hidden = !panel.hidden;
  });
}
