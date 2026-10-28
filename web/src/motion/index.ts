// Motion (U4): orbit / zoom / pan input on the deck, idle drift (settings.driftOn, never while locked), settings persistence
// and the tuning console (`~`). The camera math lives in app/camera.ts (U4 owns its internals). The sweep (0.9 / tok_s from
// measured arrivals) and the trail half-life reach the deck through store.settings -> app/params.ts.
import type { App } from "../app/app.js";
import { loadCSS } from "../app/css.js";
import { browserStorage, loadSettings, saveSettings } from "./settings.js";
import { initTuning } from "./tune.js";

/** Orbit speed: radians per CSS px of drag (about a half turn per screen height). */
const ORBIT_PER_PX = 0.006;
/** A press that moves farther than this is a drag, and its click is swallowed (so U5's click-to-lock ignores drags). */
const DRAG_PX = 4;

export function initMotion(app: App): void {
  loadCSS(new URL("./motion.css", import.meta.url));
  const { store, camera, canvas } = app, storage = browserStorage();

  // ---- settings: load on init, save on change ----
  store.setSettings(loadSettings(storage));
  store.on("change:settings", (s) => saveSettings(storage, s));

  // ---- idle drift: only with driftOn and nothing locked ----
  const syncDrift = () => { camera.driftEnabled = store.state.settings.driftOn && !store.state.locked; };
  store.on("change", (keys) => { if (keys.includes("settings") || keys.includes("locked")) syncDrift(); });
  syncDrift();

  // ---- orbit (drag), pan (right-drag, shift-drag or middle-drag), zoom (wheel, toward the cursor), double-click = overview ----
  let drag: { id: number; x: number; y: number; mode: "orbit" | "pan"; moved: number } | null = null, swallowClick = false;
  canvas.addEventListener("pointerdown", (e) => {
    if (drag) return;
    camera.onUserInput();
    const mode = e.button === 2 || e.button === 1 || e.shiftKey ? "pan" : "orbit";
    drag = { id: e.pointerId, x: e.clientX, y: e.clientY, mode, moved: 0 };
    swallowClick = false;
    canvas.setPointerCapture?.(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag || e.pointerId !== drag.id) { camera.markActive(); return; } // hover counts as input for the drift, not for tweens
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.x = e.clientX; drag.y = e.clientY; drag.moved += Math.abs(dx) + Math.abs(dy);
    if (drag.mode === "orbit") camera.orbit(-dx * ORBIT_PER_PX, dy * ORBIT_PER_PX);
    else camera.pan(dx, dy, canvas.clientHeight);
  });
  const end = (e: PointerEvent) => {
    if (!drag || e.pointerId !== drag.id) return;
    swallowClick = drag.moved > DRAG_PX;
    drag = null;
    canvas.releasePointerCapture?.(e.pointerId);
  };
  canvas.addEventListener("pointerup", end);
  canvas.addEventListener("pointercancel", end);
  // capture phase on window runs before any listener on the canvas
  window.addEventListener("click", (e) => { if (swallowClick && e.target === canvas) { e.stopImmediatePropagation(); swallowClick = false; } }, true);
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const px = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? canvas.clientHeight : 1);
    // ctrl+wheel is a trackpad pinch: small deltas, so a stronger gain
    const factor = Math.exp(Math.max(-200, Math.min(200, px)) * (e.ctrlKey ? 0.01 : 0.0015));
    const r = canvas.getBoundingClientRect();
    camera.zoomAt(factor, (e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height, r.width / Math.max(1, r.height));
  }, { passive: false });
  canvas.addEventListener("dblclick", () => { camera.onUserInput(); void camera.overview(); });

  initTuning(app, storage);
}
