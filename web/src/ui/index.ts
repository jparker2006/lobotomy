// Minimal Cinema UI (U2): HUD with the real backend, token n/256 and tok/s, the top-1% counter, legend, layer labels,
// subtitles and the prompt line. U3 replaces this module (tape, console, tuning console, full Cinema layout).
import type { App } from "../app/app.js";
import { loadCSS } from "../app/css.js";
import { provenance } from "../app/provenance.js";

const LABELLED = [0, 9, 18, 27];
const pad = (n: number, w: number) => String(n).padStart(w, "0");

export function initUI(app: App) {
  loadCSS(new URL("./ui.css", import.meta.url));
  const { store, deck } = app;
  const el = (cls: string, tag = "div") => { const e = document.createElement(tag); e.className = cls; app.ui.append(e); return e; };

  const hud = el("u2-hud"), legend = el("u2-legend"), counter = el("u2-counter"), stats = el("u2-stats");
  const sub = el("u2-sub"), subText = document.createElement("div");
  sub.append(subText);
  const status = el("u2-status");
  const form = el("u2-prompt interactive", "form") as HTMLFormElement;
  form.innerHTML = `<span>&gt;</span><input autocomplete="off" spellcheck="false" placeholder="ask Qwen something, then Enter">`;
  const input = form.querySelector("input")!;
  input.value = "Describe the ocean at night in three sentences.";
  const labels = LABELLED.map((l) => { const e = el("u2-label"); e.textContent = `L${pad(l, 2)}`; return { l, e }; });

  legend.textContent = "· TOP 10%   ∙ 3%   • 1%   ● 0.3%   ◉ 0.1%   FOR THAT NEURON · FILLED = POSITIVE, HOLLOW = NEGATIVE";

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (q) app.ask(q);
  });

  function renderHud() {
    const s = store.state, m = s.meta;
    hud.innerHTML = `<b>LOBOTOMY</b>   ${m ? m.model.split("/").pop()!.toUpperCase() : ""} · BACKEND: ${provenance(s)} · ${m ? `${m.layers} × ${m.neurons.toLocaleString("en-US")}` : "28 × 3,072"} NEURONS`;
  }
  function renderStats() {
    const s = store.state, sess = s.session, k = s.pinnedIndex ?? s.liveIndex, tok = sess && k >= 0 ? sess.tokens[k] : null;
    counter.textContent = tok ? String(tok.top1) : "—";
    const max = s.meta?.max_tokens ?? 256;
    stats.textContent = `NEURONS IN THEIR TOP 1% · TOKEN ${pad(tok ? tok.i + 1 : 0, 3)}/${max} · ${s.tokS ? s.tokS.toFixed(1) : "—"} TOK/S`;
  }
  function renderSub() {
    const s = store.state, sess = s.session;
    subText.textContent = "";
    if (!sess) return;
    const k = s.pinnedIndex ?? s.liveIndex;
    const cur = k >= 0 ? sess.tokens[k] : null;
    let before = "";
    for (let j = 0; j < k; j++) if (sess.tokens[j].kind === "gen") before += sess.tokens[j].text;
    const curText = cur?.kind === "gen" ? cur.text : "";
    const tail = k >= sess.tokens.length - 1 && sess.done && s.pinnedIndex === null ? sess.done.text : "";
    const hot = document.createElement("span"); hot.className = "hot"; hot.textContent = curText + tail;
    const cursor = document.createElement("span"); cursor.className = "cursor"; cursor.textContent = " ";
    subText.append(before.slice(-400), hot);
    if (!sess.done || s.pinnedIndex !== null || k < sess.tokens.length - 1) subText.append(cursor);
  }
  function renderStatus() {
    const s = store.state;
    status.textContent = s.status === "error" ? `error: ${s.error}` : s.status === "connecting" ? "connecting…" : s.status === "asking" ? "thinking…"
      : s.pinnedIndex !== null ? `PINNED AT TOKEN ${pad(s.session!.tokens[s.pinnedIndex].i + 1, 3)}` : "";
  }

  store.on("change", (keys) => {
    if (keys.includes("meta") || keys.includes("backend") || keys.includes("replay")) renderHud();
    if (keys.some((k) => k === "liveIndex" || k === "pinnedIndex" || k === "tokS" || k === "session")) { renderStats(); renderSub(); }
    if (keys.some((k) => k === "status" || k === "error" || k === "pinnedIndex")) renderStatus();
  });
  store.on("done", () => renderSub());
  store.on("frame", () => {
    for (const { l, e } of labels) {
      const p = deck.layerLabel(l), vis = deck.params.visibleLayers;
      e.style.display = p && (!vis || vis[l]) ? "" : "none";
      if (p) e.style.transform = `translate(${p.x}px, ${p.y}px) translate(-50%, -100%)`;
    }
  });
  renderHud(); renderStats(); renderStatus();
  input.focus();
}
