// Cinema UI + Console (U3, #21) on the 9 x 16 text grid: title + backend, legend, the big top-1% counter, subtitles on a dark
// scrim, the prompt line, the token tape, layer labels; key `c` toggles the Console (transcript, top 1% by layer, rarest now).
// Every number comes from the frame stream through data/stats.ts. Words and tape bars carry data-token + `interactive`;
// what a click does belongs to inspect/ (U5).
import type { App } from "../app/app.js";
import type { Token } from "../proto/session.js";
import { loadCSS } from "../app/css.js";
import { provenance } from "../app/provenance.js";
import { rarest, tape } from "../data/stats.js";
import { cleanText, displayText, isTemplate } from "../data/display.js";

const LABELLED = [0, 9, 18, 27], CW = 9, SUB_CW = 18, BAR_MAX = 9, TAPE_H = 16, LAYER_DOTS = 22;
const pad = (n: number, w: number) => String(n).padStart(w, "0");
const fmtR = (r: number) => { const a = Math.abs(r); return (r < 0 ? "−" : "+") + (a < 100 ? a.toFixed(1) : a < 1e4 ? a.toFixed(0) : a.toExponential(1)); };

/** Keys belong to whoever owns the focus: never toggle panels while someone types (ARCHITECTURE shared rules). */
export function typing(e: Event): boolean {
  const isField = (t: EventTarget | null) => t instanceof HTMLElement && (t.matches("input, textarea") || t.isContentEditable);
  return isField(e.target) || isField(document.activeElement);
}

/** Greedy word wrap of (char, token) pairs; whitespace runs collapse to one space. Returns lines of chars. */
export function wrapChars(chars: { c: string; k: number }[], width: number): { c: string; k: number }[][] {
  const words: { c: string; k: number }[][] = [];
  let w: { c: string; k: number }[] = [], space = false;
  for (const ch of chars) {
    const ws = /\s/.test(ch.c);
    if (ws) { if (!space && w.length) w.push({ c: " ", k: ch.k }); space = true; continue; }
    if (space && w.length) { words.push(w); w = []; }
    space = false;
    w.push(ch);
  }
  if (w.length) words.push(w);
  const lines: { c: string; k: number }[][] = [];
  let line: { c: string; k: number }[] = [];
  const len = (x: { c: string }[]) => (x.length && x[x.length - 1].c === " " ? x.length - 1 : x.length);
  for (let word of words) {
    while (len(word) > width) { // a word longer than a line: hard break
      if (line.length) { lines.push(line); line = []; }
      lines.push(word.slice(0, width)); word = word.slice(width);
    }
    if (line.length && line.length + len(word) > width) { lines.push(line); line = []; }
    line.push(...word);
  }
  if (line.length) lines.push(line);
  return lines.map((l) => (l.length && l[l.length - 1].c === " " ? l.slice(0, -1) : l));
}

export function initUI(app: App) {
  loadCSS(new URL("./ui.css", import.meta.url));
  const { store, deck } = app;
  const el = <K extends keyof HTMLElementTagNameMap = "div">(cls: string, parent: HTMLElement = app.ui, tag?: K, text = "") => {
    const e = document.createElement(tag ?? "div") as HTMLElementTagNameMap[K];
    e.className = cls;
    if (text) e.textContent = text;
    parent.append(e);
    return e;
  };
  const root = el("ui-root");

  // ---- top: title + provenance, legend, status; counter + stats ----
  const head = el("ui-head", root);
  el("ui-title", head, "b", "LOBOTOMY");
  const metaLine = el("ui-meta", head, "span");
  const legend = el("ui-legend", root);
  el("ui-legend-dots", legend, "span", "· TOP 10%  ∙ 3%  • 1%  ● 0.3%  ◉ 0.1%  FOR THAT NEURON");
  el("ui-legend-sign", legend, "span", " · FILLED = POSITIVE, HOLLOW = NEGATIVE");
  const status = el("ui-status", root);
  const counter = el("ui-counter", root), stats = el("ui-stats", root);

  // ---- bottom: scrim + subtitles, prompt line, tape; console stats line ----
  el("ui-scrim", root);
  const sub = el("ui-sub", root);
  const form = el("ui-prompt", root, "form");
  el("ui-prompt-mark", form, "span", ">");
  const input = el("ui-input interactive", form, "input");
  Object.assign(input, { autocomplete: "off", spellcheck: false, placeholder: "ask Qwen something, then Enter", value: "Describe the ocean at night in three sentences." });
  const tapeRow = el("ui-tape", root);
  el("ui-tape-label", tapeRow, "span", "TAPE");
  const bars = el("ui-bars", tapeRow);
  const cstats = el("ui-cstats", root);

  const labels = LABELLED.map((l) => ({ l, e: el("ui-label", root, "div", `L${pad(l, 2)}`) }));

  // ---- console panels ----
  const left = el("ui-panel ui-left", root), right = el("ui-panel ui-right", root);
  const transcript = el("ui-transcript", left);
  el("ui-h", right, "div", "TOP 1% BY LAYER");
  const layerRows = Array.from({ length: 28 }, (_, l) => {
    const row = el("ui-lrow", right);
    el("ui-lname", row, "span", `L${pad(l, 2)}`);
    return { dots: el("ui-ldots", row, "span"), n: el("ui-ln", row, "span") };
  });
  el("ui-h ui-h2", right, "div", "RAREST RIGHT NOW");
  const rareRows = Array.from({ length: 6 }, () => el("ui-rrow", right));

  // ---- input ----
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (!q) return;
    input.blur(); // hand the keys back to the page (`c`, `~`, `/` …)
    app.ask(q);
  });
  input.addEventListener("keydown", (e) => { if (e.key === "Escape") input.blur(); });
  addEventListener("keydown", (e) => {
    if (e.key !== "c" || e.ctrlKey || e.metaKey || e.altKey || e.repeat || typing(e)) return;
    store.set({ consoleOpen: !store.state.consoleOpen });
  });

  // ---- what is "now": the pinned word, else the newest word on the deck ----
  const current = (): { k: number; tok: Token | null } => {
    const s = store.state, k = s.pinnedIndex ?? s.liveIndex;
    return { k, tok: s.session && k >= 0 ? s.session.tokens[k] ?? null : null };
  };
  const ended = (k: number) => { const s = store.state, sess = s.session!; return !!sess.done && s.pinnedIndex === null && k >= sess.tokens.length - 1; };

  function renderHead() {
    const s = store.state, m = s.meta;
    metaLine.textContent = `${m ? m.model.split("/").pop()!.toUpperCase() + " · " : ""}BACKEND: ${provenance(s)} · ${m ? m.layers : 28} × ${(m ? m.neurons : 3072).toLocaleString("en-US")} NEURONS`;
  }

  function renderNumbers() {
    const s = store.state, { tok } = current(), max = s.meta?.max_tokens ?? 256;
    const n = tok ? tok.top1.toLocaleString("en-US") : "—";
    // elapsed at the current word: when its frame arrived, from the question (a pinned word keeps its own time)
    const sec = tok && s.session ? tok.t - s.session.startedAt : 0;
    const tokTxt = `TOKEN ${pad(tok ? tok.i + 1 : 0, 3)}/${max}`, el_ = `${sec.toFixed(1)} S`;
    const rate = `${s.tokS ? s.tokS.toFixed(1) : "—"} TOK/S`;
    counter.textContent = n;
    stats.textContent = `NEURONS IN THEIR TOP 1% · ${tokTxt} · ${el_} · ${rate}`;
    cstats.textContent = `${tokTxt} · ${el_} · ${rate} · ${n} IN TOP 1%`;
  }

  // words: spans with data-token, the current one hot, then a block cursor while the answer is still coming
  function wordSpan(parent: HTMLElement, text: string, k: number, hot: boolean) {
    const e = el(hot ? "interactive ui-hot" : "interactive", parent, "span", text);
    e.dataset.token = String(k);
  }
  function cursor(parent: HTMLElement) { el("ui-cursor", parent, "span"); }

  let subWidth = 74;
  function renderSub() {
    sub.textContent = "";
    const s = store.state, sess = s.session, { k, tok } = current();
    if (!sess || !tok) return;
    const kind = tok.kind, chars: { c: string; k: number }[] = [];
    // the chat template is never printed (data/display.ts): only the question and the answer
    for (let j = 0; j <= k; j++) if (sess.tokens[j].kind === kind) for (const c of displayText(sess, j)) chars.push({ c, k: j });
    if (kind === "gen" && ended(k)) for (const c of cleanText(sess.done!.text)) chars.push({ c, k });
    const lines = wrapChars(chars, subWidth).slice(-2);
    lines.forEach((line, li) => {
      const row = el(li === lines.length - 1 ? "ui-subline ui-last" : "ui-subline", sub);
      for (let a = 0; a < line.length;) {
        let b = a;
        while (b < line.length && line[b].k === line[a].k) b++;
        wordSpan(row, line.slice(a, b).map((x) => x.c).join(""), line[a].k, line[a].k === k);
        a = b;
      }
      if (li === lines.length - 1 && !ended(k)) cursor(row);
    });
    if (!lines.length && !ended(k)) cursor(el("ui-subline ui-last", sub));
  }

  function renderTranscript() {
    transcript.textContent = "";
    const s = store.state, sess = s.session;
    if (!sess) return;
    el("ui-tprompt", transcript, "div", `> ${sess.prompt}`);
    const ans = el("ui-tanswer", transcript), { k } = current();
    for (let j = 0; j <= k; j++) {
      const t = sess.tokens[j];
      const text = t.kind === "gen" ? displayText(sess, j) + (j === k && ended(k) ? cleanText(sess.done!.text) : "") : "";
      if (text) wordSpan(ans, text, j, j === k);
    }
    if (!ended(k)) cursor(ans);
    transcript.scrollTop = transcript.scrollHeight;
  }

  function renderPanels() {
    const { tok } = current();
    const by = tok ? tok.top1ByLayer : new Uint16Array(28);
    let mx = 0, top = -1;
    by.forEach((n, l) => { if (n > mx) { mx = n; top = l; } });
    layerRows.forEach((row, l) => {
      row.dots.textContent = "•".repeat(mx ? Math.round((LAYER_DOTS * by[l]) / mx) : 0);
      row.dots.classList.toggle("ui-hot", l === top);
      row.n.textContent = tok ? String(by[l]) : "";
    });
    const rare = tok ? rarest(tok.r, rareRows.length, store.state.meta?.neurons ?? 3072) : [];
    rareRows.forEach((row, j) => {
      const x = rare[j];
      row.textContent = x ? `L${pad(x.layer, 2)} N${pad(x.neuron, 4)}  ${fmtR(x.r)}× p99` : "";
    });
  }

  // tape: one bar per session token (bars are reused), height = its top-1% count
  const barEls: HTMLElement[] = [];
  function renderTape() {
    const s = store.state, sess = s.session, toks = sess ? sess.tokens : [];
    while (barEls.length < toks.length) { const b = el("ui-bar interactive", bars, "i"); b.dataset.token = String(barEls.length); barEls.push(b); }
    while (barEls.length > toks.length) barEls.pop()!.remove();
    const { heights, scale } = tape(toks), { k } = current(), live = s.liveIndex;
    const w = Math.max(1, Math.min(BAR_MAX, Math.floor(bars.clientWidth / Math.max(1, toks.length))));
    bars.style.setProperty("--w", `${w}px`);
    bars.style.setProperty("--gap", w >= 4 ? "1px" : "0px");
    toks.forEach((t, j) => {
      const b = barEls[j], h = Math.max(1, Math.round(TAPE_H * Math.min(1, heights[j] / scale)));
      b.style.setProperty("--h", `${h}px`);
      // template tokens keep their bar (real data), dimmed, with no text
      b.className = `ui-bar interactive${j === k ? " ui-cur" : j > live ? " ui-future" : isTemplate(sess!, j) ? " ui-tmpl" : t.kind === "prompt" ? " ui-p" : ""}`;
    });
  }

  function renderStatus() {
    const s = store.state;
    status.textContent = s.status === "error" ? `ERROR: ${s.error}` : s.status === "connecting" ? "CONNECTING…" : s.status === "asking" ? "THINKING…"
      : s.pinnedIndex !== null && s.session ? `PINNED AT TOKEN ${pad(s.session.tokens[s.pinnedIndex].i + 1, 3)}` : "";
  }

  // The Console frames the deck between its panels: the camera rig keeps the deck in the free band, so overview(),
  // faceLayer() and double-click stay framed while the Console is open (camera.setFrame; undone on close).
  function frameDeck(open: boolean) {
    const W = Math.max(1, innerWidth);
    void app.camera.setFrame(open ? { l: left.offsetWidth / W, r: right.offsetWidth / W } : { l: 0, r: 0 }, 500, "console");
  }
  function renderConsole() {
    root.classList.toggle("ui-console", store.state.consoleOpen);
    renderWords();
  }
  function renderWords() {
    if (store.state.consoleOpen) { renderTranscript(); renderPanels(); } else renderSub();
  }
  function resize() {
    subWidth = Math.max(16, Math.floor((innerWidth - 12 * CW) / SUB_CW)); // 74 double cells at 1440 px, as in q5-cinema
    renderWords(); renderTape();
  }

  store.on("change", (keys) => {
    const has = (...f: string[]) => keys.some((k) => f.includes(k));
    if (has("meta", "backend", "replay")) renderHead();
    if (has("liveIndex", "pinnedIndex", "tokS", "elapsedSec", "session")) renderNumbers();
    if (has("liveIndex", "pinnedIndex", "session")) { renderWords(); renderTape(); }
    if (has("session") && store.state.session && document.activeElement !== input) input.value = store.state.session.prompt;
    if (has("status", "error", "pinnedIndex")) renderStatus();
    if (has("consoleOpen")) { renderConsole(); frameDeck(store.state.consoleOpen); }
  });
  store.on("arrive", () => renderTape()); // arrived, not yet on the deck: a faint bar
  store.on("done", () => { renderWords(); renderNumbers(); });
  store.on("frame", () => {
    const vis = deck.params.visibleLayers;
    for (const { l, e } of labels) {
      const p = deck.layerLabel(l);
      e.style.display = p && (!vis || vis[l]) ? "" : "none";
      if (p) e.style.transform = `translate(${p.x}px, ${p.y}px) translate(-50%, -100%)`;
    }
  });
  addEventListener("resize", resize);
  renderHead(); renderNumbers(); renderStatus(); renderConsole(); resize();
  input.focus();
}
