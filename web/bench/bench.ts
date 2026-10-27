// Render benchmark (U1, issue #19): the real deck pipeline fed SYNTHETIC activations at the real frame size.
// Runs a fixed plan, measures rAF frame intervals, shows the result and POSTs it to bench/serve.py.
// URL: ?auto=0 (wait for a click), ?backend=webgl2 (force the fallback), ?scale=0.1 (shorter runs, for smoke tests).
import { createDeck, makeSynth, recordedRarityTable, NEURONS, Deck } from "lobotomy/render";

const q = new URLSearchParams(location.search);
const forceWebGL = q.get("backend") === "webgl2";
const auto = q.get("auto") !== "0";
const scale = Math.max(0.01, +(q.get("scale") || 1));
const native = window.devicePixelRatio || 1;
const WARMUP = 2; // seconds dropped at the start of each run
const BUDGET_P95 = 1000 / 60, JITTER = 0.3; // vsync timestamps jitter a little around 16.67 ms; the allowance is reported

interface Run { name: string; secs: number; dpr: number; bloomScale: number; tokS: number }
const RUNS: Run[] = [
  { name: "main", secs: 60, dpr: native, bloomScale: 1, tokS: 24 },
  { name: "bloom 0.5x", secs: 15, dpr: native, bloomScale: 0.5, tokS: 24 },
  { name: "dpr 1", secs: 15, dpr: 1, bloomScale: 1, tokS: 24 },
  { name: "6 tok/s", secs: 15, dpr: native, bloomScale: 1, tokS: 6 },
].map((r) => ({ ...r, secs: r.secs * scale }));

const $ = (id: string) => document.getElementById(id)!;
const canvas = $("deck") as HTMLCanvasElement;

const pct = (a: number[], p: number) => { if (!a.length) return NaN; const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / Math.max(1, a.length);
const r2 = (x: number) => Math.round(x * 100) / 100;

function browserName() {
  const ua = navigator.userAgent;
  if (/Edg\//.test(ua)) return "edge";
  if (/Firefox\//.test(ua)) return "firefox";
  if (/Chrome\//.test(ua)) return "chrome";
  if (/Version\/[\d.]+.*Safari/.test(ua)) return "safari";
  return "other";
}

async function adapterInfo(): Promise<Record<string, unknown>> {
  const gpu = (navigator as any).gpu;
  if (gpu && !forceWebGL) {
    try {
      const a = await gpu.requestAdapter();
      const i = a?.info ?? {};
      return { vendor: i.vendor, architecture: i.architecture, device: i.device, description: i.description, timestampQuery: !!a?.features?.has("timestamp-query") };
    } catch (e) { return { error: String(e) }; }
  }
  const gl = document.createElement("canvas").getContext("webgl2");
  const ext = gl?.getExtension("WEBGL_debug_renderer_info");
  return { webgl2: !!gl, vendor: ext ? gl!.getParameter(ext.UNMASKED_VENDOR_WEBGL) : null, renderer: ext ? gl!.getParameter(ext.UNMASKED_RENDERER_WEBGL) : null };
}

// SYNTHETIC subtitle words (a stand-in for U3's answer line, not model output)
const WORDS = "synthetic tokens at the real frame size light every neuron of the deck while the bench measures each frame".split(" ");

async function main() {
  const deck: Deck = await createDeck(canvas, { tokPerSec: 24 }, { forceWebGL, gpuTiming: true });
  deck.setRarityTable(recordedRarityTable());
  const synth = makeSynth(19);
  const buf = new Float32Array(NEURONS);
  const info = await adapterInfo();
  (window as any).__deck = deck;
  // Headless checks: draw one frame offscreen and show it in a 2D canvas over the deck (some sandboxes never composite WebGPU canvases).
  (window as any).__showSnapshot = async () => {
    const s = await deck.snapshot(performance.now() / 1000);
    let c = document.getElementById("snap") as HTMLCanvasElement | null;
    if (!c) { c = document.createElement("canvas"); c.id = "snap"; c.style.cssText = "position:fixed;inset:0;width:100vw;height:100vh;pointer-events:none"; canvas.after(c); }
    c.width = s.width; c.height = s.height;
    c.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(s.data.buffer as ArrayBuffer), s.width, s.height), 0, 0);
    return [s.width, s.height];
  };

  const labels = [0, 9, 18, 27].map((l) => { const e = document.createElement("div"); e.className = "lab"; e.textContent = `L${String(l).padStart(2, "0")}`; document.body.append(e); return { l, e }; });
  const hud = $("hud"), status = $("status"), sub = $("sub"), panel = $("panel");
  const hudText = () => `<b>LOBOTOMY</b>  RENDER BENCH · ${deck.backend.toUpperCase()}${forceWebGL ? " (forced)" : ""} · ${info.description || info.renderer || info.vendor || ""}\n` +
    `${deck.size.pxW} × ${deck.size.pxH} px · DPR ${deck.size.dpr} · ${deck.size.cols} × ${deck.size.rows} cells · 28 × 3,072 neurons`;

  let words: string[] = [], tok = 0;
  function subtitle() {
    words.push(WORDS[tok++ % WORDS.length]);
    if (words.length > 9) words = words.slice(-9);
    sub.innerHTML = words.slice(0, -1).join(" ") + " <span>" + words[words.length - 1] + "</span>";
  }

  // run state
  let runIdx = -1, runStart = 0, lastTs = 0, nextTok = 0, hidden = false;
  let intervals: number[] = [], cpuMs: number[] = [], synthMs: number[] = [], gpu: { compute: number; render: number; frames: number }[] = [];
  const results: any[] = [];
  let lastGpu = 0;
  document.addEventListener("visibilitychange", () => { if (document.hidden) hidden = true; });

  function apply(r: Run) {
    deck.params.dpr = r.dpr; deck.params.bloomScale = r.bloomScale; deck.params.tokPerSec = r.tokS;
    intervals = []; cpuMs = []; synthMs = []; gpu = []; hidden = false;
    deck.gpuTimes(); // drop timings from the previous run
  }
  function finishRun(r: Run) {
    const iv = intervals;
    const res = {
      name: r.name, secs: r.secs, dpr: deck.size.dpr, bloomScale: r.bloomScale, tokPerSec: r.tokS,
      canvasPx: [deck.size.pxW, deck.size.pxH], cells: deck.size.cols * deck.size.rows,
      frames: iv.length, p50: r2(pct(iv, 0.5)), p95: r2(pct(iv, 0.95)), p99: r2(pct(iv, 0.99)), max: r2(Math.max(...iv)),
      meanMs: r2(mean(iv)), over33Pct: r2((100 * iv.filter((x) => x > 33.3).length) / Math.max(1, iv.length)),
      over25Pct: r2((100 * iv.filter((x) => x > 25).length) / Math.max(1, iv.length)),
      cpuFrameMs: { p50: r2(pct(cpuMs, 0.5)), p95: r2(pct(cpuMs, 0.95)) }, synthTokenMs: r2(mean(synthMs)),
      gpuMs: gpu.length ? { compute: r2(mean(gpu.map((g) => g.compute))), render: r2(mean(gpu.map((g) => g.render))) } : null,
      tabHiddenDuringRun: hidden,
    };
    results.push(res);
  }

  function start() {
    panel.style.display = "none";
    results.length = 0; runIdx = 0; apply(RUNS[0]); runStart = performance.now();
  }

  async function report() {
    const main = results[0];
    const pass = main.p95 <= BUDGET_P95 + JITTER && main.over33Pct <= 1 && !main.tabHiddenDuringRun;
    const payload = {
      kind: "lobotomy-render-bench", version: 1, synthetic: true, utc: new Date().toISOString(), browser: browserName(), userAgent: navigator.userAgent,
      backend: deck.backend, forcedWebGL: forceWebGL, adapter: info, nativeDpr: native, screen: [screen.width, screen.height],
      window: [innerWidth, innerHeight], scale, budget: { p95Ms: r2(BUDGET_P95), jitterAllowanceMs: JITTER, over33MaxPct: 1 },
      verdict: pass ? "PASS" : "FAIL", runs: results,
    };
    (window as any).__bench = { done: true, payload, saved: null };
    const rows = results.map((x) => `${x.name.padEnd(11)} ${String(x.p50).padStart(6)} ${String(x.p95).padStart(6)} ${String(x.p99).padStart(6)} ${String(x.over33Pct).padStart(6)}% ${String(x.frames).padStart(6)}  ${x.canvasPx.join("×").padEnd(10)} cpu ${x.cpuFrameMs.p50}/${x.cpuFrameMs.p95}` + (x.gpuMs ? `  gpu c ${x.gpuMs.compute} r ${x.gpuMs.render}` : "") + (x.tabHiddenDuringRun ? "  TAB HIDDEN" : ""));
    let saved = "saving…";
    const draw = () => {
      panel.innerHTML = `${pass ? "PASS" : "FAIL"}  main run: p95 ${main.p95} ms (budget ${r2(BUDGET_P95)} + ${JITTER} jitter), ${main.over33Pct}% of frames over 33 ms (max 1%)\n\n` +
        `run           p50    p95    p99   >33ms frames  canvas     cpu frame() ms p50/p95\n${rows.join("\n")}\n\n${payload.browser} · ${deck.backend} · ${saved}\n<button id="again">Run again</button>`;
      $("again").onclick = start;
    };
    panel.style.display = "block"; draw();
    try {
      const r = await fetch("/api/results", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(payload) });
      const j = await r.json();
      saved = `saved ${j.saved}`; (window as any).__bench.saved = j.saved;
    } catch (e) { saved = `not saved (${e})`; }
    draw();
  }

  canvas.addEventListener("click", async (e) => {
    const p = await deck.pick(e.clientX, e.clientY);
    $("pick").textContent = p ? `L${String(p.layer).padStart(2, "0")} · N${String(p.index).padStart(4, "0")} · level ${p.level} · ${p.sign > 0 ? "+" : "−"}` : "";
  });

  let lastHud = 0, lastStatus = 0;
  function tick(ts: number) {
    requestAnimationFrame(tick);
    const t = ts / 1000;
    if (nextTok === 0 || t - nextTok > 1) nextTok = t; // first frame, or back from a hidden tab
    while (t >= nextTok) {
      const s0 = performance.now();
      synth.next(buf);
      deck.pushToken(buf, nextTok);
      synthMs.push(performance.now() - s0);
      subtitle();
      nextTok += 1 / deck.params.tokPerSec;
    }
    const c0 = performance.now();
    deck.frame(t);
    const cpu = performance.now() - c0;
    for (const { l, e } of labels) { const p = deck.layerLabel(l); if (p) e.style.transform = `translate(${p.x}px, ${p.y}px) translate(-50%, -100%)`; }
    e: if (runIdx >= 0 && runIdx < RUNS.length) {
      const r = RUNS[runIdx], el = (performance.now() - runStart) / 1000;
      if (el >= WARMUP) {
        intervals.push(ts - lastTs); cpuMs.push(cpu);
        if (ts - lastGpu > 1000) { lastGpu = ts; deck.gpuTimes().then((g) => { if (g) gpu.push(g); }); }
      }
      if (el >= WARMUP + r.secs) {
        finishRun(r);
        runIdx++;
        if (runIdx < RUNS.length) { apply(RUNS[runIdx]); runStart = performance.now(); } else { runIdx = RUNS.length; report(); }
        break e;
      }
      if (ts - lastStatus > 500) { lastStatus = ts; status.textContent = `run ${runIdx + 1}/${RUNS.length}: ${r.name} · ${Math.max(0, Math.ceil(WARMUP + r.secs - el))} s left\nkeep this tab in front, don't touch`; }
    }
    if (ts - lastHud > 1000 || lastHud === 0) { lastHud = ts; hud.innerHTML = hudText(); if (runIdx < 0 || runIdx >= RUNS.length) status.textContent = runIdx < 0 ? "click Start" : "done"; }
    lastTs = ts;
  }
  requestAnimationFrame(tick);
  hud.innerHTML = hudText();
  (window as any).__bench = { done: false };
  if (auto) setTimeout(start, 1500);
  else { panel.style.display = "block"; panel.innerHTML = `Render bench · SYNTHETIC data · about ${Math.round(RUNS.reduce((a, r) => a + r.secs + WARMUP, 0))} s\n<button id="go">Start</button>`; $("go").onclick = start; }
}

main().catch((e) => { document.getElementById("panel")!.style.display = "block"; document.getElementById("panel")!.textContent = "bench failed: " + (e?.stack || e); (window as any).__bench = { done: true, error: String(e) }; });
