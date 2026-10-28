// U5 scripted browser check (sandbox tooling, not a project dependency): feed a recorded CPU session through
// app.beginSession/app.ingest, then `/L8 908` must lock onto L8 N908 with readout values equal to the file's
// (expected values computed independently by expect.py with numpy), and clicking a word must pin that token.
import { chromium } from "playwright-core";
import { readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";

const REC = process.env.REC || "/tmp/rec", PAGE = process.env.PAGE || "http://127.0.0.1:8000/", SHOTS = process.env.SHOTS || "/tmp/rec";
const PROMPT = "Describe the ocean at night in three sentences.";
const exp = JSON.parse(execFileSync(process.env.PY || "python3", [new URL("./expect.py", import.meta.url).pathname, REC, "8", "908"]).toString());
const fails = [];
const check = (ok, msg) => { console.log(`${ok ? "ok  " : "FAIL"} ${msg}`); if (!ok) fails.push(msg); };
const fmt = (r) => (r < 0 ? "−" : "+") + Math.abs(r).toFixed(1);
const near = (a, b) => Math.abs(a - b) <= 1e-4 * Math.max(1, Math.abs(b));

const browser = await chromium.launch({ executablePath: "/usr/bin/chromium", headless: false,
  args: ["--use-angle=gl", "--ignore-gpu-blocklist", "--enable-unsafe-webgpu", "--use-webgpu-adapter=swiftshader"] });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: Number(process.env.DPR || 2) });
page.on("console", (m) => { if (m.type() === "error") console.log("console:", m.text()); });
page.on("pageerror", (e) => console.log("pageerror:", e.message));
await page.route("**/rec/ocean.bin", (r) => r.fulfill({ body: readFileSync(`${REC}/ocean.bin`), contentType: "application/octet-stream" }));
await page.goto(PAGE);
await page.waitForFunction(() => window.__app?.store.state.status === "ready", null, { timeout: 120000 });
console.log("deck backend:", await page.evaluate(() => window.__app.deck.backend));

// feed the recording: arrival times spaced at its measured pace, ending now
await page.evaluate(async (prompt) => {
  const app = window.__app, { parseFrames } = await import("/proto/frames.js");
  const frames = parseFrames(await (await fetch("/rec/ocean.bin")).arrayBuffer());
  app.beginSession(prompt, "recorded");
  const t0 = app.clock.now(), n = frames.length;
  frames.forEach((f, j) => app.ingest([f], t0 + j / 30));
}, PROMPT);
const last = exp.tokens.length - 1;
await page.waitForFunction((last) => { const s = window.__app.store.state; return s.status === "done" && s.liveIndex === last; }, last, { timeout: 120000 });
check(true, `session ingested: ${exp.tokens.length} tokens`);
// SwiftShader cannot keep up with the rAF loop, which starves GPU readbacks (pick, snapshot): freeze the clock and step frames
await page.waitForTimeout(1500);
await page.evaluate(() => { window.__T = window.__app.clock.now() + 2; window.__app.renderAt(window.__T); });
const adv = (dt = 1.2) => page.evaluate((dt) => { window.__T += dt; window.__app.renderAt(window.__T); }, dt);
const settle = async (ms = 400) => { await page.waitForTimeout(ms); await adv(); await page.waitForTimeout(200); };
// the deck canvas is never composited here (WebGPU): draw it from deck.snapshot() under the real DOM, then screenshot
const shot = async (path) => {
  await page.evaluate(async () => {
    const app = window.__app, s = await app.deck.snapshot(window.__T);
    let c = document.getElementById("shot");
    if (!c) { c = document.createElement("canvas"); c.id = "shot"; Object.assign(c.style, { position: "fixed", inset: "0", pointerEvents: "none", width: "100vw", height: "100vh" }); document.body.insertBefore(c, document.getElementById("ui")); }
    c.width = s.width; c.height = s.height;
    c.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(s.data.buffer, s.data.byteOffset, s.data.length), s.width, s.height), 0, 0);

    c.style.display = "";
  });
  await page.screenshot({ path });
  await page.evaluate(() => { document.getElementById("shot").style.display = "none" });
};

// `/` command line: the prompt input has focus at start; Esc hands the keys back to the page
await page.keyboard.press("Escape");
await page.keyboard.press("/");
await page.keyboard.type("L8 908");
await page.keyboard.press("Enter");
await settle();
const readout = () => page.evaluate(() => {
  const q = (s) => document.querySelector(s);
  return {
    locked: window.__app.store.state.locked, visible: window.__app.deck.params.visibleLayers?.map((v, l) => v ? l : -1).filter((l) => l >= 0),
    pinned: window.__app.store.state.pinnedIndex, title: q(".insp-readout .insp-title").textContent,
    now: q(".insp-readout .insp-now").textContent, nowR: Number(q(".insp-readout .insp-now").dataset.r),
    word: q(".insp-readout .insp-now").parentElement.nextElementSibling.querySelector(".insp-v").textContent,
    fires: [...document.querySelectorAll(".insp-readout .insp-fire")].map((e) => ({ k: Number(e.dataset.token), r: Number(e.dataset.r), text: e.textContent })),
    slots: [...document.querySelectorAll(".insp-readout .insp-slots .insp-row")].map((e) => e.textContent),
  };
});
function verify(ro, k, label) {
  const t = exp.tokens[k];
  check(ro.title === "L08 · N0908", `${label}: title ${ro.title}`);
  check(near(ro.nowR, t.r) && ro.now.startsWith(`${fmt(t.r)}× ITS P99 · TOP `), `${label}: NOW "${ro.now}" (session ${t.r.toFixed(4)})`);
  check(ro.word === JSON.stringify(t.text), `${label}: WORD ${ro.word} (session ${JSON.stringify(t.text)})`);
  const ok = ro.fires.length === 5 && exp.best.every((b, j) => ro.fires[j].k === b.k && near(ro.fires[j].r, b.r) && ro.fires[j].text === `${JSON.stringify(b.text)}${fmt(b.r)}×`);
  check(ok, `${label}: FIRES MOST ON ${ro.fires.map((f) => f.text).join("  ")}`);
}
let ro = await readout();
check(ro.locked?.layer === 8 && ro.locked?.neuron === 908, `/L8 908 locks: ${JSON.stringify(ro.locked)}`);
check(JSON.stringify(ro.visible) === "[8]", `only layer 8 drawn: ${JSON.stringify(ro.visible)}`);
verify(ro, last, "live end");
check(ro.slots.join("|").includes("COMING v0.0.3") && ro.slots.length === 4, `slots: ${ro.slots.join(" | ")}`);

// click the tape bar of " sea": pins that token, the readout follows
const sea = exp.tokens.findIndex((t) => t.kind === "gen" && t.text === " sea");
await page.click(`.ui-bar[data-token="${sea}"]`);
await settle();
ro = await readout();
check(ro.pinned === sea, `tape click pins token ${sea} (" sea"): pinnedIndex ${ro.pinned}`);
verify(ro, sea, "pinned \" sea\"");
await shot(`${SHOTS}/lock-slash-cmd.png`);

// a subtitle word pins its token; space returns to live
const sub = await page.evaluate(() => { const e = [...document.querySelectorAll(".ui-sub [data-token]")].find((x) => x.textContent.trim()); return e && Number(e.dataset.token); });
await page.click(`.ui-sub [data-token="${sub}"]`);
check(await page.evaluate(() => window.__app.store.state.pinnedIndex) === sub, `subtitle word click pins token ${sub} (${JSON.stringify(exp.tokens[sub]?.text)})`);
await page.keyboard.press(" ");
check(await page.evaluate(() => window.__app.store.state.pinnedIndex) === null, "space returns to live");

// invalid command says so; Esc cancels it, Esc again unlocks
await page.keyboard.press("/"); await page.keyboard.type("L99 1"); await page.keyboard.press("Enter");
const msg = await page.textContent(".insp-cmd-msg");
check(/NO LAYER 99/.test(msg), `invalid input: ${msg}`);
await page.keyboard.press("Escape");
check(await page.evaluate(() => !document.querySelector(".insp-cmd").classList.contains("insp-open")) , "Esc closes the command line");
await page.keyboard.press("Escape");
await settle();
check(await page.evaluate(() => window.__app.store.state.locked === null && window.__app.deck.params.visibleLayers === null), "Esc unlocks: every layer drawn again");

// hover: pinned at " sea", overview; the dot of L8 N908 picks L8 N908, then a click locks on
await page.click(`.ui-bar[data-token="${sea}"]`);
await settle();
const p = await page.evaluate(async () => { const { neuronPos } = await import("/render/geometry.js"); return window.__app.deck.project(neuronPos(8 * 3072 + 908)); });
await page.mouse.move(p.x - 30, p.y - 30);
await page.mouse.move(p.x, p.y, { steps: 3 });
// picks are GPU readbacks: seconds each on SwiftShader, so wait for the result instead of a fixed time
await page.waitForFunction(() => window.__app.store.state.hover, null, { timeout: 60000 }).catch(() => {});
await adv(0.02);
const hov = await page.evaluate(() => ({ hover: window.__app.store.state.hover, card: document.querySelector(".insp-card .insp-title").textContent, now: document.querySelector(".insp-card .insp-now").textContent }));
check(hov.hover?.layer === 8 && hov.hover?.neuron === 908, `hover picks ${JSON.stringify(hov.hover)}: card ${hov.card} · ${hov.now}`);
await shot(`${SHOTS}/hover-overview.png`);
await page.mouse.click(p.x, p.y);
await page.waitForFunction(() => window.__app.store.state.locked, null, { timeout: 60000 }).catch(() => {});
await settle();
await shot(`${SHOTS}/lock-click.png`);
check(JSON.stringify(await page.evaluate(() => window.__app.store.state.locked)) === '{"layer":8,"neuron":908}', "click locks on the hovered neuron");
// the Console (c) while locked: the readout takes the right panel's place
await page.keyboard.press("c");
await settle();
check(await page.evaluate(() => window.__app.store.state.consoleOpen && getComputedStyle(document.querySelector(".insp-readout")).display !== "none"), "Console open while locked: readout shown");
await shot(`${SHOTS}/lock-console.png`);

await browser.close();
console.log(fails.length ? `\n${fails.length} FAILED` : "\nALL CHECKS PASSED");
process.exit(fails.length ? 1 : 0);
