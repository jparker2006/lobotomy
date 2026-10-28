// U5 scripted browser check (#23; sandbox tooling, not a project dependency). Feeds a recorded CPU session through
// app.beginSession / app.ingest; expected values come from expect.py (numpy, f16 activation / p99 from the same file).
// Covers: no chat-template markup anywhere in the UI at any token; `/L8 908` locks while pinned at a prompt token (also with a
// typed slash); Esc order (command line -> lock -> pin); readout values equal the session's; + FIRES MOST ON lists the ocean
// words; the highlighted words carry the session's values; word clicks pin, space resumes; hover picks L8 N908; click locks.
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
const MARKUP = /<\||\|>|<\/?think>/;

const browser = await chromium.launch({ executablePath: "/usr/bin/chromium", headless: false,
  args: ["--use-angle=gl", "--ignore-gpu-blocklist", "--enable-unsafe-webgpu", "--use-webgpu-adapter=swiftshader"] });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: Number(process.env.DPR || 2) });
page.on("console", (m) => { if (m.type() === "error") console.log("console:", m.text()); });
page.on("pageerror", (e) => console.log("pageerror:", e.message));
await page.route("**/rec/ocean.bin", (r) => r.fulfill({ body: readFileSync(`${REC}/ocean.bin`), contentType: "application/octet-stream" }));
await page.goto(PAGE);
await page.waitForFunction(() => window.__app?.store.state.status === "ready", null, { timeout: 120000 });
console.log("deck backend:", await page.evaluate(() => window.__app.deck.backend));

// feed the recording at its measured pace (arrival times from now on)
await page.evaluate(async (prompt) => {
  const app = window.__app, { parseFrames } = await import("/proto/frames.js");
  const frames = parseFrames(await (await fetch("/rec/ocean.bin")).arrayBuffer());
  app.beginSession(prompt, "recorded");
  const t0 = app.clock.now();
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
    const s = await window.__app.deck.snapshot(window.__T);
    let c = document.getElementById("shot");
    if (!c) { c = document.createElement("canvas"); c.id = "shot"; Object.assign(c.style, { position: "fixed", inset: "0", pointerEvents: "none", width: "100vw", height: "100vh" }); document.body.insertBefore(c, document.getElementById("ui")); }
    c.width = s.width; c.height = s.height;
    c.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(s.data.buffer, s.data.byteOffset, s.data.length), s.width, s.height), 0, 0);
    c.style.display = "";
  });
  await page.screenshot({ path });
  await page.evaluate(() => { document.getElementById("shot").style.display = "none"; });
};
const state = () => page.evaluate(() => { const s = window.__app.store.state; return { pinned: s.pinnedIndex, locked: s.locked, cmdOpen: document.querySelector(".insp-cmd").classList.contains("insp-open") }; });
const isLocked = (st) => st.locked?.layer === 8 && st.locked?.neuron === 908;

// 1. no chat-template markup in any text or tooltip, pinned at every token, Cinema and Console
const leaks = await page.evaluate((src) => {
  const re = new RegExp(src), app = window.__app, bad = [];
  for (const consoleOpen of [false, true]) {
    app.store.set({ consoleOpen });
    for (let k = 0; k < app.store.state.session.tokens.length; k++) {
      app.pin(k);
      const text = document.getElementById("ui").innerText, titles = [...document.querySelectorAll("#ui [title]")].map((e) => e.title).join(" ");
      if (re.test(text) || re.test(titles)) bad.push(`${consoleOpen ? "console" : "cinema"} k=${k}`);
    }
  }
  app.store.set({ consoleOpen: false });
  app.unpin();
  return bad;
}, MARKUP.source);
check(leaks.length === 0, `no template markup in the UI at any of ${exp.tokens.length} tokens (Cinema + Console)${leaks.length ? ": " + leaks.slice(0, 5).join(", ") : ""}`);

// 2. pinned at a prompt template token (17, <think>), `/L8 908` + Enter must lock (Jake's bug)
const think = exp.tokens.findIndex((t) => t.text === "<think>");
await page.keyboard.press("Escape"); // the question box has focus at start: Esc hands the keys back to the page
await page.click(`.ui-bar[data-token="${think}"]`);
await settle();
const sub = await page.evaluate(() => document.querySelector(".ui-sub").innerText.replace(/\s+/g, " ").trim());
check(sub === PROMPT, `pinned at <think> (${think}): subtitles "${sub}"`);
check(await page.evaluate(() => document.querySelector(".ui-bar.ui-cur, .ui-bar.ui-tmpl") !== null && document.querySelectorAll(".ui-bar.ui-tmpl").length === 11), "template tape bars kept, dimmed (11 + the current one)");
await shot(`${SHOTS}/prompt-subtitle-no-markup.png`);
await page.keyboard.press("/");
await page.keyboard.type("L8 908");
await page.keyboard.press("Enter");
await settle();
let st = await state();
check(isLocked(st) && st.pinned === think, `pinned at ${think}: /L8 908 + Enter locks ${JSON.stringify(st.locked)}, still pinned ${st.pinned}`);
const ro = () => page.evaluate(() => {
  const q = (s) => document.querySelector(s), now = q(".insp-readout .insp-now");
  const fires = (cls) => [...document.querySelectorAll(`.insp-readout .${cls} .insp-fire`)].map((e) => ({ k: Number(e.dataset.token), r: Number(e.dataset.r), text: e.textContent }));
  return {
    title: q(".insp-readout .insp-title").textContent, now: now.textContent, nowR: Number(now.dataset.r),
    word: now.parentElement.nextElementSibling.querySelector(".insp-v").textContent,
    pos: fires("insp-fires-pos"), neg: fires("insp-fires-neg"), peak: q(".insp-readout .insp-peak")?.textContent ?? null,
    words: [...document.querySelectorAll(".insp-readout .insp-w")].map((e) => ({ k: Number(e.dataset.token), r: Number(e.dataset.r), lit: !!e.style.background, now: e.classList.contains("insp-w-now") })),
    visible: [...document.querySelectorAll(".insp-readout .insp-sub")].map((e) => e.textContent),
  };
});
let r = await ro();
check(near(r.nowR, exp.tokens[think].r) && r.word === "— (CHAT TEMPLATE)", `readout at <think>: NOW ${r.now}, WORD ${r.word} (session ${exp.tokens[think].r.toFixed(4)})`);

// 3. Esc order: command line -> lock -> pin
await page.keyboard.press("/");
st = await state();
check(st.cmdOpen, "/ opens the command line");
await page.keyboard.press("Escape"); st = await state();
check(!st.cmdOpen && isLocked(st) && st.pinned === think, "Esc 1 closes the command line only");
await page.keyboard.press("Escape"); await settle(); st = await state();
check(!st.locked && st.pinned === think, "Esc 2 unlocks, still pinned");
await page.keyboard.press("Escape"); st = await state();
check(!st.locked && st.pinned === null, "Esc 3 unpins");

// 4. pin " sea" (tape), then `/` with a typed slash: the readout equals the session
const sea = exp.tokens.findIndex((t) => t.kind === "gen" && t.text === " sea");
await page.click(`.ui-bar[data-token="${sea}"]`);
await page.keyboard.press("/");
await page.keyboard.type("/L8 908");
await page.keyboard.press("Enter");
await settle();
st = await state();
check(isLocked(st) && st.pinned === sea, `typed "/L8 908" also locks; pinned at " sea" (${sea})`);
r = await ro();
const t = exp.tokens[sea];
check(r.title === "L08 · N0908", `title ${r.title}`);
check(near(r.nowR, t.r) && r.now.startsWith(`${fmt(t.r)}× ITS P99 · TOP `), `NOW "${r.now}" (session ${t.r.toFixed(4)})`);
check(r.word === JSON.stringify(t.text), `WORD ${r.word}`);
const posText = r.pos.map((f) => f.text).join("  ");
check(["\" ocean\"", "\" sea\"", "\" water\""].every((w) => r.pos.some((f) => f.text.startsWith(w))), `+ FIRES MOST ON: ${posText}`);
check(r.pos.every((f, j) => near(f.r, exp.tokens[f.k].r) && f.r > 0 && (j === 0 || f.r <= r.pos[j - 1].r)), "+ list: the session's values, positive, strongest first");
check(r.neg.every((f) => near(f.r, exp.tokens[f.k].r) && f.r < 0), `− FIRES AGAINST: ${r.neg.map((f) => f.text).join("  ") || "(none above the top 3%)"}`);
check(r.peak === null, "no hidden-peak line for L8 N908");
check(r.visible.includes("+ FIRES MOST ON") && r.visible.includes("− FIRES AGAINST"), `sections: ${r.visible.join(" | ")}`);
const gen = exp.tokens.filter((x) => x.kind === "gen");
check(r.words.length >= gen.length - 2 && r.words.every((w) => near(w.r, exp.tokens[w.k].r)), `highlighted words: ${r.words.length} words, values equal the session's`);
const lit = r.words.filter((w) => w.lit).map((w) => JSON.stringify(exp.tokens[w.k].text));
check(["\" ocean\"", "\" sea\"", "\" water\"", "\" waters\""].every((w) => lit.includes(w)) && r.words.find((w) => w.now)?.k === sea, `lit (+): ${lit.join(" ")}; underlined: " sea"`);
await shot(`${SHOTS}/lock-words.png`);

// 5. a word in the readout and a subtitle word pin their token; space returns to live
const water = r.words.find((w) => exp.tokens[w.k].text === " water").k;
await page.click(`.insp-readout .insp-w[data-token="${water}"]`);
check((await state()).pinned === water, `readout word click pins " water" (${water})`);
const subK = await page.evaluate(() => { const e = [...document.querySelectorAll(".ui-sub [data-token]")].find((x) => x.textContent.trim()); return e && Number(e.dataset.token); });
await page.click(`.ui-sub [data-token="${subK}"]`);
check((await state()).pinned === subK, `subtitle word click pins token ${subK} (${JSON.stringify(exp.tokens[subK]?.text)})`);
await page.keyboard.press(" ");
check((await state()).pinned === null, "space returns to live");

// 6. invalid input says so; Esc closes it; Esc unlocks
await page.keyboard.press("/"); await page.keyboard.type("L99 1"); await page.keyboard.press("Enter");
const msg = await page.textContent(".insp-cmd-msg");
check(/NO LAYER 99/.test(msg), `invalid input: ${msg}`);
await page.keyboard.press("Escape");
check(!(await state()).cmdOpen, "Esc closes the command line");
await page.keyboard.press("Escape");
await settle();
check(await page.evaluate(() => window.__app.store.state.locked === null && window.__app.deck.params.visibleLayers === null), "Esc unlocks: every layer drawn again");

// 7. hover (pinned at " sea", overview): the dot of L8 N908 picks L8 N908; a click locks on
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
check(isLocked(await state()), "click locks on the hovered neuron");
await page.keyboard.press("c");
await settle();
check(await page.evaluate(() => window.__app.store.state.consoleOpen && getComputedStyle(document.querySelector(".insp-readout")).display !== "none"), "Console open while locked: readout shown");
await shot(`${SHOTS}/lock-console.png`);

await browser.close();
console.log(fails.length ? `\n${fails.length} FAILED` : "\nALL CHECKS PASSED");
process.exit(fails.length ? 1 : 0);
