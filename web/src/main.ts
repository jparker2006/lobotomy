// Bootstrap only: build the app context, then hand it to each module. Modules never edit this file (ARCHITECTURE.md).
import { createApp } from "./app/app.js";
import { initUI } from "./ui/index.js";
import { initMotion } from "./motion/index.js";
import { initInspect } from "./inspect/index.js";
import { initReplay } from "./replay/index.js";

async function main() {
  const canvas = document.getElementById("deck") as HTMLCanvasElement, ui = document.getElementById("ui")!;
  const forceWebGL = new URLSearchParams(location.search).get("backend") === "webgl2";
  const app = await createApp(canvas, ui, { forceWebGL });
  (window as any).__app = app; // for the console and headless checks
  initUI(app);
  initMotion(app);
  initInspect(app);
  initReplay(app);
  app.start();
}

main().catch((e) => {
  const d = document.createElement("div");
  d.id = "fatal";
  d.textContent = `lobotomy failed to start: ${e?.stack || e}`;
  document.body.append(d);
  console.error(e);
});
