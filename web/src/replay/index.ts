// Replay (U6): record / replay sessions (app.beginSession + app.ingest, store.replay), deterministic clips via app.renderAt().
// Stub from U2: the ticket that owns this folder fills it in. See web/ARCHITECTURE.md.
import type { App } from "../app/app.js";

export function initReplay(app: App): void {
  void app;
}
