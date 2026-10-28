// What the page may say about where the data came from (design.md Principle 5; honesty rules).
// The backend name always comes from /api/meta or the replay file. Never print "NPU" or a power figure unless the backend
// really is the board and a measured power record is attached (state.replay.power).
import type { AppState } from "./state.js";

/** Display label for a backend name. Unknown names are shown as they are, upper-cased, with no claims added. */
export function backendLabel(backend: string): string {
  if (backend === "cpu") return "CPU STAND-IN";
  if (backend === "fake") return "FAKE · SYNTHETIC DATA";
  return backend ? backend.toUpperCase() : "…";
}

/** The HUD's provenance line: live backend, or REPLAY · <backend> · <date>. */
export function provenance(s: Pick<AppState, "backend" | "replay">): string {
  if (s.replay) return `REPLAY · ${backendLabel(s.replay.backend)} · ${s.replay.recordedAt}`;
  return backendLabel(s.backend);
}
