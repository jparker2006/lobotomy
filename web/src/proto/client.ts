// HTTP client for the T2 protocol: /api/meta, /api/calibration, POST /api/generate, long-poll /api/frames.
// Paths are relative to the page, so the page works behind a path prefix (preview proxies).
import { Frame, Meta, parseFrames } from "./frames.js";

export interface Client {
  meta(): Promise<Meta>;
  /** Per-neuron p99 of |activation|, float32, layer-major. */
  calibration(): Promise<Float32Array>;
  /**
   * Asks a question and streams its frames in order, ending after the done frame. Asking again (here or from another tab)
   * replaces the generation: the old stream ends quietly. abort() ends the current stream from the page.
   */
  generate(prompt: string, onFrames: (frames: Frame[], id: string) => void): Promise<{ id: string; ended: "done" | "replaced" }>;
  abort(): void;
}

export function createClient(base = ""): Client {
  let current: AbortController | null = null;
  const url = (p: string) => base + p;

  async function json<T>(r: Response): Promise<T> {
    if (!r.ok) throw new Error(`${r.url}: HTTP ${r.status} ${await r.text().catch(() => "")}`.trim());
    return r.json() as Promise<T>;
  }

  return {
    async meta() { return json<Meta>(await fetch(url("api/meta"))); },
    async calibration() {
      const r = await fetch(url("api/calibration"));
      if (!r.ok) throw new Error(`api/calibration: HTTP ${r.status}`);
      return new Float32Array(await r.arrayBuffer());
    },
    abort() { current?.abort(); current = null; },
    async generate(prompt, onFrames) {
      current?.abort();
      const ctl = (current = new AbortController()), signal = ctl.signal;
      const { id } = await json<{ id: string }>(await fetch(url("api/generate"), {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt }), signal,
      }));
      let next = 0, fails = 0;
      try {
        for (;;) {
          if (signal.aborted) return { id, ended: "replaced" };
          let r: Response;
          try {
            r = await fetch(url(`api/frames?id=${id}&from=${next}`), { signal, cache: "no-store" });
          } catch (e) {
            if (signal.aborted) return { id, ended: "replaced" };
            if (++fails > 5) throw e;
            await new Promise((res) => setTimeout(res, 200 * fails)); // network blip: retry from the same frame
            continue;
          }
          if (r.status === 404) return { id, ended: "replaced" }; // a newer question replaced this one
          if (!r.ok) throw new Error(`api/frames: HTTP ${r.status}`);
          const frames = parseFrames(await r.arrayBuffer());
          fails = 0;
          if (signal.aborted) return { id, ended: "replaced" };
          if (!frames.length) continue;
          next += frames.length;
          onFrames(frames, id);
          if (frames[frames.length - 1].h.kind === "done") return { id, ended: "done" };
        }
      } finally {
        if (current === ctl) current = null;
      }
    },
  };
}
