# U3 sandbox renders (#21)

Cloud sandbox (no GPU): Chromium 156, SwiftShader WebGPU, 1440 × 900 CSS px at DPR 2 (the mockups' size), CPU stand-in
(Qwen3-0.6B). The deck is composited from `deck.snapshot()` (this sandbox never composites WebGPU canvases); the UI is the
real DOM. Timings and tok/s from here mean nothing for the Mac.

- `cpu-ocean-cinema-token053-*`, `cpu-ocean-console-token053-*`: "Describe the ocean at night in three sentences.", word 53
  (" glow", the q5 mockups' word) shown with word replay (`app.pin`), Cinema and Console (`c`).
- `cpu-sky-cinema-live-*`: "Why is the sky blue? Answer in two sentences." mid-stream, live (tape past the current word faint).
- `cpu-sky-cinema-live-1100px-*`: the same at 1100 × 760: the legend drops its sign half.
