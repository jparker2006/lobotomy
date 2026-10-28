# U4 sandbox renders (#22)

Taken in the cloud sandbox (no GPU): Chromium with SwiftShader WebGPU, deck drawn via `deck.snapshot()` (this sandbox never
composites WebGPU canvases). Data: the CPU stand-in (Qwen3-0.6B), "Describe the ocean at night in three sentences." The fps
numbers in these shots are software rendering and mean nothing for the Mac.

- `tuning-console-*`: the `~` panel at DPR 2, default amber and the "astrophage red" preset (ink hue 8°; the UI's `--ink` /
  `--hot` follow the deck's ink).
- `drift-sweep-cpu.gif`: deck only, 13 s at 10 fps, deterministic (manual clock, `app.renderAt`). The answer's real tokens are
  re-pushed at 6 tok/s (the q7 mockup's pace; the stand-in measured 16-18 tok/s here, which ends the answer in 4 s), so the sweep
  is 0.15 s. The camera has been idle since t = 0, so the ±9° drift starts at 8 s and ramps in.
- `sweep-slowmo-strip-cpu.png`: one word sweeping L0 → L27, frames at 0.1, 0.35, 0.6, 0.85 and 1.2 × the sweep time
  (0.05 s at the measured 18 tok/s).
