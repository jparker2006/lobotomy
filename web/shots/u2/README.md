# U2 sandbox renders (#20)

Taken in the cloud sandbox (no GPU): Chromium 156 with SwiftShader WebGPU (DPR 2) or llvmpipe WebGL2 (DPR 1), deck drawn via
`deck.snapshot()` because this sandbox never composites WebGPU canvases. Timings from here mean nothing for the Mac.

- `cpu-ocean-*`: the CPU stand-in (Qwen3-0.6B), "Describe the ocean at night in three sentences." Word 52 and the busiest word
  (1,391 neurons in their top 1%) shown with word replay (`app.pin`); `done-hold` is the held state after the last token.
- `fake-*`: the `fake` backend (random activations, labelled SYNTHETIC in the HUD): live mid-stream on WebGL2, and lock-on
  (`store.locked` = L8 + `camera.faceLayer(8)`), which draws only the locked layer.
