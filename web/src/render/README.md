# web/src/render: the deck renderer

Draws all 86,016 MLP neurons (28 layers × 3,072) every frame, as specified in [docs/ui/design.md](../../../docs/ui/design.md).
three.js r186 `WebGPURenderer` + TSL. Built by U1 (#19), extended by U2 (#20: `setGlow`, `getView`, `visibleLayers`, `glowAt`). **Frozen after U2**: U3–U6 only call the API.

## API

```ts
import { createDeck, recordedRarityTable } from "lobotomy/render"; // import map -> /render/index.js

const deck = await createDeck(canvas, { tokPerSec: 6 }, { gpuTiming: false, forceWebGL: false });
deck.setRarityTable(F);          // pooled CDF of |r|: { logLo, logHi, cdf } on log-spaced bins (rarity.ts)
deck.pushToken(r, tSec);         // r: Float32Array(86016), signed activation / p99, layer-major [28][3072]
requestAnimationFrame(function tick(ts) { deck.frame(ts / 1000); requestAnimationFrame(tick); });
```

| Member | What it does |
|---|---|
| `backend` | `"webgpu"` or `"webgl2"` (three's fallback; the compute stages then run on the CPU, same math) |
| `params` | live knobs, read every frame: `tokPerSec`, `halfLifeTokens` (2), `sweepSec` (`null` = 0.9 / tok_s, `0` = off), `bloomScale` (bloom target size / canvas, default 0.5), `bloomStrength`, `hazeStrength`, `cellW` × `cellH` (5 × 8 CSS px), `dpr` (`null` = devicePixelRatio), `ink`, `hot`, `lattice`, `background`, `thresholds` ([.90, .97, .99, .997, .999]), `lattFade`, `lattFadeAmount`, `outlineAlpha`, `visibleLayers` (`null` = all; lock-on shows one layer: hidden layers take no cells, so crowding stays honest, and draw no outline). The page maps app state onto these in `app/params.ts` only |
| `pushToken(r, tSec)` | queues a token; it starts sweeping L0 → L27 at `tSec` (same clock as `frame`; times in the past start now) |
| `setRarityTable(F)` | the ladder's F. Default: `recordedRarityTable()` (the recorded CPU quantiles). U2 seeds it from the calibration and updates it with a running histogram |
| `frame(tSec)` | runs every stage and draws; call once per animation frame with a monotonic clock in seconds |
| `setGlow(state \| null)` | draws a fixed glow state (86,016 signed values) instead of the live trail; `null` = live again. The trail keeps running underneath. Word replay uses `glowAt(tokens, k, halfLifeTokens)` from `cpu.ts` (the trail at token k, computed on the CPU) |
| `camera`, `setView({eye, target, fov, center})`, `getView()` | the overview of `final-cinema-astrophage.png` by default; `center` is the principal point as a fraction of the canvas. U4's rig can drive `camera` directly |
| `project(xyz)`, `layerLabel(l)` | CSS px of a world point / a layer's label anchor (for DOM labels) |
| `pick(x, y)` | `{ neuron, layer, index, level, sign }` of the neuron that set the dot under a CSS-px point (one GPU readback) |
| `snapshot(tSec)` | draws one frame offscreen, returns RGBA pixels (stills, clips, headless checks) |
| `gpuTimes()` | mean GPU ms per frame (compute, render) since the last call, with `opts.gpuTiming` and `timestamp-query` |
| `size` | `{ cssW, cssH, dpr, pxW, pxH, cols, rows }`; the canvas follows its CSS size |

## Stages (all every frame)

1. **Trail + sweep** (compute, one thread per neuron): `glow = maxHold(glow · 0.5^(dt/HL), r · 0.5^(age/HL))` once a token reaches the layer at `t0 + (l/28) · sweepSec`. Signed; `maxHold` keeps the larger |value|. 4 token slots in flight.
2. **Splat** (compute, per neuron): project, then per 5 × 8 CSS-px cell `atomicMax(key)` and `atomicAdd(count)`. Key = 15 bits of quantized log|glow| ≪ 17 | neuron index, so the winner is recoverable.
3. **Resolve** (compute, per cell): `u = F(|glow[winner]|)^k` → level 0–5, sign, view depth, winner; clears the cell. Writes one `vec4` per cell.
4. **Glyph pass**: one instanced quad per cell, sized to its dot, SDF disks and rings (`· ∙ • ● ◉` / `∘ ○ ◯ ◎`), amber ink, hot ink for levels 4–5, dim lattice for level 0 with the view-only depth fade. Sheet outlines are three.js lines.
5. **Bloom + haze**: the top-1% cells (level ≥ 3) drawn again into an emissive target at `bloomScale`; blur σ 3 CSS px (full target), σ 10 at 1/4, haze σ 40 at 1/8; added onto the canvas.

`cpu.ts` holds stages 1–3 on the CPU: the WebGL2 fallback, and the reference the tests check (`web/test/render.test.ts`).
`synth.ts` makes **SYNTHETIC** tokens with the recorded |r| distribution, for the bench and tests; label anything drawn from it.

## Notes

- Colours are display values; the renderer outputs without colour management (blends like the canvas mockups).
- Index order inside a sheet is arbitrary until R9 (64 × 48 row-major).
- The kernels are written in TSL through an untyped alias (`TSL as any`); tsc checks the rest.
