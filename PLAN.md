# Plan

**lobotomy is Qwen running on my Coralboard so you can watch its neurons light up as it talks, and later push them.**

## Versions
| Version | What you get |
|---|---|
| **v0.0.1 Talks + Glows** | Qwen3-0.6B runs on the board's NPU. A plain page, served by the board, streams the answer and lights up all 86,016 neurons for each word. Spec: [#1](https://github.com/jparker2006/lobotomy/issues/1). |
| v0.0.2 Beautiful | A UI grill with rendered mockups, then the real 3D UI (TypeScript + three.js). |
| v0.0.3 Hints | Hover a neuron to see the words it pushes toward (logit lens, computed from the weights with no GPU). |
| v0.1.0 First post | The atlas, steering sliders, research-grade checks and the beautiful UI. Being mapped with a wayfinder in [docs/v0.1.0-map.md](docs/v0.1.0-map.md). |

## v0.0.1: Talks + Glows
The spec, done criteria and tickets live in GitHub issues, so their state can't drift from this file. The spec is [#1](https://github.com/jparker2006/lobotomy/issues/1). It closes when all its sub-issues are closed and every done box is checked.

| Ticket | Issue |
|---|---|
| T1 Boot the board (physical) | [#2](https://github.com/jparker2006/lobotomy/issues/2) |
| T2 Tracer bullet on CPU | [#3](https://github.com/jparker2006/lobotomy/issues/3) |
| T3 Talks: Synaptics' Qwen3 on the board | [#4](https://github.com/jparker2006/lobotomy/issues/4) |
| T4 Reproduce Synaptics' compile on RunPod | [#5](https://github.com/jparker2006/lobotomy/issues/5) |
| T5 CPU 4-bit reference | [#6](https://github.com/jparker2006/lobotomy/issues/6) |
| T6 Glow graph | [#7](https://github.com/jparker2006/lobotomy/issues/7) |
| T7 Parity check | [#8](https://github.com/jparker2006/lobotomy/issues/8) |
| T8 Glow on the board | [#9](https://github.com/jparker2006/lobotomy/issues/9) |
| T9 Acceptance (physical) | [#10](https://github.com/jparker2006/lobotomy/issues/10) |

## Notes for v0.1.0 (input to the wayfinder)
Earlier targets, kept so the wayfinder can sharpen them:
- **Reference model** with steering: `a_n += dose * median_resid_norm(L) / ||W_down[:, n]||`, plus a multiply mode.
- **Neuron hunt (rented GPU):** push every neuron up and down on fixed prompts, against norm-matched random-direction controls per layer. Label the neurons with Claude, and score each label by how well it predicts activations on held-out text, against a random-label baseline.
- **Statistics warning:** "beats the controls' 95th percentile" across 86,016 neurons would let about 4,300 neurons through even if none were real. Use empirical p-values with FDR correction (Benjamini–Hochberg).
- **Parity:** the NPU and the reference match on the first 32 greedy tokens for at least 90% of a fixed steered test set.
- **Live:** a slider move changes the output within one token, at ≥ 5 tok/s with activations streaming.
- **Stranger test:** someone new finds and steers a neuron in under 5 minutes.

## Why Qwen3-0.6B
It has no post-feedforward norm, unlike Gemma 3, so a push at the `down_proj` input moves the residual by exactly the intended amount. It is Apache 2.0, and Synaptics already compiles it for this NPU.
