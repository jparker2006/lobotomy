# Plan

**lobotomy is Qwen running on my Coralboard so you can watch its neurons light up as it talks, and later push them.**

## Versions
| Version | What you get |
|---|---|
| **v0.0.1 Talks + Glows** | Qwen3-0.6B runs on the board's NPU. A plain page, served by the board, streams the answer and lights up all 86,016 neurons for each word. Spec below. |
| v0.0.2 Beautiful | A UI grill with rendered mockups, then the real 3D UI (TypeScript + three.js). |
| v0.0.3 Hints | Hover a neuron to see the words it pushes toward (logit lens, computed from the weights with no GPU). |
| v0.1.0 First post | The atlas, steering sliders, research-grade checks and the beautiful UI. Being mapped with a wayfinder in [docs/v0.1.0-map.md](docs/v0.1.0-map.md). |

## v0.0.1 spec
**What you do:** plug the board into the Mac, open `localhost:8000`, and type a question. The board answers word by word, and a flat grid lights up for each word: 28 rows (layers) × 3,072 dots (neurons).

**In scope:**
- Synaptics' Qwen3-0.6B (INT4 GPTQ, non-thinking mode), recompiled so the NPU also outputs each layer's `down_proj` input.
- A Python server on the board, using the stdlib plus numpy where possible.
- A plain TypeScript page.
- Per-neuron brightness, calibrated on the board.

**Out of scope:** steering, the atlas, hover hints, 3D, any GPU.

**Limits:** prompt plus answer must fit in 256 tokens, which is fixed at export. Non-thinking mode, so the glow starts right away.

### Done when
1. The board boots and `adb shell` works from the Mac.
2. Synaptics' Qwen3 demo chats on the board.
3. Our recompiled Qwen also outputs all 86,016 neuron values per word, at ≥ 5 tok/s. (Synaptics gets 6.0–6.4 without those outputs.)
4. A page served by the board opens in the Mac's browser. You type a question, the answer streams in, and the 28 × 3,072 grid lights up for each word.
5. **Sanity check:** on fixed test prompts, with the same tokens fed in, the board's activations match the same 4-bit model running on CPU (per-layer cosine ≥ 0.99). The gap to full precision is measured and recorded.

### Tickets
These run in tracer-bullet order: first a thin path through the whole stack, then real parts swapped in. **Physical** means the ticket needs Jake's hands.

**T1. Boot the board.** Physical (Jake + agent). Blocked by: nothing.
- Plug a known data cable straight into the Mac. Approve the macOS accessory prompt, press no buttons, and wait 60–90 s for the red heartbeat. The agent checks `ioreg` and `adb devices -l`.
- **Done when:** `adb shell` gives a root prompt, and CONTEXT.md records the OS/SDK version, Python, glibc, RAM and disk.
- **If it fails:** use the serial console at 115200 baud on the 20-pin header (needs a USB-UART adapter). Reflash only from SDK v2.4 or v2.5, never v2.6.1.

**T2. Tracer bullet on CPU.** Agent. Blocked by: nothing.
- Build the whole v0.0.1 path with the board swapped for Qwen3-0.6B running on the cloud sandbox's CPU. A Python server streams tokens plus all 86,016 activations per word, and a plain TypeScript page draws the grid. Use the same frame format the board will send.
- **Done when:** Jake opens it through a Cloudroom preview, asks a question, and watches the text stream and the grid light up.
- **Why first:** it proves the server, the protocol and the page without waiting on the board.

**T3. Talks: Synaptics' Qwen3 on the board.** Agent, plus Jake for Internet Sharing. Blocked by: T1.
- Get files onto the board, either through macOS Internet Sharing to the NCM interface or by downloading on the Mac and running `adb push`. Install `torq-runtime==2.3.0` following torq-examples, then run `python src/infer.py --instruct-model`.
- **Done when:** a chat works on the board, and tok/s is recorded (Synaptics gets 6.0–6.4).

**T4. Reproduce Synaptics' compile on RunPod.** Agent. Jake adds $10 of credit and supplies an API key through a Cloudroom secret. Blocked by: nothing for the compile; the board check needs T3.
- Start an Ubuntu 24.04 CPU pod, install `torq-compiler==2.3.0` and torq-tools from git, and run `torq-export-model qwen -s qwen3-0.6b`. Record the time and peak RAM.
- **Done when:** our own `transformer.vmfb` gives the same answers and tok/s on the board as the downloaded one. We reproduce the stock build before modifying anything.

**T5. CPU 4-bit reference.** Agent. Blocked by: nothing; the final check needs T3.
- Run Qwen3 in PyTorch with Synaptics' GPTQ weights (`gptq/model.safetensors`) dequantized, bf16 activations, the same trimmed vocabulary (`token_id_lut.npy`), and a hook on every `down_proj` input.
- **Done when:** its greedy answers match the board's stock Qwen for the first 32 tokens on the test prompts, and its gap to full precision (per-layer cosine and top-100 overlap) is recorded.

**T6. Glow graph.** Agent. Blocked by: T4.
- Add each layer's `down_proj` input (28 × 3,072, bf16) as an output of the exported ONNX before `torq-compile`. Check on CPU that the logits don't change, then compile with the same flags. The prediction is under 1% slowdown (CONTEXT.md, "Cost of glowing").
- **Done when:** it compiles and runs on the board, outputs 86,016 values per word, and runs at ≥ 5 tok/s.

**T7. Parity check.** Agent. Blocked by: T5, T6.
- With the same tokens fed in on fixed prompts, compare the board's activations to the T5 reference, layer by layer.
- **Done when:** per-layer cosine is ≥ 0.99 against the 4-bit reference, and the gap to full precision is recorded. This is done criterion 5.

**T8. Glow on the board.** Agent. Blocked by: T2, T6.
- Swap the tracer's CPU backend for the board. The server runs on the board, `adb forward tcp:8000 tcp:8000` maps it to the Mac, and the page opens at `localhost:8000`. Calibrate per-neuron brightness on the board with a few hundred tokens of text.
- **Done when:** done criteria 3 and 4 pass.

**T9. Acceptance.** Physical (Jake). Blocked by: T7, T8.
- Plug in the board, open `localhost:8000`, ask a question and watch the grid.
- **Done when:** all 5 done criteria are checked and recorded in CONTEXT.md. Tag `v0.0.1`.

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
