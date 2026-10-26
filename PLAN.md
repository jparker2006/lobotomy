# Plan

## Step 0: de-risk the chip
Export Qwen3-0.6B to ONNX with a `[28, 3072]` steering input added at each layer's `down_proj` input, plus per-token activation outputs. Compile with Torq, run it on the board, and measure tokens/sec. If it fails, find out why before building anything else.

## Steps
1. **Reference model (Rust, candle)** with steering: `a_n += dose * median_resid_norm(L) / ||W_down[:, n]||`, plus a multiply mode.
2. **Neuron hunt (rented GPU):** push every neuron up and down on 4 fixed prompts, with 256 norm-matched random-direction controls per layer. Keep only neurons that beat the controls' 95th percentile. Label with Claude, and score each label by how well it predicts activations on held-out text, against a random-label baseline.
3. **Board server (Rust):** runs the compiled model through the IREE runtime and streams tokens plus activations over USB.
4. **Live UI:** a 3D map in the laptop browser where neurons light up per generated token. Qwiic knobs on the board push chosen neurons in real time.

## Done when
- **Parity:** the NPU and the fp32 candle reference match on the first 32 greedy tokens for at least 90% of a fixed steered test set.
- **Reality:** every neuron in the shipped atlas beats the random-control 95th percentile.
- **Labels:** label scores beat the random-label baseline on held-out text.
- **Live:** at least 8 tok/s on the board with activations streaming, and a knob turn changes the output within one token.
- **Stranger test:** someone new finds and steers a neuron in under 5 minutes.

## Why Qwen3-0.6B
It has no post-feedforward norm, unlike Gemma 3, so a push at the `down_proj` input moves the residual by exactly the intended amount. It is Apache 2.0, and Synaptics already compiles it for this NPU.
