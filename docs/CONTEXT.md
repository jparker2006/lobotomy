# Context handoff

Everything learned so far, so any new session can pick up without the original chat.

## The project
A live neuron explorer for Qwen3-0.6B running on a 1-watt NPU. The model runs offline on the Coralboard. Physical knobs push single MLP neurons while it talks, and a 3D map in the laptop browser shows every neuron lighting up per generated token. Inspired by Lyra's neuron-explorer (see below). The name "brainbox" is a placeholder.

## Decisions
- **Model:** Qwen3-0.6B (28 layers × 3,072 MLP neurons = 86,016). Not Gemma 3: its `post_feedforward_layernorm` breaks the exact steering math.
- **Language:** Rust everywhere except model export and compile, which need the Python Torq toolchain.
- **Split:** a rented GPU builds the neuron atlas once. The board runs the steered model live. The laptop browser shows the map.
- **License:** MIT, public repo.
- **Ambition:** high. This is a side project built mostly with Claude Code.

## Machines
| Machine | What it is | Notes |
|---|---|---|
| MacBook Air | Jake's laptop, Apple Silicon | `adb` installed via `brew install --cask android-platform-tools` |
| jake-desktop | Arch Linux, x86_64 | `adb` in `~/.local/bin`. No passwordless sudo. `gh` logged in. This is the machine Cloudroom cloud threads reach with `room-cli mac run`. |
| Cloud sandbox | aarch64 Linux | Can't run the Torq compiler. |

**The Torq compiler ships only as x86_64 Linux wheels** (`torq-compiler==2.3.0`, `manylinux_2_39_x86_64`). Compile on jake-desktop or a rented x86 GPU pod, not on the Mac or the cloud sandbox.

## The board
Synaptics Coralboard, Limited Edition 2GB, also called the Google I/O 2026 edition. The module is Grinn's AstraSOM-2619.
- **SoC:** Astra SL2619. 2× Cortex-A55 at 2 GHz running Yocto Linux. A Cortex-M52 in an always-on domain (FreeRTOS/Zephyr). Torq NPU, 1 TOPS, which includes Google's open-source RISC-V Coral NPU. Mali-G31 GPU.
- **Memory:** 2 GB DDR4, 16 GB eMMC, microSD.
- **I/O:** MIPI CSI (OV5647 camera in the box), MIPI DSI, USB-C (5 V power plus device-mode data, lower-right corner), USB-A host, 20-pin GPIO header (UART console pins), mikroBUS (hat in the box), Qwiic (I2C), I3C, M.2 E-key for Wi-Fi/BT, LiPo input with no charger, and power shunt test points.
- **Not on this board:** Ethernet, CAN, built-in Wi-Fi, a mic, an audio jack. The chip supports Ethernet and CAN, but the board doesn't break them out.
- **Power:** under 500 mW typical on the A55, under 50 mW on the M52 (vendor figures).

### Bring-up status: not booted yet
- The first try was plugged into the MacBook. Only the green LED lit; there was no red heartbeat (two short blinks, then a pause). `adb devices` was empty. Unclear whether the macOS "Allow accessory to connect?" prompt was approved.
- **Next checks:** use a known data cable plugged in directly, approve the macOS accessory prompt, press no buttons (reset during boot enters USB boot mode), and wait 60–90 s for the heartbeat. Then:
  `ioreg -p IOUSB -w0 -l | grep -E '"USB Product Name"|"USB Vendor Name"'` and `adb devices -l`.
- **Login:** `adb shell`, or serial at 115200 baud. User `root`, no password.
- **Internet:** macOS System Settings → General → Sharing → Internet Sharing to the "NCM" USB interface. Or add an M.2 Wi-Fi card.
- **Do not flash SDK v2.6.1.** Its release notes say the Google I/O edition is not supported. The I/O edition was supported in v2.4 and v2.5 (`sl2619_coralboard*` images at github.com/synaptics-astra/sdk/releases). How those images get onto the eMMC is unverified.
- **Docs:** Synaptics Coralboard getting started and user guide at developer.synaptics.com/docs/developer-kits/coralboard/sl/. Demos: github.com/synaptics-astra-demos/sl2610-examples.

## Toolchain
- **Compiler:** github.com/synaptics-torq/torq-compiler (MLIR/IREE, Apache 2.0). Input is ONNX or TFLite, output is `.vmfb`. Ops it can't place on the NPU fall back to the CPU.
- **Exporter:** github.com/synaptics-torq/torq-tools. `torq-export-model qwen -s qwen3-0.6b` does the following:
  1. Exports to ONNX with optimum.
  2. Makes a static single-token decoder with a 256-slot KV cache.
  3. Keeps the token embeddings on the host and trims the vocabulary to Latin text.
  4. Rewrites attention.
  5. Applies INT4 GPTQ weights (block 32, bf16 scales).
  6. Compiles `transformer.vmfb` and `lm_head.vmfb.trim`.

  Its README says the Torq model repos are private and you must run `hf auth login`. Unverified: `Synaptics/Qwen3-0.6B-GPTQ-int4-torq` looked public from the API.
- **Runner:** github.com/synaptics-torq/torq-examples (`qwen3/`, `functiongemma/`, `gemma3/`), in Python.
- **Speed (Synaptics' numbers):** Qwen3-0.6B 6.0–6.4 tok/s, Qwen2.5-0.5B 11.3 tok/s, Gemma 3 270M INT8 12.2 tok/s.
- **Already on the NPU** (huggingface.co/Synaptics): Moonshine speech-to-text, Piper TTS (vocoder on the NPU, about 3–4× real time), YOLOv8, SigLIP2, FunctionGemma 270M, LFM2/LFM2.5, Qwen2.5-0.5B, Qwen3-0.6B, and face ID.
- **Rust on the board:** the board runs aarch64 Linux, so Rust cross-compiles. The NPU runtime is IREE-based with a C API. The `eerie` crate wraps the IREE compiler and runtime. **Unverified** whether it can load the Torq NPU driver.
- **Open-source Coral NPU:** github.com/google-coral/coralnpu (RTL, plus MPACT and Verilator simulators that run without the board). There's also github.com/google-coral/coralnpu-compiler.

## Steering design (Step 0 target)
- Hook point: the input to each layer's `down_proj`, i.e. the neuron activations `h = silu(gate(x)) * up(x)`, shape `[*, 3072]`.
- Add mode: `h[n] += dose * median_resid_norm(L) / ||W_down[:, n]||`. This moves the residual by exactly `dose × median residual norm`, because Qwen3 has no post-MLP norm and `down_proj` has no bias.
- On the NPU, graphs are compiled ahead of time with static shapes and have no hooks. So add graph inputs `A[28, 3072]` (default 0) and optionally `M[28, 3072]` (default 1), and compute `h = h * M[l] + A[l]` before `down_proj`. Changing a dose then needs no recompile.
- For live visualisation, add per-layer activation outputs: 86,016 values per token, about 172 KB at bf16.
- Recalibrate the median residual norm and activation scales on the quantized graph, since host fp32 values won't match.
- Likely tool: `onnx_graphsurgeon` on the exported ONNX, before `torq-compile`.

## Lyra's neuron-explorer
github.com/lyramakesmusic/neuron-explorer @ f79b219. It has **no license**, so read it but don't copy code. 20 code-reader reports are in [docs/lyra-reports/](lyra-reports/). Key points:
- **Sweep:** each neuron is pushed on 4 prompts (`who`, `snake`, `orange`, `free`) at doses {−0.7, +0.5, +0.8} ×3 plus free at +3, measured in median-residual-norm units. Greedy decoding, 256 tokens. 256 random-direction controls per layer, norm-matched.
- **Ranking:** distinctive words versus the pool and the baselines. `z = (cnt − mean) / sqrt(mean + 0.5)`, mass = sum of the top 5 z, and `null_pct` against the controls. At Qwen3.5-4B L22, 19.6% of neurons beat every control.
- **Lessons:** controls are essential (one random direction made the model claim to be "Xiaomi AI"). "Did the output change?" is useless, because ±1 changes greedy output 83–98% of the time. Calibrate doses per layer. Early layers break at relative doses.
- **Precomputed tables:** logit lens (cheap), steer lens (±0.8 pushes on 4 contexts with KL, about 1,000× costlier), weight-only neuron connections (top 24 per neuron), and a 3D "writes" layout (UMAP of `W_down` columns).
- **Explorer:** FastAPI plus a three.js map (one points buffer, additive glow shader) plus a Claude agent. The agent runs the `claude` CLI with a 21-tool MCP server: steer, activations, logprobs, compare, fires, camera, and so on.
- **Agent method:** read the lenses, steer gently then hard, find the peak text, ablate upstream, try to falsify, write a blind sentence, check logprobs. Report format: Name, Verdict, +/− pole, Evidence, Test.
- **Bugs and risks found:** the agent gets unrestricted Bash in `$HOME`. Control RNG vectors go wrong on partial runs. Caches aren't keyed by model. The 3D layout only weakly groups similar neurons (Lyra's own numbers).
- **Cost:** Qwen3.5-4B full sweep about 44–112 GPU-hours. Gemma 270M estimated 2–15 GPU-hours (about $2–14). Qwen3-0.6B is roughly 2–3× Gemma 270M (estimate).

## Open questions
1. Does the board boot at all? See bring-up status above.
2. Can Torq compile Qwen3-0.6B with steering inputs and activation outputs, and how fast does it run?
3. Can Rust drive the Torq runtime (via `eerie` or bindgen on the C API)?
4. Where does compiling happen: jake-desktop or a GPU pod?
5. Final project name.
