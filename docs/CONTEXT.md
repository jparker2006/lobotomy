# Context handoff

Everything learned so far, so any new session can pick up without the original chat.

## The project
**lobotomy is Qwen running on my Coralboard so you can watch its neurons light up as it talks, and later push them.**

Qwen3-0.6B runs offline on the Coralboard's 1-watt NPU, and the board also serves the web page. The Mac's browser only draws: a map where all 86,016 MLP neurons light up for each generated word. It's a hobby demo built for wow (and Twitter), not a product. The wow is two things: it runs on a real chip, and it looks beautiful. Inspired by Lyra's neuron-explorer (see below), but aiming higher on ML rigor and design. Versions and the v0.0.1 spec are in [PLAN.md](../PLAN.md).

## Decisions (grill with Jake, 2026-10-26)
- **Name:** lobotomy (was brainbox). It's also slang for a model that got dumber after editing, which is what a too-hard push does. The GitHub repo was renamed to `jparker2006/lobotomy` on 2026-10-26; the old URL redirects.
- **Terms:** a *neuron* is one value at a layer's `down_proj` input (28 layers × 3,072). *Steering* adds a dose to it on every token. A multiply mode (×0 switches it off) comes later. We push activations, never weights.
- **Model:** Qwen3-0.6B. Not Gemma 3: its `post_feedforward_layernorm` breaks the exact steering math.
- **Where things run:** the board runs the model and the web server. The Mac's browser draws the map at `localhost:8000` via `adb forward`. Later, a rented GPU builds the neuron atlas once. Physical knobs are dropped. Steering will use on-screen sliders.
- **Languages:** Python on the board, because Synaptics' runtime and examples are Python. TypeScript + three.js in the browser. Export and compile are Python too (the Torq toolchain). No Rust for now: it has no natural place, and whether it can drive the Torq runtime is unproven. Revisit only if Python becomes the bottleneck.
- **Compute:** a RunPod x86 CPU pod for export and compile, with a $10 cap for v0.0.x (prepaid, auto-pay off). The agent drives it with `runpodctl` and Jake's API key, requested through a Cloudroom secret and never pasted in chat. No GPU before the atlas. The atlas budget gets decided later (Lyra's numbers suggest $10–50). jake-desktop is the free fallback.
- **UI:** plain in v0.0.1. v0.0.2's design is in [docs/ui/design.md](ui/design.md) (spec #18). The target browser is **Chrome only** (Jake doesn't use Safari, 2026-10-28). The U1 benchmark measured 60.0 fps on his M4 Air in Chrome at 3308×2146 with glow on.
- **Process:** v0.0.1 goes grill → spec → tickets, published as GitHub issues (spec #1, tickets #2–#10 as sub-issues). v0.1.0 is mapped with a wayfinder, because its path has real unknowns.
- **License:** MIT, public repo.

## Machines
| Machine | What it is | Notes |
|---|---|---|
| MacBook Air (`Jakes-MacBook-Air-2`) | Jake's laptop, Apple Silicon (Mac16,13), 16 GB, macOS 26.5, no SD slot | The machine Cloudroom cloud threads reach with `room-cli mac run`. Has `adb` (`brew install --cask android-platform-tools`), `uv` and `brew`. No PyTorch, no Docker. The disk is 90% full (~44 GB free). The board plugs in here. |
| jake-desktop | Arch Linux, x86_64, remote | `adb` in `~/.local/bin`. No passwordless sudo. `gh` logged in. Cloud threads can't reach it, and the Mac has no SSH route to it. It could run its own Local Cloudroom thread. |
| RunPod CPU pod | x86_64 Linux, rented | Export and compile. Use Ubuntu 24.04 (glibc 2.39, Python 3.12) to match the `torq-compiler` wheel. Delete it when idle. |
| Cloud sandbox | aarch64 Linux, 4 cores | Can't run the Torq compiler. Runs PyTorch on CPU fine (the steering demo ran here). The real memory limit is 8 GB, even though `free` shows 247 GB. **Outbound SSH is blocked** (only HTTP/HTTPS/WebSocket get out), so pods are reached through a WebSocket tunnel (see RunPod below). |

## The board
Synaptics Coralboard, Limited Edition 2GB, also called the Google I/O 2026 edition. The module is Grinn's AstraSOM-2619.
- **SoC:** Astra SL2619. 2× Cortex-A55 at 2 GHz running Yocto Linux. A Cortex-M52 in an always-on domain (FreeRTOS/Zephyr). Torq NPU, 1 TOPS, which includes Google's open-source RISC-V Coral NPU. Mali-G31 GPU.
- **Memory:** 2 GB DDR4, 16 GB eMMC, microSD.
- **I/O:** MIPI CSI (OV5647 camera in the box), MIPI DSI, USB-C (5 V power plus device-mode data, lower-right corner), USB-A host, 20-pin GPIO header (UART console pins), mikroBUS (hat in the box), Qwiic (I2C), I3C, M.2 E-key for Wi-Fi/BT, LiPo input with no charger, and power shunt test points.
- **Not on this board:** Ethernet, CAN, built-in Wi-Fi, a mic, an audio jack. The chip supports Ethernet and CAN, but the board doesn't break them out.
- **Power:** under 500 mW typical on the A55, under 50 mW on the M52 (vendor figures).

### Board layout (checked against Jake's photo, 2026-10-26)
Jake's board matches Synaptics' "Coralboard SL" photo, not the older Google I/O photo with a black "ON/1" DIP switch.
- **Boot switch:** one slide switch labelled **eMMC / xSPI**, next to the USB-A port. It's in the same position as Synaptics' product photo (factory default, eMMC). Leave it.
- **Buttons:** both are on the right edge, side-facing. **RESET** is near the USB-C port, **USER** near the bottom. Holding USER while tapping RESET enters USB download mode (the chip's ROM shows up over USB even with no OS).
- **LEDs:** green = power, red = heartbeat (two blinks, pause) once Linux runs, blue = storage activity.
- **Factory image:** Astra SDK v2.2 customised by Grinn, on eMMC (I/O edition guide). Official support is v2.4 and v2.5. Updates go to eMMC only, with github.com/synaptics-astra/usb-tool and `sl2619_coralboard_oobe_scarthgap_6.12_v2.5.0.zip` from the SDK releases.
- **mikroBUS SPI:** `SM_SPI1_SDO` is a boot-strap pin (`BOOT_SRC1`), so a hat driving it can change the boot mode.

### Bring-up status: not booted yet (2026-10-26 evening)
- **The board is a Coralboard SL, not the Google I/O edition.** It has the eMMC/xSPI slider and an empty footprint where the I/O edition's black "ON/1" S3 switch sits, and it matches Synaptics' SL product photo. Synaptics: "official SL support began in Astra SDK 2.6.1; 2.4 and 2.5 supported the Coralboard Google I/O Edition only". The v2.6.1 notes say "Coralboard SL only. Google I/O edition not supported". So the old rule "don't flash v2.6.1" was written for the wrong edition. Jake still has to decide before anything is written to the board.
- **The cable and the chip work.** In USB download mode (hold USER, tap RESET) the Mac sees `06CB:019E` "TinyUSB Device", the Coralboard ID in usb-tool's manifest. Normal power-on still gives green only and no USB device, so the OS on storage isn't starting.
- **RAM boot attempt:** `sudo ./astra-boot usbBOOTimg` (usb-tool `bin/mac/arm64/astra-boot` v2.0.4, image `sl2619_coralboard_usb_boot_scarthgap_6.12_v2.6.1.zip`, kit in `~/lobotomy-boot` on the Mac) sent key.bin, spk.bin and m52bl.bin, then failed with `Failed to run SL26XX SysMgr boot sequence`. Afterwards the board showed as `06CB:019E` "SL2610 Bootloader". The RAM disk includes adbd and NCM, so a successful RAM boot gives `adb shell` with nothing written.
- **Next try:** set macOS "Allow accessories to connect" to "Automatically allow when unlocked" for the test (the chip changes USB identity mid-boot), re-enter download mode, then run `sudo ./astra-boot -D -S -l ~/lobotomy-boot/boot.log usbBOOTimg` and read the log. If the RAM boot works, choose a permanent install: SD card (v2.6.1 wic, needs a microSD and a USB-C card reader, since the MacBook Air has no SD slot) or eMMC with astra-update.

### Earlier bring-up notes
- The first try was plugged into the MacBook. Only the green LED lit; there was no red heartbeat (two short blinks, then a pause). `adb devices` was empty. Unclear whether the macOS "Allow accessory to connect?" prompt was approved.
- **2026-10-26:** with the switch confirmed at eMMC, only green lit again, and the Mac saw **no USB device at all**, not even an unnamed one. Next test: USB download mode (hold USER, tap RESET). If the Mac sees a device, the cable carries data and the OS isn't starting, so reinstall v2.5 with usb-tool. If not, the cable is probably charge-only.
- **Earlier checks:** use a known data cable plugged in directly, approve the macOS accessory prompt, press no buttons (reset during boot enters USB boot mode), and wait 60–90 s for the heartbeat. Then:
  `ioreg -p IOUSB -w0 -l | grep -E '"USB Product Name"|"USB Vendor Name"'` and `adb devices -l`.
- **Login:** `adb shell`, or serial at 115200 baud. User `root`, no password.
- **First facts to record:** OS and SDK version, `python3 --version`, `ldd --version`, free RAM and disk. The `torq-runtime` 2.3.0 wheels are `cp312` / `manylinux_2_39_aarch64`, so the board needs **Python 3.12 and glibc ≥ 2.39**. Yocto Scarthgap (5.0) ships both, if that's what the image is built on.
- **Web page on the Mac:** `adb forward tcp:8000 tcp:8000`, then open `localhost:8000`.
- **Internet:** macOS System Settings → General → Sharing → Internet Sharing to the "NCM" USB interface. Or download on the Mac and `adb push`. Or add an M.2 Wi-Fi card.
- **Do not flash SDK v2.6.1.** Its release notes say the Google I/O edition is not supported. The I/O edition was supported in v2.4 and v2.5 (`sl2619_coralboard*` images at github.com/synaptics-astra/sdk/releases). How those images get onto the eMMC is unverified.
- **Docs:** Synaptics Coralboard getting started and user guide at developer.synaptics.com/docs/developer-kits/coralboard/sl/. Demos: github.com/synaptics-astra-demos/sl2610-examples.

## Toolchain
Everything below is from Synaptics' public repos, checked 2026-10-26.
- **Compiler:** github.com/synaptics-torq/torq-compiler (MLIR/IREE, Apache 2.0). Input is ONNX or TFLite, output is `.vmfb`. Ops it can't place on the NPU fall back to the CPU. On PyPI, `torq-compiler==2.3.0` ships only `cp312-manylinux_2_39_x86_64`. It also provides simulation of the hardware.
- **Runtime:** `torq-runtime==2.3.0` on PyPI, `cp312-manylinux_2_39` for both aarch64 and x86_64. Its source is open, including the IREE HAL driver in C (`runtime/driver/torq_driver.c` in torq-compiler) and Python bindings.
- **Exporter:** github.com/synaptics-torq/torq-tools (install from git, it's not on PyPI). `torq-export-model qwen -s qwen3-0.6b` does the following:
  1. Exports to ONNX with optimum.
  2. Makes a static single-token decoder with a 256-slot KV cache (`-t` changes it).
  3. Keeps the token embeddings on the host and trims the output vocabulary to Latin text, punctuation and digits (`token_id_lut.npy`).
  4. Rewrites attention (read-only KV cache inputs, new rows as outputs).
  5. Applies INT4 GPTQ weights (block 32, bf16 scales), downloaded from the model repo's `gptq/` folder.
  6. Compiles `transformer.vmfb` and `lm_head.vmfb.trim` with the flags in `compile_flags.json`: `--torq-disable-slicing`, `--torq-prefetch-algorithm=stream-schedule` (hides compute under weight loads, since decode is DMA-bound), and `--torq-max-nss-programs-size=12582912` (transformer only).

  It needs only a CPU. Making new GPTQ weights takes about 2 h on 8 cores, and we only need that for a different model.
- **Model repo:** `Synaptics/Qwen3-0.6B-GPTQ-int4-torq` is **public, not gated, Apache 2.0**, even though torq-tools' README calls the Torq repos private. Files: `transformer.vmfb`, `lm_head.vmfb.trim`, `token_embeddings.npy`, `token_id_lut.npy`, tokenizer, config, and `gptq/model.safetensors`. The only tag is `v2.3.0`.
- **Runner:** github.com/synaptics-torq/torq-examples (Python, Apache 2.0). Qwen3: `python src/infer.py --instruct-model` for non-thinking mode. Prompt plus answer must fit in the 256-token context set at export. Thinking mode adds 100–300 hidden tokens (20–50 s) before the answer. There's also a `profile.py`.
- **Speed:** on an SL2610 at 1 GHz, the Qwen3-0.6B transformer takes 155.5 ms and the LM head 15.7 ms per token, giving 6.0–6.4 tok/s in the chat demo. Qwen2.5-0.5B gets 11.3 tok/s and Gemma 3 270M INT8 gets 12.2.
- **I/O limit:** the runtime allows at most 64 buffer bindings per dispatch (`IREE_HAL_EXECUTABLE_MAX_BINDING_COUNT 64`, synaptics-torq/iree `executable_library.h:380`), and Qwen already uses about 59. Glow must be one `[28, 3072]` output, and steering one `[28, 3072]` input.
- **Runner quirk:** torq-examples' `ManagedSelfAttnCacheRunner` treats every output after the first as a KV cache. Extra outputs go at index 1 with `cache_start_idx=2`.
- **KV layout change:** on 2026-10-23 torq-tools `0f4af63` made a read-only KV cache the default: 8.5 vs 6.0 tok/s on SL2610. The v2.3.0 files on Hugging Face use the old layout (`--no-read-only-kv-cache`). Running the new layout needs torq-examples ≥ `290a95c`.
- **INT4 details (T5, `ref/README.md`):** every transformer matrix and the LM head use `w = (q − zp) · bf16(scale)` in blocks of 32 along the input dimension. Scales are truncated, not rounded, to bf16. The NPU never builds dequantized weights; per block it sums `x·q` and `x` in bf16. The trimmed vocabulary has 99,477 tokens, padded to 99,520. GPTQ vs fp32: KL 0.107, top-1 85.4% (matches Synaptics' published numbers).
- **Cost of glowing (prediction):** each token streams about 250 MB of INT4 weights (440M non-embedding params at 4 bits, plus scales). Activation outputs add 86,016 × 2 B ≈ 172 KB, about 0.1%. Expect under 1% slowdown, unless the extra outputs break the compiler's fusion. Measure it.
- **Open vs closed:** open (Apache 2.0) are the compiler, runtime and driver, export tools, examples, a simulator, a QEMU fork with Torq extensions (`synaptics-torq/qemu`), an MPACT simulator for the Coral NPU, and Google's coralnpu RTL and compiler. Closed are the silicon and, as far as we can see, the rest of the Torq NPU design.
- **Already on the NPU** (huggingface.co/Synaptics): Moonshine speech-to-text, Piper TTS (vocoder on the NPU, about 3–4× real time), YOLOv8, SigLIP2, FunctionGemma 270M, LFM2/LFM2.5, Qwen2.5-0.5B, Qwen3-0.6B, and face ID.
- **Licenses:** Synaptics' code is Apache 2.0, so adapting it is fine if we keep their notices. Lyra's has no license, so we never copy it.

## Steering design (for v0.1.0)
- Hook point: the input to each layer's `down_proj`, i.e. the neuron activations `h = silu(gate(x)) * up(x)`, shape `[*, 3072]`.
- Add mode: `h[n] += dose * median_resid_norm(L) / ||W_down[:, n]||`. This moves the residual by exactly `dose × median residual norm`, because Qwen3 has no post-MLP norm and `down_proj` has no bias.
- On the NPU, graphs are compiled ahead of time with static shapes and have no hooks. So add graph inputs `A[28, 3072]` (default 0) and optionally `M[28, 3072]` (default 1), and compute `h = h * M[l] + A[l]` before `down_proj`. Changing a dose then needs no recompile.
- For live visualisation (v0.0.1), add per-layer activation outputs: 86,016 values per token, about 172 KB at bf16.
- Recalibrate the median residual norm and activation scales on the quantized graph, since host fp32 values won't match.
- Likely tool: `onnx_graphsurgeon` on the exported ONNX, before `torq-compile`.

## CPU steering demo (2026-10-26)
Qwen3-0.6B in fp32 on the cloud sandbox's CPU. The prompt was "Describe your ideal Sunday in three sentences.", decoded greedily with the push applied at every position. Scripts are in [scripts/cpu-demo/](../scripts/cpu-demo/).
- **Finding neurons:** compare mean activations on 5 ocean sentences against 6 neutral ones (`probe.py`), then check each candidate's logit lens (the words its `W_down` column pushes toward).
- **Layer 8, neuron 908** (lens: 海底, 航行, 海岸, shore, reef, ocean). At +0.5: "A serene and fulfilling Sunday at the heart of a tranquil ocean, where the waves whisper…". At +1 it drifts and gets odd. At +2 and +4 it's garbage.
- **Layer 19, neuron 834** (lens of the negative push: 船, marine, fish, sailed, sailor). At −0.5: "relax by the water… fresh seafood". At −1: "swim in the ocean… sailing". At −2 it's garbage.
- **Controls:** three random residual directions of the same size (+0.5 at layer 8) changed the text ("countryside", "park") but never moved it toward the ocean.
- **Lessons:** one neuron can pull a topic. The safe dose range is narrow and differs by layer. Only the controls prove a neuron is special. On this prompt the median residual norm was 29.7 at layer 8 and 145.4 at layer 19.

## RunPod
- **Account:** prepaid, $10 credit, auto-pay off. The API key is in the cloud sandbox at `/code/brainbox-demo/.env.runpod`, outside the repo.
- **CPU prices:** `cpu3g` (4 GB RAM per vCPU) costs $0.04 per vCPU-hour, so 8 vCPUs and 32 GB are $0.32/hr. `cpu3c` (2 GB per vCPU) is $0.03 and `cpu3m` (8 GB per vCPU) is $0.055. Get them from `GET https://api.runpod.io/v2/catalog/cpus`.
- **Recipe:** create pods with `POST /v2/pods` using `{"cpu": {"id": "cpu3g", "vcpuCount": 8}, "image": "ubuntu:24.04", "ports": ["8080/http"], "env": {"PUBLIC_KEY": ...}, "cmd": ["bash", "-c", ...]}`. The start command installs openssh-server, writes the key, starts `sshd` and runs `websocat --binary ws-l:0.0.0.0:8080 tcp:127.0.0.1:22`. The sandbox connects with `ProxyCommand websocat --binary wss://<podId>-8080.proxy.runpod.net`. `runpod/base` images didn't answer on SSH.
- **Toolchain on the pod:** `uv venv --python 3.12`, then `uv pip install ./torq-tools torq-compiler==2.3.0 --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match`. This takes 15 s. torq-tools was at `0319541`.

## Lyra's neuron-explorer
github.com/lyramakesmusic/neuron-explorer @ f79b219. It has **no license**, so read it but don't copy code. 20 code-reader reports are in [docs/lyra-reports/](lyra-reports/). She rented RunPod GPUs (4× RTX 4090) for the sweep and ran the explorer as a localhost site on her own machine. She never used a board. Key points:
- **Sweep:** each neuron is pushed on 4 prompts (`who`, `snake`, `orange`, `free`) at doses {−0.7, +0.5, +0.8} ×3 plus free at +3, measured in median-residual-norm units. Greedy decoding, 256 tokens. 256 random-direction controls per layer, norm-matched.
- **Ranking:** distinctive words versus the pool and the baselines. `z = (cnt − mean) / sqrt(mean + 0.5)`, mass = sum of the top 5 z, and `null_pct` against the controls. At Qwen3.5-4B L22, 19.6% of neurons beat every control.
- **Lessons:** controls are essential (one random direction made the model claim to be "Xiaomi AI"). "Did the output change?" is useless, because ±1 changes greedy output 83–98% of the time. Calibrate doses per layer. Early layers break at relative doses.
- **Precomputed tables:** logit lens (cheap), steer lens (±0.8 pushes on 4 contexts with KL, about 1,000× costlier), weight-only neuron connections (top 24 per neuron), and a 3D "writes" layout (UMAP of `W_down` columns).
- **Explorer:** FastAPI plus a three.js map (one points buffer, additive glow shader) plus a Claude agent. The agent runs the `claude` CLI with a 21-tool MCP server: steer, activations, logprobs, compare, fires, camera, and so on.
- **Agent method:** read the lenses, steer gently then hard, find the peak text, ablate upstream, try to falsify, write a blind sentence, check logprobs. Report format: Name, Verdict, +/− pole, Evidence, Test.
- **Bugs and risks found:** the agent gets unrestricted Bash in `$HOME`. Control RNG vectors go wrong on partial runs. Caches aren't keyed by model. The 3D layout only weakly groups similar neurons (Lyra's own numbers).
- **Cost:** Qwen3.5-4B full sweep about 44–112 GPU-hours. Gemma 270M estimated 2–15 GPU-hours (about $2–14). Qwen3-0.6B is roughly 2–3× Gemma 270M (estimate).

## Open questions
1. Does the board boot at all? (ticket T1)
2. Do the board's Python and glibc match the `torq-runtime` 2.3.0 wheels (3.12, ≥ 2.39)? (T3)
3. Can Torq compile Qwen3-0.6B with 28 activation outputs, and what does it cost in speed? (T6)
4. Do the board's 4-bit activations match a CPU reference? (T7)
5. Everything for v0.1.0 is in the wayfinder map, [docs/v0.1.0-map.md](v0.1.0-map.md): done criteria E1–E6, risks R1–R14, and 14 open questions for Jake, each with a recommended answer.
