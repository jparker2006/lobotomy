# CPU 4-bit reference (T5)

`int4_ref.py` runs Qwen3-0.6B on CPU with exactly the weights, embeddings and vocabulary of
Synaptics' board model (`Synaptics/Qwen3-0.6B-GPTQ-int4-torq`, tag `v2.3.0`), so the board can be
checked against it. `board_greedy.py` runs the stock model on the board for the same check.

```sh
uv venv -p 3.12 .venv && uv pip install -p .venv torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -p .venv transformers safetensors numpy huggingface_hub   # ran with torch 2.14.1, transformers 5.20.2
.venv/bin/python ref/int4_ref.py greedy          # -> eval/ref_greedy.json (32 tokens per test prompt)
.venv/bin/python ref/int4_ref.py gap             # -> eval/int4_vs_fp32.json
.venv/bin/python ref/int4_ref.py synaptics       # -> eval/synaptics_compare.json
.venv/bin/python ref/int4_ref.py dump --out acts.npz   # teacher-forced logits + [T,28,3072] activations
```

Files download from Hugging Face on first use (base model pinned to a commit, Torq repo to `v2.3.0`).
The sandbox has an 8 GB memory cap; one model plus results fits.

## What the board computes (settled from source)

Synaptics' code below is Apache 2.0. Lines are torq-compiler tag `v2.3.0` (1e4bbce), torq-tools
`main` (0319541) and torq-examples `main` (f99b491).

| Fact | Source |
|---|---|
| All 196 transformer Linears and the LM head are INT4 GPTQ (llmcompressor, asymmetric, group 32 along the **input** dim K). The checkpoint holds 197 modules. | `gptq/config.json`, `recipe.yaml`; loader `_gptq.py:105-145` |
| `weight_packed` int32 [N, K/8]: 8 nibbles, low first, value `q + 8`, so `q` in [-8, 7]. Zero points: signed int4, packed 8 per int32 along N. | `_gptq.py:54-72, 132-133` |
| Dequant: `w = (q - zp) * scale`, block 32 on K, i.e. `DequantizeLinear(axis=0, block_size=32)` on the [K, N] MatMul operand. | `_quantization.py:183-206, 458-467`; `export.py:1080-1083` |
| Scales are stored fp32 and written to the graph as **bf16 by truncation** (top 16 bits). Only ~7% of them are already bf16-exact, so this matters. | `_quantization.py:220-224, 434-439` |
| The NPU kernel never builds dequantized weights. Per 32-block it takes bf16 partials `P = Σ x·q` and `X = Σ x`, then `y = Σ_g P·s + X·bf16(-zp·s)`; `-zp·s` is rounded to nearest even. bf16 end to end. | `MixedQuantMatMulPattern.cpp` (TorqHLToTorqHW) `8-10, 55, 158`; (LinalgToTorqHL) `95-99, 627-628` |
| LM head: Qwen3 ties it to the embeddings. The export makes it a constant, keeps the `token_id_lut.npy` rows, and loads GPTQ's own (untied) `lm_head` for those rows. | `export.py:503-517`; `_graph.py:326-371`; `_gptq.py:44, 135-136, 169` |
| Trimmed vocab: Latin, punctuation and digits, with byte fallback, gives 99,477 ids (sorted, unique). The table is padded to 99,520 (a multiple of 64) with repeats of the last id (151668). The runner takes the argmax over 99,520 and maps it through the LUT. | `export.py:486-509`; `torq-examples utils/llm.py:688-692` |
| Host embeddings: bf16 [151936, 1024], stored as a 2-byte void dtype, and full vocab. The table equals Qwen's bf16 `embed_tokens` bit for bit (checked). | `export.py:1195-1207`; `utils/llm.py:476` |
| Activations bf16 everywhere: the whole graph goes fp32→bf16 after quantization. | `export.py:1108-1116`, README of the HF repo |
| Stock chat demo (`--instruct-model --no-thinking`): system prompt "You are a helpful assistant. Answer in 1-2 sentences. No lists, no bullet points, no repetition." prefilled as warm-up, then user turn, then `<|im_start|>assistant\n<think>\n\n</think>\n\n`. This equals `apply_chat_template(..., enable_thinking=False)` (checked on every test prompt). Greedy at temperature 0. Stops on `<|im_end|>` / `<|endoftext|>`. | `torq-examples qwen3/src/runner.py:12-15, 97-129` |

`int4_ref.py` reimplements the loader, and every one of the 197 dequantized matrices is bit-identical to
Synaptics' `GPTQCheckpoint.load` plus `dequantize_weight`. The `int4` variant then runs the NPU kernel's
math (`Int4Linear`, mode `npu`). It runs everything else (RMSNorm, RoPE, attention, SiLU) as Hugging
Face's bf16 Qwen3 with eager attention. How the NPU rounds inside those ops is not modelled.
`int4-onnx` (bf16 dequantized weights, the graph's literal meaning) is a second guess at the
numerics. Both variants are reported, so T7 can see how much the unknowns matter.

## Results

**Faithfulness.** Synaptics' `compare_static_quant.py` can't run here: it needs their fp32 static
ONNX export, and torq-tools imports `torq.lab` from the x86-only `torq-compiler` wheel. `int4_ref.py synaptics`
redoes the same experiment on this replica: fp32 activations, fake-quant weights, the same 6 prompts,
288 positions, and the padded vocabulary. The numbers match their README
([eval/synaptics_compare.json](../eval/synaptics_compare.json)):

| | KL | cosine | top-1 |
|---|---|---|---|
| GPTQ, this replica / Synaptics | 0.1069 / 0.107 | 0.9678 / 0.968 | 85.42% / 85.4% |
| RTN, this replica / Synaptics | 0.1516 / 0.152 | 0.9479 / 0.948 | 83.33% / 83.3% |
| GPTQ, board replica (bf16, NPU kernel) | 0.1071 | 0.9671 | 84.72% |

**Gap to full precision** ([eval/int4_vs_fp32.json](../eval/int4_vs_fp32.json)): 8 test prompts, 534
teacher-forced positions. The INT4 gap is about 100× the bf16 gap and grows with depth until layer 12,
then stays flat.

| | L0–3 | L4–7 | L8–11 | L12–15 | L16–19 | L20–23 | L24–27 |
|---|---|---|---|---|---|---|---|
| `down_proj`-input cosine, INT4 vs fp32 | 0.991 | 0.977 | 0.959 | 0.949 | 0.957 | 0.944 | 0.945 |
| top-100 neuron overlap (by \|act\|) | 0.90 | 0.84 | 0.81 | 0.79 | 0.81 | 0.79 | 0.78 |

- Logits (99,477 kept tokens): KL 0.39 and top-1 73% over all positions. On the answer positions only,
  KL 0.17 and top-1 80%. bf16 alone: KL 0.003, top-1 98%, cosine ≥ 0.999 on every layer.
- The worst single tokens drop to cosine 0.50 (layer 11).
- `npu` vs `onnx` numerics: cosine ≥ 0.9991 on every layer (worst token 0.973). How the NPU rounds
  barely matters next to T7's 0.99 bar, but INT4 vs fp32 fails that bar badly. So the check tells a
  4-bit board apart from a full-precision one.
- For the v0.1.0 atlas: a fp32 atlas would rank different neurons than the board shows (top-100
  overlap about 0.8). Build the atlas on this replica.

**Greedy ties.** The LM head outputs bf16, so logits near 20 are 0.125 apart. In
`eval/ref_greedy.json`, 7 of 134 greedy steps are exact ties and 16 have a margin of 0.25 or less. The
replica forks from itself at such steps when only the batching changes (prefill in one call vs one
token at a time): 3 of 8 prompts did. `greedy` therefore runs one token at a time, like the board, and
stores each step's margin. `compare` counts a fork at a margin ≤ 0.25 as a near-tie, not a mismatch.
"Matches for 32 tokens" can only mean "matches or forks at a near-tie".

## For T7

- Teacher-forced sequences (the gap above uses them): `eval/ref_greedy.json`, made from the prompt tokens and the reference's
  32 greedy tokens. Feed `prompt_ids + gen_ids` and compare activations at all positions.
  `int4_ref.py dump` writes the reference side.
- Greedy parity: on the board, run
  `python3 ref/board_greedy.py --examples <torq-examples> -m <model>/transformer.vmfb --out board_greedy.json`,
  then `int4_ref.py compare eval/ref_greedy.json board_greedy.json`. `--tf eval/ref_greedy.json` gives
  the board's teacher-forced top-1 instead, which does not depend on where two greedy runs fork.
  The board script was only tested against a mock runner (the replica), not on a board.
- Two checks to do once you have the compiled graph:
  - The fast INT4 kernel applies only when tiling splits N evenly. Otherwise the compiler falls back to
    a generic dequant (`MixedQuantMatmulPattern.cpp` header comment). Check the compile log to see
    which path each MatMul took.
  - Position 0 (the attention sink) has huge activations. Report cosine with and without it.
