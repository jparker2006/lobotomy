# 04-steer-sweep

Code reader report on lyramakesmusic/neuron-explorer @ f79b219. Analysis only; no code copied.

## 1. What this slice does
`sweep/steer_sweep.py` steers every MLP neuron of one layer of Qwen3.5-4B, plus random-direction controls, with one greedy rollout per (prompt, dose) from a fixed design. It writes each rollout to `runs/<run>.jsonl` and optionally captures `mlp_in` through the steering vLLM fork. Reruns resume.

## 2. Key mechanics
- **Equation** (unit u at layer L, applied at every prompt and generated position t):
  **mlp_out_L(t) ← mlp_out_L(t) + α_u · v_u**, with v_u = W_down[:,n] and α_u = d·abs_max[n] (absmax) or d·resid_p50/‖W_down[:,n]‖ (rel). This equals a_n ← a_n + d·abs_max[n] before down_proj, the HF reference in `sweep/check_equivalence.py:76-91`. Injection: `steer_sweep.py:193-194`; α at `:186-189`.
- **Controls** (`:143-149`): v_k is Gaussian, rescaled to ‖W_down[:,r]‖, where r = `ctrl_ref[k]` is drawn uniformly over all neurons. α uses r. Controls are spread evenly among neurons (`:151-153`).
- **rel_push** = |α|·‖v‖/resid_p50 (`:190`). In rel mode this is exactly |d|, so the column is redundant there. resid_p50 is the median per-token layer-output norm, positions ≥1 (`neuron_stats.py:53-57,117`).
- **Designs** (`:48-55`): main = {who, snake, orange}×{-2, 2, 10} + free×10. calib = {who, orange, free}×±{1…32} (36 per unit). main_rel = {who, snake, orange}×{-0.7, +0.5, +0.8} + free×+3. Four unsteered baselines (`:161`).
- **Prompts** (`:41-46`), user-turn chat template, `enable_thinking=False` (`:126-131`).
- **Sampling** (`:192`): temperature 0 (greedy), max_tokens 256, logprobs=5.
- **Row** (`:79-84`): layer, kind, neuron, ref_neuron, prompt, dose, add, rel_push, text, finish, n_gen, token_ids, logprob, top_ids, top_lps, capture_id.

## 3. Inputs -> outputs
- **In:** stats npz (`layer`, `abs_max[I]`, `w_down[I,H]` float16, `resid_norm_q10_50_90`), HF tokenizer and config, and fork-only kwargs (`enable_steering`, `steering_vectors`, `capture`, `capture_consumers`).
- **Out:** `runs/<run>.meta.json`, rewritten on every launch (`:135-141`); `runs/<run>.jsonl`, one line per rollout, flushed per 4000-request chunk; `captures/<run>/<capture_id>` holding mlp_in at layers L+0, +1, +4, +8 (clipped), last prompt token only (`:123, 195-200`).

## 4. Porting notes
- **Transfers:** the design, the equation, greedy decoding with top-k logging, and rel units. Rel needs one scalar per layer. Precompute c_n = α_n·W_down[:,n] offline; the device then only adds an H-vector to the MLP output.
- **Must change:** the only guard (`:120-121`) checks `layer_scale`. Gemma 3 has `post_feedforward_layernorm`, which normalizes the MLP output, so an add before it is not a linear push. Hook after that norm and re-derive. Gemma's MLP is GeGLU, not silu(gate)·up. Steering must be a runtime input tensor so one compiled graph serves all doses.
- **Can't run on-device:** the sweep itself. For Gemma-270M's 2048 neurons, one layer is 2048×10×256 ≈ 5.2M generated tokens, about 94M across 18 layers, before controls. Sweep on a workstation and ship only chosen configs to the Coralboard. Use float W_down for v_u, not a quantized copy.

## 5. Surprises, bugs, or open questions
- **Resume key bug:** the key `(kind, neuron, prompt, dose)` (`:155`) omits dose_unit and design. Running `--design calib --dose-unit rel` on an absmax run's file, with the same neurons and controls, skips every rollout. meta.json is then overwritten to say `rel` (`:135`, before the early return at `:166`). Rows carry no dose_unit or design.
- **Silent unit footgun:** `--design main_rel` without `--dose-unit rel` runs -0.7×abs_max with no warning (`:92-94`).
- **Position 0:** steering covers position 0, the chat's first token (an attention sink). The stats exclude it (`neuron_stats.py:53, 60`). Unverified whether the fork skips it.
- **Capture +0 is a free check:** mlp_in at L is computed before mlp_out at L, so it should be identical across doses for a given prompt.
- **Config count:** a chunk of 4000 requests holds about 400 units with 3–4 distinct configs each, so up to ~1,200–1,600 configs per `generate` call, against `max_configs=512` (`:107`). Fork behavior is not in this repo.
- **Asymmetric and weak null:** main_rel has one negative dose, and free gets only +3. Gaussian controls are not matched to residual-stream anisotropy, so they are a weak null.
