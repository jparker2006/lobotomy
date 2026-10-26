# 03-neuron-stats

Code reader report on lyramakesmusic/neuron-explorer @ f79b219. Analysis only; no code copied.

**1. What this slice does**
`sweep/neuron_stats.py` runs one stats pass of bf16 Qwen3.5-4B on CUDA over a packed web + chat corpus. It hooks chosen MLP layers and writes, per layer, sign-regime extremes, `abs_max` (the steering dose unit), `w_down` columns, and residual-norm quantiles. `steer_sweep.py` and `explorer/runtime.py` read its `.npz` output.

**2. Key mechanics**
- Activation `a = silu(gate)*up` is the `down_proj` input. Hooks capture `gate_proj`/`up_proj` outputs and that input (`neuron_stats.py:92-94`). `sign(silu(g)) = sign(g)`, so regimes use raw gate and up signs.
- Regimes (`:65-68`): `pp_max` = max `a` where g>0,u>0; `nn_max` = max `a` where g≤0,u≤0; `pn_min` = min `a` where g>0,u<0; `np_min` = min `a` where g≤0,u>0. All start at 0 (`:50-51`).
- `abs_max = max(pp, nn, -pn, -np)` (`:109`). Position 0 of every window is dropped (`:61`, `[:, 1:]`).
- Corpus (`:29-33`): fineweb-edu sample-10BT and ultrachat_200k train_sft, alternated one-to-one by document. Chat docs go through `apply_chat_template(enable_thinking=False)`. Docs are joined with EOS only, `add_special_tokens=False` (`:41`), and cut into `[16, 1024]` windows with no attention masking (`:36-44`).
- Token count: `n_steps = ceil(tokens/(1024·16))` (`:97`). Default 2M gives 123 steps: 2,015,232 tokens fed and 2,013,264 scored. The npz `tokens` field stores the fed count (`:114`).
- Residual scale: forward hook on decoder layer `l`'s output, `h[:,1:].float().norm(dim=-1)` (`:55-57`). Quantiles are `np.quantile(…, [0.1,0.5,0.9])` (`:111`). Consumers take index 1 (`steer_sweep.py:117`, `runtime.py:137`). The median is the unsteered L2 norm leaving layer `l`, pooled over all corpus tokens.
- Batching: 16 windows per forward, `use_cache=False`, all target layers hooked in one pass.

**3. Inputs -> outputs**
- In: streamed HF datasets, the model via `AutoModelForImageTextToText` (`:84-85`).
- Out: `{out}_L{l}.npz` with `pp_max, nn_max, pn_min, np_min, abs_max` (length I=9216), `w_down` [I,H] float16 (row n = W_down[:,n]), `resid_norm_q10_50_90`, `model`, `layer`, `tokens`.

**4. Porting notes for Gemma-270M / Coralboard**
- Transfers unchanged: the regime logic and `abs_max`. Gemma's MLP is `down(act(gate)·up)` with no bias (`modeling_gemma3.py:125-131`), and its default activation keeps the sign of gate (`configuration_gemma3.py:81`).
- Must change:
  - Loader: Gemma-270M is text-only, so `model.model.language_model` (`:84-85`) and `device_map="cuda"` have to go.
  - Hook point: Gemma3 applies `post_feedforward_layernorm` to the MLP output before the residual add (`modeling_gemma3.py:414-416`). A push at `mlp_out` is then RMS-normalised and not additive. Steer at the down_proj input instead, `a_n += d·abs_max[n]`, which matches the stat's definition.
  - BOS: EOS-only separators (`:41`) mean web windows get no `<bos>`, and Gemma's sink depends on BOS. Prepend it, and dedupe against the chat template (from memory; the gated repo was unreachable here).
- Can't run on device: the streaming corpus, the fp32 stats pass, and the 2M-token quantile. Ship only `abs_max` (2048 floats), `w_down` (2048×H), and one residual-median scalar. Recalibrate `abs_max` and the median on the quantized NPU graph. Make the dose a runtime input, not a compile-time constant.

**5. Surprises, bugs, open questions**
- `run_depth.sh:12-14` writes 200k-token stats to the same `stats/q35-4b_L{l}.npz` names. The guards only check existence (`run_layer.sh:13`), so a 200k file can stand in for the 2M one. `abs_max` differs (6.9 vs 10.25, `research_log.md:49-52`). Rel-mode sweeps use only the median and down norms, so they are less affected.
- `pp_max`/`nn_max`/`pn_min`/`np_min` are saved but nothing in the repo reads them.
- The explorer has a second "typical max", `build_act_scale` (`runtime.py:485-500`): 10k fineweb tokens, positions ≥1, 512-token chunks. It is not `abs_max`.
- If the npz is missing, `runtime.py:139-144` takes the median from one `REFERENCE_TEXT` sequence, not the corpus.
- `dn_top` uses strict `>` (`:116`), so ties count as not (−,−).
- Open: packed windows with no reset make the stats depend on cross-document context. Only position 0 is excluded, so other massive-activation tokens may still inflate `abs_max`. Not checked.
- `w_down` is cast to float16 (`:110`).
