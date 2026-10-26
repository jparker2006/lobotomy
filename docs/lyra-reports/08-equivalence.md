# 08-equivalence

Code reader report on lyramakesmusic/neuron-explorer @ f79b219. Analysis only; no code copied.

### 1. What this slice does
Three scripts check the two assumptions the steering sweep rests on: that vLLM's `mlp_out` push equals HF's "add v to neuron n", and that vLLM's `mlp_in` captures match HF's neuron activations. `capture_prefill.py` rebuilds the captures offline with one prefill per rollout. No file in the slice has a pass/fail threshold. The only numbers are in the research log.

### 2. Key mechanics
- Steering scale matches on both engines: vLLM gets `scale = d*abs_max[n]` (`check_equivalence.py:65`), and HF adds `d*abs_max[n]` to the down_proj input (`:88`, `:95`). Both equal `d*abs_max[n]*W_down[:,n]` at mlp_out.
- Steered agreement is the shared greedy prefix, `common_prefix(hi, vi)`, over 48 tokens (`:99`). It prints HF text when the prefix is under 8 (`:101`). "Looks the same" is judged by eye.
- Capture check: `recomputed = silu(gate_proj(x)) * up_proj(x)` in bf16 (`:114-116`). Then `rel = ‖rec−ref‖/‖ref‖` per position and `same_top` is argmax agreement (`:121-122`).
- Prefill push: a forward hook on every `layer.mlp` returns `out + vec.to(bf16)` at all positions (`capture_prefill.py:39-43`). A pre-hook captures `inp[0][:,-1]` and raises `Stop` after the last capture layer (`:45-51`).
- Vectors: neuron `w_down[n]*add`, random `ctrl_vecs[k]*add`, baseline zero (`:76-81`). `add` is read from each jsonl row (`steer_sweep.py:186-189`).
- Documented bands: 0.1–3% with 100% top-neuron (`research_log.md:42`), and median 1.1–1.5%, max 2% over 294 rollouts (`research_log.md:280`).

### 3. Inputs -> outputs
- In: stats npz with `layer`, `abs_max`, `w_down` (fp16 `[N,H]`, `neuron_stats.py:110`). Run `.jsonl` plus `.meta.json` (`layer`, `capture_layers`, `prompt_ids`).
- Out: `captures/<run>.mlp_in.npy`, float16 `[rows, len(capture_layers), H]`. Row i is jsonl line i, and baseline rows are unsteered. Also `.mlp_in.json` with `{run, capture_layers, hook, position, rows}`. `check_equivalence` writes only `captures-check/`.
- Fork captures are read from `captures/<run>/<capture_id>` via `read_request`.

### 4. Porting notes for Gemma-270M / Coralboard NPU
- Transfers: the greedy-prefix metric, causal prefill with the push at all prompt positions, and float16 storage (exact for bf16 values in fp16's normal range).
- Gemma 3 uses a GeGLU MLP (gelu-tanh gate), so the formula at `:116` and docstring `:8` change.
- Gemma 3 has `post_feedforward_layernorm` after the MLP (from my memory of transformers; verify). The push hook on `layer.mlp` (`capture_prefill.py:42`) lands before that norm, so it is no longer a pure `W_down[:,n]` add. Hook after the norm, and replace the `layer_scale` assert (`steer_sweep.py:120`) with a post-FFN-norm assert.
- Drop `enable_thinking` (`check_equivalence.py:49`), which is Qwen-specific.
- The bf16 band won't transfer. Int8 Torq error will dominate, so re-derive tolerances against a host fp32 reference.
- Nothing here runs on the board. Captures and checks run on host CPU, which is cheap at 270M. The NPU graph needs extra outputs to expose mlp_in, and steering needs a runtime input vector rather than a recompile.

### 5. Surprises, bugs, open questions
- **Silent wrong random controls (highest priority).** `capture_prefill.py:67` infers `n_ctrl` from the largest random index in the jsonl. For a partial run, that is less than the sweep's `--random-controls`. `integers(size=n)` is prefix-consistent, so the assert at `:74` passes. But the `standard_normal` draw at `:70` starts at a different RNG position, so every control vector is wrong. `steer_sweep.py:136-141` does not store K or the seed. I did not execute this: numpy isn't installed here and I couldn't install it.
- Stale skip: `:61-64` skips any run whose `.npy` exists. It never compares rows to the jsonl, though `rows` is written at `:104-105`.
- `check_equivalence.py:106-107` reads captures after `del llm` with no engine shutdown. `steer_sweep.py:219` does shut down. Possible race, unverified.
- The vLLM engine stays resident during the HF load and generation (`:57-60` before `:78`). That is tight on a 24 GB card. Unverified.
- `--capture-gen` (`:40`) is unused, and `shutil.rmtree(a.capture_root)` (`:56`) is unguarded.
- The production `last_prompt` mode is never checked against HF. The HF check uses `all_prompt` (`:69`), and the prefill check compares against fork captures, not HF.
- `check_capture_prefill.py` exits 0 whatever the error.
