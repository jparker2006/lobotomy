# 09-runtime-steering

Code reader report on lyramakesmusic/neuron-explorer @ f79b219. Analysis only; no code copied.

**1. What this slice does**
`Runtime` loads Qwen3.5-4B and puts forward hooks on every layer's `down_proj` input (the MLP neuron activations) and on each attention output projection (the heads). `steer_multi` then runs greedy, batched generation with three rows: row 0 pushed up, row 1 baseline, row 2 pushed down. It also turns each "dose" into a per-row vector, using each layer's median residual norm as the unit.

**2. Key mechanics**
- Hook point: `forward_pre_hook` on `layer.mlp.down_proj` (runtime.py:68). It edits `h = act(gate(x)) * up(x)` on a clone (95), shape `[B,T,2048]`.
- Add: `unit = p50[l] / ||W_down[:,n]||`; `vals = [+1,0,-1] * dose * unit` (244-245, 174); applied as `h[:,:,n] += vals` (102). The residual change is `dose·p50` in L2 norm, exact because `down_proj` is linear with no bias.
- Mul: `vals = [dose, 1.0, -dose]` (242); applied as `h[:,:,n] *= vals` (100). The down row is a sign flip, not `1/dose`.
- Whole-layer mul (`n=None`): `h *= vals[:,None,None]`, mul only (98, 239-240).
- Several pushes apply in list order inside one hook (93-102). Adds sum; muls compound.
- Pushes are set before prefill (183) and stay on for every decode step (185, 207), so they also hit prompt tokens.
- Heads: pre-hook on `o_proj` (16 heads of 256) or `out_proj` (32 heads of 128) (74, 77). `v[...,head,:] *= scale`; 0 ablates (112-117).
- p50: stats npz `[1]` if present (135-137), else the median of per-token norms over positions ≥1 of `REFERENCE_TEXT` (139-144).

```
pre-hook on down_proj[l], h: [3,T,I]
add: h[:,:,n] += [+1,0,-1]*dose*p50[l]/||Wd[:,n]||
mul: h[:,:,n] *= [dose,1,-dose]
```

**3. Inputs -> outputs**
- In: optional `stats_dir/q35-4b_L{l}.npz`; `EXPLORER_MODEL`; three left-padded message lists; pushes `(layer, n, dose, mode)`; `max_new` (200 here, 150 from server.py:671).
- Out: a generator of `(row, text_delta)` then `(None, None)`. The server returns `{up, baseline, down}` (server.py:671-680).
- Creates `cache/connections.sqlite` (34-35). Nothing shipped under `explorer/cache` matches `q35-4b_L*.npz`.

**4. Porting notes for Gemma-270M / Coralboard**
- **Breaks the exact math.** Gemma 3 is sandwich-normed. Per transformers `modeling_gemma3.py` (read via WebFetch), each sublayer is followed by an RMSNorm before the residual add: `post_feedforward_layernorm(mlp(...))` and `post_attention_layernorm(attn(...))`. The norm sits between `down_proj` and the residual, so "dose × p50" is no longer exact. The sweep already guards this with `assert not layer_scale` (sweep/steer_sweep.py:120).
- **Hook point still works.** Gemma's MLP is `down_proj(act(gate(x)) * up(x))` with no down bias. Its activation is `gelu_pytorch_tanh`.
- **Dims are unverified.** The HF config returned 401 (gated). From memory: hidden 640, intermediate 2048, 18 layers, 4 heads × 256.
- **Coralboard needs graph inputs, not hooks.** Give each layer `M[l,2048]` (default 1) and `A[l,2048]` (default 0), and apply `h = h*M[l] + A[l]` before `down_proj`. That covers add and mul, and dose changes need no recompile. Head scales become an input vector. Read becomes an extra output. The p50 values are baked in as constants. Shapes must be static: fixed max length and KV cache, and explicit `position_ids` and mask in place of the left-pad logic (178-206). Check that Torq supports GELU-tanh, RMSNorm and the elementwise ops. I have not checked this.

**5. Surprises, bugs, open questions**
- **Read order.** `_read` captures before the push is applied (90-92 vs 93-103). A read at a pushed layer shows the unsteered value. Is that intended?
- **Mul is asymmetric.** The "reverse" is `-dose`, not `1/dose`. The docstring at 167 says so, but it is easy to misread.
- **"Exact" is Qwen-only.** The docstring claim at 1-7 and 238-246 depends on Qwen having no post-MLP norm.
- **p50 can disagree with the sweep.** The fallback uses one short paragraph, while `sweep/neuron_stats.py:57, 111` uses the full corpus. The npz is not shipped, so the unit may differ between live and sweep.
- **Hardcoded Qwen strings.** The filename `q35-4b_L{l}.npz` (135) and the EOS `<|im_end|>` (181) are both Qwen-specific.
