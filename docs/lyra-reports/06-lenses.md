# 06-lenses

Code reader report on lyramakesmusic/neuron-explorer @ f79b219. Analysis only; no code copied.

## 1. What this slice does
Two per-neuron token readouts of `W_down[:, n]`. The **logit lens** (`Runtime.lens`, `build_lens_index`) reads the column through the final norm and LM head, direct path only. The **steer lens** (`sweep/steer_lens.py`) pushes the neuron ±0.8 residual-norm units, reruns the full model on 4 contexts, and keeps the 12 tokens whose log-prob rises most plus a KL.

## 2. Key mechanics
- Push: `unit = norms[l] / W.float().norm(dim=0)` (steer_lens.py:80), `vals = sign * a.dose * unit[cols]`, `dose=0.8` (:29, :89), added at every position (:58). The residual moves by 0.8× its median norm because Qwen adds the MLP output unnormalised.
- Median is over real tokens leaving layer l, position 0 included (:73). Contexts: 2 chat, 2 plain (:21–22).
- Ranking: `delta = (lp - base).mean(1)`, `topk(12)` (:94, :96). It is a log-ratio, not a probability change.
- KL: `(lp.exp() * (lp - base)).sum(-1).mean(1)` (:95), i.e. KL(steered‖base) averaged over contexts.
- Logit lens: `lm_head(lm.norm(sign * d))` (runtime.py:346). Build uses `topk(logits)` for + and `topk(-logits)` for − (:452–454). RMSNorm is scale-invariant, so only the direction of `d` matters.
- `lens()` uses the cache only when `k <= 12` (runtime.py:331), adding `steer_*`, `kl_up`, `kl_down` (:336–337).
- Cost: 4 ctx × 2 signs × 9,216 neurons = 73,728 full forwards per layer. Layers below l are recomputed too. Research log: 111 s/layer on a 4090 (research_log.md:145). Logit lens: one 9216×2560 @ 2560×248k matmul per layer ≈ 11.7 TFLOP, seconds on a GPU. Steer ≈ 10 PFLOP/layer (assuming ~20 tokens per context), roughly 10³× the logit lens.

## 3. Inputs -> outputs
- `sweep/steer_lens.py` → `stats/steer_lens_L{l}.npz`: `up_tok`, `down_tok` [N,12], `up_kl`, `down_kl` [N], `dose`, `contexts`.
- Shipped cache: `steer_index.npz` (`up`, `down` int32 [32,9216,12]; `up_kl`, `down_kl` float32 [32,9216]; `dose`=0.8), `lens_index.npz` (`up`, `down`, same shape), `vocab.json` (248,077 strings). I checked these with a stdlib parser, since numpy isn't installed here.
- `/api/lens` (server.py:233–234) calls `lens(layer, n, 12)`. No script in the repo converts the per-layer sweep files into `steer_index.npz`.

## 4. Porting notes for Gemma-270M / Coralboard NPU
- **Transfers:** the logit lens and the table format. For 18×2048 with top-12, each int32 table is ≈1.8 MB. Build tables offline; the device only needs lookups.
- **Can't run on-device:** the steer sweep. Gemma needs 2048×2×4 = 16,384 forwards per layer, about 1.2 PFLOP over 18 layers at ~20 tokens. That is minutes on a GPU and hopeless on a 1-TOPS NPU. Run it on a host and ship the tables.
- **Dose unit probably won't transfer:** I believe Gemma 3 normalises the MLP output (`post_feedforward_layernorm`), so the residual no longer moves by c·‖W_col‖. Measure the real displacement and recalibrate.
- **Live steering on device:** feed the push as an input tensor [18, 2048] added to MLP activations, so one compiled graph serves every neuron.
- **Memory:** with V=262k, a 96×4 batch of fp32 logits is ≈400 MB per tensor, and the sweep holds several. Lower `--neurons-per-batch` on a 2 GB host.

## 5. Surprises, bugs, open questions
- Logits are bf16 before `.float()` (steer_lens.py:68; runtime.py:346, :452). Spacing is 0.125 for logits in [16,32), so Δlogprob and top-k ties are quantised. Probably worst at late layers (median KL ≈ 0.3). Unmeasured.
- Position 0 is in the sweep median (steer_lens.py:73) but excluded in `neuron_stats.py:53,57`. The runtime push (`_push_entry`, runtime.py:244) uses a different median, so "0.8" is not the same push across sweep and explorer. Likely small; unmeasured.
- Latent: the default `k=16` exceeds the cached 12, so `lens()` falls to the live path. That path has no steer keys and raises under `EXPLORER_NO_MODEL` (runtime.py:57–58). Current callers pass 12.
- Shipped data matches the research log: median KL 10.26 at L0 and 1.38 at L31; L22 N5344 − gives responsible / indicator / 责任人.
- 953 `vocab.json` entries are lone U+FFFD byte fragments.
