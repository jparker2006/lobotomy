# 01-architecture

Code reader report on lyramakesmusic/neuron-explorer @ f79b219. Analysis only; no code copied.

**1. What this slice does**
Maps the artifact graph at f79b219: which script writes each file in `runs/`, `stats/` and `explorer/cache/`, and which server code reads it. Covers README, NOTES, pyproject, uv.lock and .gitignore. I inspected the shipped caches with a stdlib zip/.npy parser because numpy isn't installed here. Nothing was run.

**2. Key mechanics**
- Pipeline: `neuron_stats.py` → `stats/q35-4b_L{l}.npz` → `steer_sweep.py --design main_rel --dose-unit rel` → `runs/<run>.jsonl` + `.meta.json` → `server.py:84-101` (`load_layer`) writes `.units.jsonl` lazily.
- Push: adds `d·abs_max[n]·W_down[:,n]` at `mlp_out` (`steer_sweep.py:3-7`). `main_rel` doses are fractions of median residual norm.
- Startup builds missing caches: `lens_index` (`server.py:1067`), `vocab.json` (`runtime.py:468`), `act_scale` (`server.py:1070`).
- Unmerged connection slots hold `z = -60000` (layer 0 `up`, layer 31 `down`; `build_connections.py:85`). Runtime filters `z > -1e4` (`runtime.py:369`). Verified.
- Deps (uv.lock): torch 2.14.1, transformers 5.20.1, sentence-transformers 6.1.0, umap-learn 0.5.12, fastapi 0.146.0. No vLLM.

**3. Inputs -> outputs** (writer → artifact → reader)
- `neuron_stats.py:113` → `stats/q35-4b_L*.npz` (`abs_max`, `w_down`, `resid_norm_q10_50_90`) → `steer_sweep.py:113`, `runtime.py:135`
- `steer_sweep.py:134-213` → `runs/<run>.jsonl` + `.meta.json` → `server.py:66-115`, `capture_prefill.py`
- `rank.py:95`, `depth_report.py:43`, `server.py:96` → `runs/<run>.units.jsonl` → `server.py:102`
- `build_embeddings.py:38` → `runs/<run>.umap.json` → `server.py:203`
- `build_connections.py:89` → `cache/connections.npz` → `runtime.py:305`
- `runtime.py:455` → `cache/lens_index.npz` → `server.py:286`
- `runtime.py:503` → `cache/act_scale.npz` → `server.py:591`
- `build_layout.py:59` → `cache/layout_writes.npz` → `server.py:474`
- `runtime.py:434` → `cache/connections.sqlite` (live-computed, k>24 only)
- Shapes (Qwen, L=32, N=9216): connections `[L,N,24]`; lens/steer `up,down` int32 `[L,N,12]`; `act_scale` f32 `[L,N]`, tokens=10240; layout int16 `[L,N]`, each layer a permutation of 0..9215 (verified); vocab 248,077 strings.
- Endpoints: `/api/units`, `/api/rollout`, `/api/embeddings` read runs; `/api/connections`, `/api/lens`, `/api/layout` read caches; `/api/steer`, `/api/infer*`, `/api/agent/*` use the live model. `static/app.js` calls them.

**4. Porting notes**
- Transfers: the hook at down_proj input (`runtime.py` header), the sign-regime stats, and the lens and connection formulas. GELU-tanh has the same gate sign structure as SiLU.
- Breaks on Gemma 3: `modeling_gemma3.py:415` applies `post_feedforward_layernorm` to the MLP output. A push into `mlp_out` is then RMS-normalized, so it is no longer additive. The `steer_sweep.py:120-121` guard checks only `layer_scale`, so it passes anyway. `build_connections.py:45` uses `post_attention_layernorm` as the pre-MLP norm; Gemma's is `pre_feedforward_layernorm`.
- `build_layout.py:30-31` asserts N is a perfect square. 2048 fails, so use a 32×64 grid.
- Qwen is hard-coded at `server.py:29` and `server.py:147`. `EXPLORER_MODEL` only works in `runtime.py:17`.
- Coralboard can't run any of this (torch, transformers, UMAP, FastAPI, Claude CLI). Precompute on the host. At 18×2048 = 36,864 neurons, the neuron-indexed caches are 1/8 of Qwen's, about 15 MB. Steering on the NPU must be a graph input (per-layer add/mul vector), not a baked constant, since Torq compiles ahead of time. That is an inference, not tested.
- Qwen3-0.6B has no post-FFN norm (`modeling_qwen3.py:283-284`), so the formulas should carry over.

**5. Surprises**
- `steer_index.npz` is read at `runtime.py:325`, but no script writes it. Its keys match a repack of `steer_lens.py` output. `research_log.md` doesn't mention it.
- `runs/depth_report.json` and `.emb.npy` are written and never read. `stats/steer_lens_L*.npz` has no reader.
- The vLLM steering fork, fla-core and the Python 3.12 venv appear only in `setup_pod.sh:27`, so `uv sync` can't reproduce the sweep. `huggingface_hub` is imported but not declared.
- `.gitignore` misses `*.log` (`run_layer.sh` writes `stats_L*.log` to the repo root) and `captures-check/` (`check_equivalence.py:41`).
- `server.py:29` loads the HF config at import, so `EXPLORER_NO_MODEL` still needs network or cache.
