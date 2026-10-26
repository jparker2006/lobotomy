# 02-research-log

Code reader report on lyramakesmusic/neuron-explorer @ f79b219. Analysis only; no code copied.

### 1. What this slice does
`research_log.md` is Lyra's dated lab notebook for the steering-first explorer on Qwen3.5-4B (L8, L22, then all 32 layers). It records design, controls, numbers, failures, and pipeline lessons. `scripts/owl_delta_vs_base.py` audits the tawny-owl Qwen3-8B finetune's 128 edited neurons against base weights.

### 2. Key mechanics
- **Design** (`sweep/steer_sweep.py:54`): 10 rollouts per neuron. `main_rel` doses are in residual-norm units: {who, snake, orange} × {−0.7, +0.5, +0.8}, free × +3. The abs_max design was dropped because its +10 was 1.8× the residual (log:51).
- **Worked:** N2742 produced "Extrali… planet Xeno-888" additively at +2 (log:55–57). Full L22: 1,803/9,216 (19.6%) neurons above every control. L8: 757 (8.2%) (log:135).
- **Controls matter:** random direction 1 → "Xiaomi AI" (log:68). Without controls, that reads as a persona neuron.
- **Failed:** "did the output change?" is useless. ±1 changes greedy output 83–98% (log:64–65). Relative doses break early layers: 92% of L0 rollouts are empty at +0.8 (log:192).
- **Depth:** share above control p95 is 29–40% at L4–7, 28–41% at L20–26, and dips to 8–17% at L13–19 and 1–6% at L29–31 (log:194–200). The dip survives an all-words metric (log:261–265).
- **Lessons:** calibrate doses per layer before sweeping (log:244). Pull runs, stats and captures together (log:280). Check engine parity token-for-token (log:39–41).

### 3. Inputs -> outputs
- Log: prose only. It cites `runs/depth_report.json`, `captures/<run>.mlp_in.npy` (6.2 GB), and explorer caches.
- Script in: safetensors with `layers.{l}.neuron_index`, `gate_proj_rows_add`, `up_proj_rows_add`, `down_proj_columns_add` (`script:29–30`). Not shipped (`.gitignore:14`). Base weights come from Qwen/Qwen3-8B by HTTP range (`script:7–23`).
- Script out: JSON per neuron, per gate/up/down: `rel=|Δ|/|w|`, `cos`, `along=w·Δ/|w|²` (`script:37–45`).

### 4. Porting notes
- **Transfers:** stats, rel doses, controls, `sweep/shift.py`, steer lens, and the owl audit (pure numpy).
- **Hook trap (verify):** from memory, Gemma 3 has a post-FFN RMSNorm. That rescales the whole MLP output, so a push at `mlp_out` is no longer additive to the residual. The `layer_scale` assert (`steer_sweep.py:120`) won't catch this. Hook the `down_proj` input (width 2048) instead. That is an exact neuron edit, and the log says ×k and ablation need it (log:30–31).
- **Redo for 18×2048:** calibrate all 18 layers, L0–2 included. A full sweep is about 370k rollouts (Qwen's 32 layers ≈ 2.9M), so no subsampling. Use 256 controls per layer. The linear-attention concurrency cap (log:47) won't apply. Precompute lens tables offline over the 262k vocab.
- **Coralboard:** make steering a static `[18, 2048]` graph input, zeros by default, added at the `down_proj` input. That avoids per-request tables (the fork's row table, log:24, won't port). int8 shifts residual norms, so recalibrate on the quantized graph. Sweeps, stats, lens, UMAP and the agent stay on the host.

### 5. Surprises, bugs, or open questions
- **Anisotropy confound:** controls are Gaussian vectors rescaled to the reference norm (`steer_sweep.py:143–149`). "Real directions break text sooner" (log:63–64) may just reflect neuron columns loading high-gain residual dims. Needs a coordinate-matched control.
- **Stats mix:** log:53 says every sweep reuses the 2M-token stats. `sweep/run_depth.sh:14` uses 200k tokens for depth layers.
- **Overclaim:** "coherent window … for every neuron" (log:73) rests on 21 neurons plus 8 random directions near one L22 neuron.
- **Noise:** depth shares use 384 neurons against a 64-control p95 (`sweep/depth_report.py:49`). SE on a 14% share is about 1.8 pp.
- **Owl audit:** no control. NOTES.md:45–46 says 128 random neurons trained an equally good owl, so deltas need a random-128 baseline. The docstring says "only the needed bytes," but line 36 fetches each full `down_proj` (about 100 MB per layer for Qwen3-8B). A bf16 delta file would fail at `load_file` (line 25). Untested, because the input isn't shipped.
