# 07-pod-infra

Code reader report on lyramakesmusic/neuron-explorer @ f79b219. Analysis only; no code copied.

## 1. What this slice does
Sets up a rented CUDA pod for the Qwen3.5-4B steering sweep, runs per-layer and multi-layer sweeps, merges shards, and builds the depth curve. `scripts/mirror.py` is unrelated: it crawls the Goodfire essay into a gitignored `source/`.

## 2. Key mechanics
- **Pod:** RunPod Secure, 1 GPU from 4090/3090/A40/A6000/5090/6000 Ada/L40S, 60 GB `/workspace` (`sweep/pod_template.json:1`). Real runs used 4× 4090 (`research_log.md:133`).
- **Fork:** cloned at unpinned HEAD (`setup_pod.sh:23`). Only the upstream wheel is pinned, to `755492e3` (`:27-28`).
- **Design `main_rel`:** 10 rollouts per unit (`steer_sweep.py:54`). Push is `add = d·resid_p50/‖W_down[:,n]‖` (`:187`), so `rel_push = |d|` (`:190`).
- **Sharding:** manual `--neurons` ranges (`steer_sweep.py:58-67`). Every shard re-runs all 256 controls (`:151-153`), about 8% extra work at 4 shards.
- **Merge:** dedupes on (kind, neuron, prompt, dose), first copy wins (`merge_runs.py:25`).
- **Depth:** uses `-r384` runs, or subsamples `-full` with `random.Random(l)` (`depth_report.py:36-40`). Scores come from `rank.compute_units`. `above_p95` compares neurons to control p95 (`:49,60`).
- **Throughput:** 3.8k gen tok/s unsteered on a 4090 (`research_log.md:77`). The real L22 sweep ran at 1.5k, CPU-bound (`:119`).

## 3. Inputs -> outputs
- **In:** `stats/q35-4b_L{l}.npz` (abs_max, w_down, resid norms); `runs/<run>.jsonl` and `.meta.json` from `steer_sweep.py`.
- **Out:** `runs/<run>.units.jsonl`; `runs/depth_report.json` (per layer: null and neuron percentiles, above_p95, coherence at +0.8/−0.7, top-5 units); `captures/<run>/`.

## 4. Porting notes
- **Qwen full sweep:** about 95k rollouts × ~200 tok per layer. 32 layers is about 607M tokens: ~44 GPU-h at 3.8k tok/s, ~112 at 1.5k (~$40–100 at ~$0.89/h, `research_log.md:80`).
- **Gemma-270M:** (2,048+256)×10 ≈ 23k rollouts per layer, 18 layers ≈ 83M tokens. Unmeasured: 2–15 GPU-h at 1.5k–10k tok/s (~$2–14).
- **Silent hazard:** the only architecture guard is `layer_scale` (`steer_sweep.py:120`). Gemma's post-FFN norm (verify in transformers) makes an `mlp_out` push non-linear. Hook the down_proj input instead (`research_log.md:27-32`).
- **Hard-coded Qwen:** `depth_report.py:19` defaults to `0-31`, `q35-4b` filenames, and `bench.py:68` uses `<|im_end|>` (Gemma uses `<end_of_turn>`).
- **Coralboard:** the fork, captures, and logprobs can't run there. Transferable: the dose design, norm-matched controls, the jsonl schema, and host-side merge/peek/depth_report. The board needs only a per-config steering-vector input. Sweeps stay on pods; at an assumed 10–50 tok/s decode, one on-device layer would take 1–5 days.
- **Disk:** Qwen captures are ~1.9 GB per layer (`research_log.md:280`), so 32 layers is ~62 GB, over the 60 GB volume.

## 5. Surprises, bugs, or open questions
- Merge validates nothing. Shards with different `--stats`, `--seed`, `--dose-unit`, or `--design` merge silently (`merge_runs.py:19-33`).
- meta.json omits seed, neurons, control count, and fork/torch versions (`steer_sweep.py:136-142`).
- Resume would crash on a truncated last line, since every line goes through `json.loads` (`steer_sweep.py:159`).
- Designs without `+0.8`/`-0.7` doses make `depth_report` emit NaN (`depth_report.py:52-54`).
- `uv run` in docstrings (`steer_sweep.py:24`, `bench.py:4`) would re-sync `.venv` to `uv.lock`, which has no vLLM. `setup_pod.sh` doesn't install umap-learn, yet embeddings were built on the pod (`research_log.md:282`).
