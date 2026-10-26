# 05-ranking

Code reader report on lyramakesmusic/neuron-explorer @ f79b219. Analysis only; no code copied.

1. What this slice does

Scores each steered neuron or random-direction control by the words its rollouts share beyond the pool average and the unsteered baselines. `rank.py` outputs "mass" (summed top-5 z) and `null_pct` against controls, and feeds the explorer. `shift.py` is a control-only, all-words alternative that the explorer does not use. The matched random-direction null is the sound part. The z-score is ad hoc.

2. Key mechanics

- Words: `[^\W\d_]{3,}` on lowercased text, counted once per rollout (rank.py:22,47; shift.py:16,26). Rollouts with `rel_push > 1.6` are dropped (rank.py:46; shift.py:23).
- rank: baseline words removed (rank.py:36). Reference is the mean over all units, neurons and controls: `mean = total / len(counts)` (rank.py:52).
- `z = (cnt - mean) / sqrt(mean + 0.5)`, only for `cnt >= 2` (rank.py:57). `mass = Σ max(z,0)` over top 5 (rank.py:72). No z cutoff exists.
- `null_pct` = share of controls with strictly lower mass (rank.py:77-79).
- shift: reference is controls only, leave-one-out for controls: `mu = (tot - c)/(nc - 1)`, else `tot/nc` (shift.py:36-39). `z = (c - mu)/sqrt(mu + 0.5)`, skipped if `mu == 0 and c < 2` (shift.py:40-42). `shift = Σ|z|` of top-5 rises plus top-5 falls (shift.py:44-46).
- Under `main_rel`, `rel_push = |dose|` exactly (steer_sweep.py:68,71). The 1.6 filter therefore drops only free@+3. Each unit is scored on 9 rollouts from 3 prompts.

3. Inputs -> outputs

- In: rollout jsonl (`kind, neuron, prompt, dose, rel_push, text, token_ids, finish, logprob, layer`).
- Out: `*.units.jsonl`, one record per unit: `{kind, neuron, z, mass, words:[[w,cnt,z]], per_prompt, dose, null_pct}`, sorted by mass (rank.py:69-80). `shift_scores` returns `{(kind,neuron): {shift, up, down}}` in memory only.
- Consumers: explorer/server.py:98, depth_report.py:41, app.js:487.

4. Porting notes

- The scoring is host-side Python/numpy and transfers to Gemma-270M unchanged.
- The sweep is the expensive part: 36,864 neurons × 9 rollouts is about 330k generations, so it stays on a host GPU. The NPU can serve only the chosen steered model. Steering is `d × W_down[:,n]` added at mlp_out (steer_sweep.py:3-4). Passing that vector as a graph input would let one compiled model serve every unit. That is outside this slice.
- Switch to the controls-only reference that shift uses. Otherwise mass depends on pool size (research_log.md:189).
- With 256 controls, P(above all) ≈ 1/257 per unit. Across 36,864 neurons that is about 143 chance hits. A 270M port needs more controls or a corrected cutoff.

5. Surprises, bugs, open questions

- The dose table reads too healthy. Empty rollouts get logprob 0.0 (rank.py:67), and `distinct4` returns 1.0 for fewer than 4 tokens (rank.py:31). Empties were 92% at L0 +0.8 (research_log.md:192).
- `n_roll` is computed and never used (shift.py:21,27), so counts are not normalized by retained rollouts.
- `per_prompt` ignores the `max_rel` filter (rank.py:59), so it can name `free` for a word that was not counted.
- shift's falls are length-driven. A common word with control mean 8 and count 0 gives z = -2.75 (shift.py:42). The log says falls are mostly topic displacement (research_log.md:267), yet they are still summed.
- False-positive paths:
  - (a) One prompt at 3 doses gives cnt=3, z=4.24. No rule requires two or more prompts.
  - (b) Homogeneous loops across rollouts give cnt=9, z=12.7.
  - (c) Poisson variance ignores overdispersion from shared prompts, which inflates z.
  - (d) Controls are not exchangeable with neurons. At L22, distinct-4 is 0.71 for neurons vs 0.84 for random (research_log.md:220-228), so `null_pct` partly measures degeneration.
- Verdict: the 1,803 "above every control" hits at L22 (research_log.md:135) far exceed the ~36 chance hits, so the signal is probably real. Its semantic share is unverified.
- Not run: no numpy in this sandbox, so the numbers are hand-checked arithmetic, not a simulation.
