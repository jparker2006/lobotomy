# R8 results: a safe range for every slider (2026-10-28)

Ticket #14, spec #11 I2. GPU work ran on one RunPod RTX 3090 (community, $0.22/h, deleted when done; spend
in `ranges.json`). Judging used blind Haiku sub-agents on Jake's plan, $0. Code: `atlas/cliff.py`, `atlas/pod.py`.

## Verdict: passes
- **κ = 1.41 nats.** For 202 neuron-signs, the teacher-forced KL cliff lands within one ladder step of the
  judged cliff **99.5%** of the time (**98.5%** leave-one-out). The bar was 80%.
- **The bar is weak on a doubling ladder.** Judged cliffs cluster at doses 1–2, so the best constant guess
  ("dose 1 for everyone") also reaches 96.5% within one step. The KL cliff carries real per-neuron signal
  on top of that:
  - exact-step agreement is 51% vs 46% for the constant guess;
  - Spearman 0.72 between the interpolated KL cliff and the judged cliff.
- **Safety, the part the sliders depend on:**
  - The KL cliff falls later than the judged cliff only 2% of the time, and earlier 47% of the time.
  - The slider maximum (80% of the KL cliff) never reaches a dose the judges called broken: 0 of 202.
- **Judges are consistent and grounded:**
  - 97.9% agreement on 606 blind repeats.
  - Unsteered text is judged broken 0% of the time.
  - Judged-broken samples have unsteered log-prob −4.44 per token vs −1.99 for coherent ones, and
    distinct-2 of 0.41 vs 0.96. Both metrics agree with the judges.
- **Breakage by dose** (all calibration samples): 3% at 0.25, 16% at 0.5, 58% at 1, 97% at 2, 100% at 4.
  At dose 1 it is lowest in the middle layers: 42% broken for L7–13, against 57–71% elsewhere.

## The ranges
- **Every BH survivor has a range** for each sign: 21,879 neurons × 2 signs. All 43,758 cross κ inside the
  ladder.
- **Safe |dose|:** median 0.48, p10 0.26, p90 0.57. It's similar across depth (median 0.49 for L0–6, 0.50
  for L7–13, 0.47 for L14–20, 0.46 for L21–27), because KL climbs fast between doses 0.5 and 1 almost
  everywhere.
- **Controls:** L8n908 (ocean) gets +0.59 / −0.56, in line with the CPU demo (fine at +0.5, odd at +1).
  L19n834 is not a BH survivor, so it gets no slider.
- **Cost:** the GPU ran 1.63 h, about **$0.36** of the $3 cap, and the pod was deleted.

## Method
- **Calibration set:** 100 BH survivors, 4 per layer for L0–15 and 3 per layer for L16–27, seeded random,
  plus the L19n834 control. Both signs, 202 neuron-signs.
- **Judged cliff:**
  - Sampled text: temperature 0.7, 48 tokens, at ±{0.25, 0.5, 1, 2, 4}, on 2 chat prompts × 3 seeds
    (6,066 samples).
  - The same Gumbel noise is used for every row of a prompt/seed, so rows differ only by the push.
  - Haiku judged each sample coherent or broken, blind, shuffled and 100 per job.
  - The judged cliff is the first dose where at least half of a neuron-sign's 6 samples are broken.
- **KL cliff:**
  - Teacher-forced KL(steered ‖ base) on the 16 R3 set-1 contexts × 64 tokens (positions ≥ 1), on the
    trimmed vocabulary, at the same doses.
  - The KL cliff is the first dose whose KL exceeds κ. κ maximizes the within-one-step agreement; ties go
    to the smaller, safer κ.
- **Ranges:** for every BH survivor (21,879 neurons × 2 signs), the safe |dose| is 80% of the dose where KL
  crosses κ. This uses linear interpolation along the ladder with KL(0) = 0, and is capped at 80% of 4 when
  KL never crosses.
- **Dose unit:** dose × the layer's median residual norm on those contexts (stored with the ranges). A push
  is `h[n] += dose × med(L) / ‖W_down[:, n]‖`.

## Files
- `ranges.f16`: raw float16 of shape [28, 3072, 2] = (layer, neuron, (+ pole, − pole)). Each entry is the
  largest safe |dose|; NaN marks a non-survivor.
- `ranges.json`: format, κ, the per-layer median residual norm and the calibration summary.
- Calibration evidence: `calib.json`, `ladder_calib.npz`, `samples.json.gz` (all samples with metrics),
  `judge_items.json` and `judgments.json` (blind ids → judgments), and `calib_results.json`.
