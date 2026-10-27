# R6 planner p5 — slice report (layers 25–26, 542 poles)
The per-batch QC notes are in `qc_log.md`.

## Labelling
- v1: 542/542 poles labelled in 28 batches (b0 from the orchestrator's wave 1), 0 failed. v2: 6 batches, 120 poles, 0 failed.
- Final mix: v1 458, v2 80, pilot 4. Verdicts: clear 33, likely 124, mixed 385.
- 59 names over 4 words were shortened without changing their meaning (57 in v1, 2 in v2); `name_orig` keeps the old name.

## Dev (used only for QC and picking)
- v1: real 0.377, word list 0.187 (n=315), lookup 0.473. By verdict: clear 0.58, likely 0.43, mixed 0.34.
  The 0.377 comes after the 7 flat-prediction poles were rescored; it was 0.371 before. No batch is under 0.10.
- Re-runs (v2) and why: b7 (worst, 0.110), b10 (0.263), b22 (low only because of the lazy scorer), b4 (next worst),
  and b11 and b13 (QC flags). Batch means v1 → v2: b7 0.11→0.34, b10 0.26→0.41, b22 0.36→0.45, b4 0.29→0.33,
  b13 0.33→0.33, b11 0.49→0.45. No batch stayed under 0.10, so there was no v3.
- Final dev is 0.406, but each pole keeps its best of v1/v2/pilot, so that number is biased up. Test is the honest score.
- Opus second opinion on 10 ambiguous poles (used for re-run choice only): every label carries signal. Opus r is
  0.14–0.70 where Haiku gave ≤ 0, for example L25n2875 "'Public' and 'public domain'" 0→0.65. So the scorer caused
  those low Haiku scores, not the labels.

## QC findings
- Most labels match their evidence. 71% are `mixed`, mostly subword-fragment and name-piece poles, which is honest.
  Good examples: L26n1425 "Stem before '-ist'/'-ian'" (hydro|log|ist), L26n3007 "'years' in age phrases",
  L25n2834 "'the' before cause nouns".
- Weak v1 labels that v2 fixed: L25n2150 "Word before month name" described the writes, but its top peaks were
  'ger'trude and 'convers'ational (v2 names both). L26n759 led with wh/car, but 7/15 peaks were infinitive 'to'
  (v2: "Word before bare verb"). L26n985 claimed `likely` from a weak World Bank tail.
- Verdict calibration: single-token chat-template poles are under-claimed as `mixed` (L26n317 `<think>` and
  L25n1274 `</think>` fire on it in 15/15 snippets). A few `clear` verdicts are generous (L26n1480 treaty,
  L26n2193 hardware).
- Concepts often copy top-token fragments instead of the trigger the description names. For example, L26n2358
  "Preposition before place names" has the concepts Warsaw, aj, Nan. Some concepts carry meta words ("split words").
- Scorer: one Haiku dev job gave all-1s for 7 poles (r=0). I rescored those on dev, as the orchestrator rule allows.
  On test, the script's lazy-batch rescore handled two all-zero shuffled replies. The final test files have 0, 0
  and 1 flat predictions (real, shuffled, word list), kept as returned.

## Test (final; each condition scored once; n=542)
- real 0.394 | shuffled 0.062 | word list 0.235 | lookup 0.482
- real − shuffled +0.333 [0.301, 0.364]; real − word list +0.159 [0.134, 0.186]; real − lookup −0.088 [−0.112, −0.060] (lookup still wins, as in wave 1).
- By verdict: clear 0.640 (33), likely 0.425 (124), mixed 0.364 (385). By version: v1 0.391, v2 0.413, pilot 0.357.
- 47/542 poles score test r < 0. Re-run batches on test: b7 0.359 (v1 dev was 0.110), b10 0.410, b13 0.522, b22 0.521.
