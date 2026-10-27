# R6 planner p6: QC report (535 poles, layers 0–24)

## Labelling
- 535/535 labelled in v1 (b0 by the orchestrator's first wave). 0 poles failed after retries; 2 needed a 2nd attempt. 63 v1 and 13 v2 names over 4 words shortened in place, same meaning (original kept in `name_orig`).
- Final verdicts: clear 127, likely 223, mixed 185. Final versions: v1 445, v2 88, pilot 2 (best dev r per pole).

## QC findings (read ≥2 labels per batch against `show` evidence)
- No copied or generic labels, and no pole sign misread. Most labels name the slot before a predicted word and what the pole writes ('insertion|sort', 'convert|… binary', 'the| exact number' → 'numbers').
- Good examples: L24n2560 "Words leading into failure" ('susceptible| to', 'without| breaking', writes 'failure'); L13n1255 "Numbered list opener punctuation" ('1|.' after </think>); L21n2321 "'ur' syllable in names" (Ast|ur|ias, |Sur|iname), which says outright that read and write don't match.
- Recurring miss, a second trigger left out: L23n1084 "IT and admin nouns" skips the leading '1' of years (5/15 snippets; still missing in v2); L24n1310 "Middle of split names" skips year digits (half the snippets); L24n1662 skips mid-name pieces.
- Misread: L22n1337 "Court-case and proper names". The real trigger is the first item of a comparison before 'vs' ('Jones| v.', '57%| vs 36%', '2.5 minutes| compared to'); v2 fixed it ("Contrast pairs and vs").
- Overclaimed verdict: L24n2010 `clear`, but only 3/9 top snippets fit (v2 `likely`). Concept lists sometimes copy junk top tokens (L24n2517 ']$', '=('; L22n1061 'DK').
- Batch-wide style issue only in b7: write-side names ("Null-value placeholder writer", "URL-start colon writer"); descriptions still named the triggers.
- L0n38 (layer-0 fixed-token detector) is a label-format failure: "Scattered content words" got dev r −0.50 while the word list got 0.56 and lookup 0.80. v2 "Abstract social-state nouns" reached 0.48.

## Dev scores
- Scorer laziness: 19 v1 dev rows came back flat or near-flat (one 15-pole job was lazy 3 times). I dropped them and rescored on dev only; none flat after. No flat rows in v2.
- v1 dev: real r 0.505, word list 0.370, lookup 0.422; real − word list +0.14 [0.11, 0.17]. By verdict: clear 0.60, likely 0.53, mixed 0.40. Worst batches: b20 0.36, b0 0.42, b7 0.43. No batch under 0.10, so no v3 round.
- Pilot labels (12 poles, older evidence) dev r 0.40 vs lookup 0.49; 2 of them won the pick.
- Final picked dev r 0.53 (inflated by picking the best of 3 on dev; test below is the honest number).

## Re-runs (v2, 8 batches, 160 poles)
- b20, b0, b7: worst 3 by dev mean (b7 was also QC-flagged for write-side names). b9: L22n1337 misread, L0n38 −0.50. b15: L24n2010 overclaimed.
- Opus second opinion on 12 negative/zero-r v1 poles outside those batches (dev only, never in final scores): L24n1824 −0.24 → +0.82 and L24n1420 −0.21 → +0.49 under Opus, so the scorer was confused and b17 was not re-run. L22n1761, L24n1526, L21n1335, L23n353, L1n1682 stayed ≤0 under Opus, so those labels are weak.
- Then b6 (4th worst), b4 (L24n1298 Opus 0.04; L23n1084 missing a trigger) and b21 (L21n1335, L23n353 weak under Opus).
- Same 160 poles: v1 0.447 → v2 0.466. v2 gains: L0n38 −0.50 → 0.48, L23n694 0.21 → 0.61, L24n373 −0.45 → 0.09, L24n1298 −0.27 → 0.26, L23n353 0.00 → 0.32. v2 lost on b6 (0.43 → 0.39) and b4 (0.45 → 0.39); the pick kept v1 there.

## Test (final labels, scored once per condition; flat predictions count as r = 0)
- n = 535. Real r 0.526, shuffled 0.025, lookup 0.416, word list: run by the orchestrator (merged later). Real − shuffled +0.501 [0.468, 0.533]; real − lookup +0.109 [0.086, 0.134]; real − word list: pending the orchestrator's merge.
- By verdict: clear 0.64 (n = 127), likely 0.53 (223), mixed 0.44 (185). Worst test batches: b3 0.38, b4 0.43. 5 flat rows per condition were kept as r = 0.
- Unlike wave 1, these labels beat the token-lookup predictor on test, by +0.11.
