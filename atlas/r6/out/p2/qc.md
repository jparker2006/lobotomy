# R6 planner p2: QC and scores (L27, 461 poles, 24 batches of 20)
**Labelled:** 461/461, 0 failed keys (v1: one 5-pole retry; v2: one 20-pole job re-parsed). Final verdicts: 362 mixed,
91 likely, 8 clear. 75 names over 4 words shortened in place, same meaning (old text in `name_orig`).

**Test (final, scored once per condition):** real r **0.322**, shuffled 0.066, word list 0.202, token lookup 0.491
(n=461). Real − shuffled +0.257, 95% CI [0.223, 0.290]; real − word list +0.121, CI [0.093, 0.149]; real − lookup
−0.169, CI [−0.197, −0.141]. By verdict: clear 0.36 (n=8), likely 0.38 (91), mixed 0.31 (362). Lazy all-flat scorer
replies were kept as the script returned them: 1 flat pole in real, 15 in shuffled (one batch was lazy on all 3
attempts), 2 in word list, all r = 0.

**Dev:** v1 0.289 (lookup 0.469; word list 0.153 on 275 poles, real − word list +0.128, CI [0.09, 0.16]). No batch under
0.10; worst b13 0.14, b4 0.16, b5 0.20. Final after pick 0.336, inflated because it takes the max of noisy draws. Dev and test
per-pole r correlate only 0.26: the 20-excerpt Haiku scorer is noisy per pole.

**Re-runs (v2, 160 poles):** b13, b4, b5 (worst 3), b9 (QC flag), b18, b14, b21, b7 (next lowest, 5–7 poles under 0.1 each).
Same poles v1 → v2 on dev: 0.198 → 0.261, v2 better on 91/160 (b4 0.16→0.37, b7 0.22→0.38; b14 and b21 slightly worse).
No v3 (nothing under 0.10). Pick: 368 v1, 91 v2, 2 pilot (L27n390, L27n1672: vaguer, older-evidence labels that
won on dev). On test the re-run batches score 0.299 vs 0.335 for the rest. Part of the dev gain is regression to the mean.

**Scorer hygiene (dev only):** flat (constant) predictions are non-answers. I dropped and rescored 6 in v1 and 17 in v2
from lazy scorers that fell under the script's 80% whole-reply cutoff. Opus second opinion on 8 poles with Haiku r ≤
−0.21 whose labels read right (side file, not in any score): L27n1379 −0.44→+0.51, L27n1385 −0.32→+0.40, L27n2011
−0.25→+0.33, L27n2526 +0.17, L27n2132 +0.06, so the scorer was confused and the labels are fine (all in v2 batches anyway). L27n680
−0.31, L27n1658 −0.17, L27n612 −0.03 under Opus: those labels really fail, but their batches (b19, b8, b20) sat above 0.30
and outside the 8-batch cap, so they keep v1.

**QC findings (≥2 labels read per batch against evidence):**
- Most poles are split-word/name-fragment readers with an unrelated write. Workers describe them concretely
  (which fragment, examples, write) and mark them `mixed`. Good: L27n2371 "piece before 'ps'" (ile|psy, se|ptic; writes ps),
  L27n283 "slot before a number" (space before 2014/21 million; writes digits, `clear` earned), L27n1327 newline before
  the assistant turn, L27n238 digits inside numbers.
- Misreads: L27n1251 led with its 'check' write while its 7 top peaks are 'in' in get/stay idioms; v2 "'in' after
  get" is final. L27n716 was off by one ("verb after modal"; really the slot before a participle); v2 is final.
  L27n3022 followed its write ('network') while its 9–10 peaks are other name pieces; v2 fixed the text, but v1 kept
  the higher dev r, so the misread v1 label stands.
- Over-claimed `clear`: L27n784 (surname write is clear, the read is diffuse name pieces; v2 `likely` is final), L27n1346
  "search and index" (top peak Gu|imarães; about half the snippets fit; still `clear` in final).
- Weak spots left: some workers copy the TOP TOKENS list into `concepts`, and one (b8) used category words ("split
  word") instead. Generic slot labels (L27n612 "determiner before noun") cannot rank text. Several labels lead on a
  secondary pattern (L27n221 genre words, while its 9–10 peaks are 'be' in "should be told").
