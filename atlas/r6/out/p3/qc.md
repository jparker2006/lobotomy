# R6 planner p3 — L27, 461 poles (S-rank order, 24 batches of 20)

**Labelled:** 461/461, 0 failed (b0–b1 from wave 1; one b6 and one v2-b8 pole needed the script's retry).
**Final verdicts:** mixed 358, likely 89, clear 14. Final = 358 v1 + 101 v2 + 2 pilot (`pick`, best dev r).
**Dev r:** v1 0.296 (461 poles; word list 0.162 on its 300; lookup 0.468). v2 0.285 vs v1 0.224 on the same
160 poles. Pilot 0.109 vs v1 0.338 on its 11 poles. Final (best-of per pole) 0.341: biased upward by picking.
**Flat dev replies:** 17 v1 and 7 v2 poles got constant predictions (mostly all-zero, from 3 lazy scorer
jobs) → dropped and rescored once on dev (lists in `dev_v{1,2}_flat_rescored.json`). Test was not touched.

## Re-runs (v2: batches 0, 1, 2, 7, 8, 10, 16, 22)
No batch had dev r < 0.10 (worst b1 0.171), so: worst 3 (b1, b2, b0) + QC flags b7, b8 + b10, b16, b22
(next worst, 0.21–0.23; b10 also has Opus-confirmed weak labels). After v2 no batch was < 0.10 → no v3.
- b7 L27n2814: v1 label was literally "placeholder" (concepts a…j); the validator let it through. v2 fixed it.
- b8 L27n1705: v1 said it writes 'degree'; its logit/steer lens is 'approximation/approx'. v2 fixed it.
- v2 per batch: b0 0.197, b16 0.247, b22 0.258, b10 0.260, b1 0.282, b2 0.326, b7 0.327, b8 0.379.

## Opus second opinion (dev, 6 low-r labels that read right on the evidence; not in any score)
Haiku r → Opus r: L27n472 −0.39→+0.68, L27n1808 −0.40→+0.60, L27n2020 −0.34→+0.33 (scorer error, the label
is fine); L27n1109 −0.46→−0.15, L27n2622 −0.26→−0.08, L27n2653 −0.41→−0.32 (label really doesn't predict).
So some negative dev r is Haiku-scorer noise on 20 excerpts. That adds noise but doesn't point at a batch.

## QC findings (2+ labels/batch read against `show` peak tokens, plus automatic checks)
- Most labels name the peak ⟦n⟧ tokens correctly and say what the pole writes. ~75% are split-word-fragment
  token poles ("first piece of a split word, e.g. 'pl' in pl|agues"); `mixed` is the honest verdict there.
  Several such labels in one batch look alike, so a scorer can't tell them apart: that's the main limit.
- `concepts` copied verbatim from TOP TOKENS in 232/461 final labels. Harmless on clean token poles, but
  misleading when the top-token list disagrees with the snippets: b1 L27n1474 (dense prose pole, concepts
  floral/contaminated/knock…), b4 L27n1370 (chat turn-end newline, concepts assistant/pr/ø/Rob…).
- Good concept-level reads exist: L27n1328 URL 'www' → file extensions (dev 0.80), L27n305 electronics
  specs (0.73), L27n2615 token before 'science', L27n1242 object noun before 'case'.
- Names: 51 v1 and 18 v2 names over 4 words shortened, meaning kept (v1 after its dev scoring, v2 before).
- Ops: one 15-pole dev rescore job failed to spawn every time (rc 1, no message), though each of its pole
  sections had gone through before; I cancelled it and resubmitted the same poles in jobs of 5, which worked.
  While debugging I printed the first ~1.5 kB of that prompt, which included part of 3 dev excerpts for
  L27n1325. I made no decision from it. L27n1325 (b5) was not re-run and kept its v1 label.
## Test (final labels, scored once per condition)
real **0.311**, shuffled 0.083, word list 0.188, lookup 0.478 (n=461). real−shuffled +0.228, CI95 [0.194, 0.263];
real−word list +0.123, CI95 [0.092, 0.154]; real−lookup −0.168, CI95 [−0.198, −0.137]. By verdict: clear 0.347 (14),
likely 0.372 (89), mixed 0.294 (358). Flat test replies, counted as r = 0 as the rule says: real 7, shuffled 2,
word list 3. Labels beat both baselines clearly but stay below the free token-lookup predictor, as in wave 1.
