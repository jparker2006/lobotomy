# R6 planner p4 — layers 25–26, 542 poles

**Labelled:** 542/542 (v1 0 failures, v2 0 failures). Final verdicts: mixed 402, likely 111, clear 29.
**Dev r:** v1 0.376 (word list 0.17 on 344 poles, lookup 0.50). Final 0.408, inflated by picking the best of 2–3
versions on the same dev excerpts (test came in at 0.387).

**Test (final, scored once, 542 poles):** real **0.387**, shuffled 0.051, word list 0.254, lookup 0.514.
real − shuffled +0.336, CI95 [0.304, 0.367]. real − word list +0.134, CI95 [0.111, 0.160]. real − lookup −0.127
[−0.154, −0.101]: the free token lookup still beats the labels. By verdict: clear 0.60, likely 0.45, mixed 0.36.
77 poles are under r 0.10 on test. Flat test predictions (count as r = 0): real 2, shuffled 0, word list 3.

## Re-runs
- No v1 batch was under 0.10 (worst: b20 0.26, b15 0.29, b12 0.30, b2 0.30). I re-ran the 8 lowest batches as v2:
  20, 15, 12, 2, 10, 3, 21, 4. Batch 4 was also QC-flagged for write-only names.
- On those 160 poles, v1 dev 0.305 vs v2 0.327; v2 better on about half, so mostly scorer noise. Pick: v1 462, v2 78,
  pilot 2. Test by picked version: v1 0.392, v2 0.365, pilot 0.16 (n=2). No v3 (no batch under 0.10).
- Dev-only rescoring: one scorer reply for 15 batch-10 v2 poles came back flat three times; I rescored those plus
  L26n1617 (v2) and L26n2729 (v1). Nothing was rescored by hand on test.
- Opus second opinion (dev, 4 poles, used only for re-run calls). L25n943 Haiku −0.31 → Opus 0.98: label fine,
  scorer confused. L26n403 −0.21 → 0.35: scorer noise, batch 6 not re-run. L26n2678 0.00 → 0.11 and L26n2979
  −0.41 → 0.05: weak labels under both scorers (b15 re-run; b0 at 0.39 left alone).

## QC findings
- Most strong poles here are subword-onset/fragment neurons ("first piece of a split name"). Labellers mostly call
  them `mixed`, which is honest; the steer/logit writes are often an unrelated token ('rank', 'sector', 'button').
- Good: L26n1853 "Articles before noun phrases" (dev 0.72), L25n2283 "Foot/thumb before print" (clear; foot|print,
  thumb, Palm), L26n823 "Third digit of years" (8|1|4, 16|1|1), L25n630 "Section-label noun before number".
- Misreads: L26n241 "Noun-phrase end words" misses the real trigger (last word of a bulleted line before ⏎). L25n1854
  "Books and fiction words" describes top tokens, but the peaks are words just before 'read' (worth|, a must|, widely|).
  L26n1367 "Shell paths" is marked `likely` but rests on 2 of 15 scattered snippets. L26n1001 "Sequence words" is
  generic. L26n713 misses that its letters sit right before a 'J' (MS|JE, BH|J), which matches its 'js' write.
- Write-led labels: L26n435 "Buy/purchase token writer" names only the write and puts buy/purchase in concepts
  (scorers see concepts). Same pattern in L26n1354 ('button') and L25n2905 ('best'). v2 rewrote L26n435 around its reads, but v1 won on dev and stays.
- Concepts: 343/542 v1 labels copy ≥8 of 10 concepts from TOP TOKENS. That is allowed, but it can contradict the
  description: L26n1122 describes 'that'/'they' opening a clause but lists 'ideally, Sal, val, galaxies'.
- L26n1211: the pilot label ("miles" after a number, older evidence) doesn't match current snippets yet won on dev
  (0.45 vs −0.02); the pick rule kept it.
- Names: 44 v1, 25 v2 and 1 pilot name were over 4 words. I shortened them without changing the meaning; the
  original is kept in `name_orig`.
