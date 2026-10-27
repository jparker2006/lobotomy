# R6 planner p1 — layer 27, 461 poles (per-batch notes: qc_batches.md)

## Numbers
- Labelled 461/461, 0 failed (25 Haiku jobs, 1 retry). Final verdicts: mixed 370, likely 80, clear 11.
- Dev v1: real 0.273, word list 0.154 (n=300), lookup 0.450; real−wordlist +0.114 [0.080, 0.150].
- Final labels = best dev r of v1/v2/pilot per pole: v1 374, v2 84, pilot 3. Final dev r 0.311 (optimistic: the
  pick used dev). On test the v2-picked poles drop from 0.388 (dev) to 0.345, so the dev pick is partly noise.
- TEST (scored once, labels frozen): real 0.320, shuffled 0.068, word list WL, lookup 0.483.
  real−shuffled +0.252 [0.215, 0.289]; real−wordlist GAPWL; real−lookup −0.162 [−0.192, −0.133].
  By verdict (test): clear 0.384 (n=11), likely 0.349 (n=80), mixed 0.312 (n=370).

## Re-runs (v2) and why
- No batch under 0.10 on dev except b23 (1 pole). v2 went to the max of 8 batches: 23, 9, 20 (worst 3), 0, 6, 10, 20
  (QC flags), 16, 17 (next worst). v2 dev: b9 0.18→0.25, b16 0.18→0.30, b17 0.21→0.30, b20 0.18→0.24 (part
  regression to the mean). No v3: nothing left under 0.10.
- Opus second opinion (re-run decisions only, not in any score) on 6 labels that read right but scored low.
  Haiku→Opus dev r: L27n911 −0.76→0.59, L27n386 −0.22→0.62, L27n2738 −0.03→0.42, L27n480 −0.26→0.38, L27n725
  −0.34→0.37, L27n308 0.01→0.16. The labels are fine and the Haiku scorer misread them, so b3/b5/b13 were not re-run.
- Dev only: 3 v1 and 6 v2 all-flat predictions were rescored before the pick (all in b20/b9, mostly "goes strongly
  negative" labels). Test predictions were left exactly as returned (3 flat real ones count as r=0).

## QC findings
- ~80% of poles are token neurons: they fire on one piece of a split rare word (Phryno|soma, sarc|oph|agi) and
  write an unrelated code or format token. Labels mostly describe this right, but the scorer can't tell which
  fragment of an unseen word fires, so r stays low even for correct labels (L27n2738, L27n308).
- Main labeller error: naming the pole from the TOP TOKENS list instead of the ⟦n⟧ peaks. L27n232 "street and
  downtown geography": the peaks are the first digit of measurements (1|150 BC, 1|3 miles). v2 fixed it (test
  0.56). L27n22 "closing quote/bracket": the peaks are opening marks before a capital word. v2 kept the error.
  L27n1468 "power/alliance words": the peaks are p-pieces (desp|ise, under|p|innings). v2 kept it too.
- The retry hint did not stop the anchoring: v2 repeated 3 of the 4 flagged misreads.
- Verdicts over-claim `clear`: L27n865 (6 of 15 peaks are not quotes), L27n1468, L27n1915.
- Off-by-one token: L27n891 names 'user', but every peak is the <|im_start|> before it (test still 0.63).
- Good, specific labels score well: L27n72 newline after a user <|im_end|> (test 0.94), L27n2438 'the' before a
  region (0.58), L27n746 stem before -id/-ide (0.48).
- Scorer artefacts seen on test (kept as is): flat all-zero lists, and identical predictions copied across two
  poles in one reply (L27n461 and L27n1555).
- Edits: shortened 70 names over 4 words without changing meaning (e.g. 'Chat turn boundary, closing
  quote-paren' → 'Turn boundary, quote-paren closers'). Applied after each label run, since the fan-out
  rewrites the labels file.
