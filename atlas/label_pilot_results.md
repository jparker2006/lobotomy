# R6 labeling pilot results (2026-10-27)

Pilot before the full 3,000-pole fan-out (spec #11, ticket #13). 50 poles (2 controls + a spread of
top-S survivors across layers). **All on Cloudroom sub-agents, $0.** Spawning, labeling, scoring and
grading all verified end to end.

## Setup
- **Evidence per pole:** 15 top-activating snippets (peak marked) + 10 logit-lens + 10 steer-lens
  words + 1 steered sample. Built on the INT4 replica (`atlas/label_pilot.py`).
- **Workers:** Claude Haiku (high). Two arms: **batch-1** (one pole per worker, 12 poles) and
  **batch-20** (all 50, 3 workers). **Scorer:** Haiku (high) — predicts each neuron's firing 0–10 on
  **held-out** excerpts from the label alone; real vs shuffled vs the held-out truth.

## Results
- **E4 PASS:** real-label simulation r **+0.283** vs shuffled **+0.144**, gap **+0.139**, 95% bootstrap
  CI **[+0.060, +0.222]** — excludes 0. Labels predict held-out firing better than shuffled labels.
- **batch-20 ≈ batch-1:** real r +0.419 (batch-20) vs +0.398 (batch-1) on the 11 overlap poles — no
  quality loss. **Use batch-20** for the full run (20× fewer calls).
- **Strong separators (clean neurons):** L23n2405 "than-comparison" real r +0.82 vs shuffled +0.16;
  L25n2781 "safety warnings" +0.32 vs −0.35; L19n834 "ships/sea" +0.74 vs −0.10.
- **Labels are honest:** of the 50 high-S poles, verdicts were **40 mixed, 9 likely, 1 clear**. The
  Haiku workers correctly flagged the many late-layer (L25–27) token/structural neurons that *read*
  word-fragments but *write* punctuation/`$`/`=`/brackets as `mixed` (reads ≠ writes), and reserved
  `clear`/`likely` for genuine concepts (sea, than, safety, security, months).

## Findings for the full run
1. **The top-S tail is mostly structural.** ~80% of the highest-effect-size poles are polysemantic
   token neurons, not clean concepts. Ranking the atlas purely by S surfaces these first. For the UI
   and the post, **also rank/feature by verdict** (`clear`/`likely`) so the vivid concept neurons lead.
2. **Shuffle baseline fix:** the derangement rotated labels within a batch, so an ocean neuron got the
   *other* ocean neuron's label — a too-easy "wrong" label (n908 shuffled r spuriously +0.64). For the
   full run, shuffle draws a **random label from a different, non-adjacent neuron**. E4 still passed
   comfortably despite this conservative leak.
3. **Spawn mechanics verified:** Haiku workers + Haiku scorers spawn, run and return parseable JSON.
   The **"Terra 6.1" codex model string is invalid** (`Terra 6.1` and `terra-6.1` both rejected) — so
   ambiguous-case escalation falls back to an **Opus planner scoring**, still $0. If Jake wants Terra,
   he can supply the exact model id.

## Recommendation
Pilot is **green**: the pipeline works, batch-20 holds quality, and labels beat shuffled with a clean
CI. **Proceed to the full 3,000-pole fan-out** — 6 Opus-xhigh planners × 25 Haiku-high workers, batch
20, with the shuffle fix and verdict-aware ranking — **pending Jake's final OK.** $0 API throughout.
