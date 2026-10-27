# R3 full sweep results (2026-10-27)

The full 28-layer atlas screen (spec #11, ticket #12). Run on a rented RunPod RTX A4000 (community,
$0.17/h) via the WebSocket tunnel, auto-ubatch 5. **Sweep spend ~$0.98; total R3 RunPod spend ~$1.55**
(pilot + bad-host retries + sweep), under the $2 pilot and $3 sweep hard stops. Raw output:
`atlas/sweep_run.npz`. Reproduce: `python atlas/screen.py screen --layers 0,..,27 --nulls 1024
--device cuda --variant int4-onnx` then `python atlas/analyze_pilot.py --run atlas/sweep_run.npz`.

## Setup
- **Replica:** `int4-onnx` (the exported INT4 graph's literal bf16 reading). Faithfulness already
  PASSED on the pilot's 3080 Ti run: GPU-vs-CPU mean cosine ≥ 0.99918 on all 28 layers (excl. pos 0).
- **Statistic (frozen):** cross-fitted symmetric 2-fold steer-lens lift at ±0.5, **no floor** (the
  base-prob floor was reverted in the pilot — it buried concept neurons), S = max over signs, on the
  trimmed 99,477-token vocab. p = Phipson-Smyth empirical; BH & BY over m = 86,016.
- **Units:** all 86,016 neurons + **1,024 column-space nulls/layer** (Mary-Roquain floor from the
  pilot's k was ~181; 1,024 used for power margin — nulls are cheap vs the 3,072 neurons).
- **Contexts:** 2 disjoint sets of 16 (8 FineWeb-Edu + 8 UltraChat answered by our replica), 64 tokens.

## Survivors
- **14,771 neurons (17.2%) beat every one of the 1,024 nulls** (p = 9.8e-4, the empirical floor).
- **21,879 neurons (25.4%) survive Benjamini-Hochberg at q = 0.05** (p ≤ 1.27e-2 over m = 86,016).
- **BY = 0** everywhere — but this is the null-count floor, not absence of signal: with 1,024 nulls
  the smallest p is 9.8e-4, while the BY threshold at m = 86,016 is ~5e-8. BH is the operative control
  here (conformal/empirical p-values are PRDS, so BH controls FDR — Bates et al. 2023). Reporting BY
  survivors would need orders of magnitude more nulls; not worth it.
- **Stability:** top-200 overlap across the two disjoint context sets is **0.85-0.96 every layer** (bar
  0.70). The survivor set is reproducible, not noise.

## Survivors rise with depth
Neurons beating all nulls per layer climb from the middle of the network to the output:
`L1: 11,  L12: 68,  L13: 88  …  L21: 1376,  L23: 1518,  L27: 1601`. Late layers write directly to the
residual that becomes logits, so many more neurons measurably shift the next token there.

## Positive controls
- **n908 (L8, ocean +):** S = +3.13, **p = 0.0068, a BH survivor** (GPD tail-fit agrees, 6.1e-3). In
  the full sweep the BH threshold (1.27e-2, set by the whole p-value curve) is looser than the pilot's
  partial-set threshold, so the ocean neuron now clears FDR. Real.
- **n834 (L19, ships/sea −):** S = +4.96, p = 0.156, rank 1436 — still not distinguishable from the
  fair column-space null. Honest finding, unchanged from the pilot.

## Lens words are coherent
Decoded top steer-lens tokens for the strongest neurons per layer read as real concepts (one record
per pole; `+`/`-` = sign). Examples:
- **Concept:** L21 n1527 `random, randomly, randomness`; L22 n796 `return`; L22 n1639 `column`;
  L23 n1482 `interpretation, interpret`; L24 n1478 `recognition, recognize`; L18 n2736 months;
  L21 n741 months; L17 n861 `again`.
- **Structural:** bracket/paren/quote closers (L3 n33 `)))`, L20), punctuation, `.txt/.com/.py` (L20
  n2149), code `=B/<N` (L20), `@`/`[]` (L21).
- **Script / language:** CJK punctuation neurons (many layers), Portuguese (L17 n1021), Spanish (L4).
A handful of top neurons decode to whitespace/empty (`[]`) — low-content, filtered later.

## Projection and cost
- With k = 21,879, the Mary-Roquain rule now says only **~79 nulls/layer** are needed; 1,024 was
  conservative (good — it cost ~26% over the 3,072 neurons, negligible).
- Sweep wall time ~6 h on the A4000 (the LM-head over 99,520 tokens is a ~constant floor per forward,
  so later layers only modestly faster). A 24 GB+ GPU with larger auto-ubatch would roughly halve it.

## Reading the 25%
A quarter of neurons clearing FDR is a lot, and expected: the lift test flags any neuron that
*reliably* shifts the output, and many late-layer neurons are weak token-promotion units (they nudge
`return`, `value`, punctuation). This is real but not all equally interesting. For the UI and labels,
**rank by effect size S** (p50 3.5, p99 13.8, max 19.8) and feature the strongest; the long tail stays
greyed as "real but minor". Labeling all 21,879 poles is a large fan-out — scope R6 to the top band
first (Jake's call).

## Recommendation
R3 is **done and green**: faithful replica, a frozen statistic stable across context sets, a large
reproducible survivor set, and the ocean control recovered. Next is **R6 (labels)**, which needs
Jake's explicit OK for the 3-planner × 15-worker fan-out — and a decision on how many poles to label
(recommend the top few thousand by S, not all 21,879).
