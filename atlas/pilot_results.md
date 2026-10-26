# R3 pilot results (2026-10-26)

Two-layer FDR pilot for the atlas (spec #11, ticket #12). Run on a rented RunPod RTX 3080 Ti
(community, $0.18/h) via the WebSocket tunnel. **Total RunPod spend: $0.44** (hard stop was $2).
Raw output: `atlas/pilot_run.npz`. Reproduce: `python atlas/screen.py screen --layers 8,19 --nulls 8192
--device cuda --variant int4-onnx` then `python atlas/analyze_pilot.py --run atlas/pilot_run.npz`.

## Setup
- **Replica:** `int4-onnx` (the exported INT4 graph's literal bf16 reading). T5's `npu` board kernel
  tracks it to cosine ≥ 0.97 worst-position; onnx runs ~10x faster, so the screen uses it.
- **Contexts:** 2 disjoint sets of 16 (8 FineWeb-Edu + 8 UltraChat answered by our own replica), 64
  tokens each, balanced A/B halves (`atlas/pilot_contexts.npz`).
- **Units:** all 3,072 neurons/layer + **8,192 column-space nulls/layer**, both signs.
- **Statistic (frozen):** cross-fitted symmetric 2-fold steer-lens lift at ±0.5, S = max over signs,
  on the trimmed 99,477-token vocab. **No base-prob floor** (see below). p = Phipson-Smyth empirical;
  BH & BY over m = 86,016.

## Faithfulness (GPU vs CPU replica) — PASS
GPU `int4-onnx` vs CPU `int4-onnx`, teacher-forced on `eval/ref_greedy.json`: **mean per-layer cosine
≥ 0.99918 on every one of the 28 layers** (excluding position 0). The attention-sink position 0 dips
to ~0.95 at worst and is excluded (ref/README flags it); the lift averages over positions, so it is
immaterial. The GPU replica reproduces the CPU replica.

## The base-prob floor was tried and reverted
A base-probability floor (only tokens with mean base prob > 1e-4 may be picked) was added after a CPU
mini-test, to kill rare-token log-prob "explosions". The pilot showed it is **wrong on real text**:
concept tokens too rare in the eval corpus (e.g. `sea`, `naval`) fall below the floor, so it buries
real neurons — L8 n908 went to floored p = 0.943 (rank 2803/3072). The CPU test that motivated it was
contaminated (one context was an ocean sentence). **Reverted**: with enough real contexts the cross-fit
alone defeats rare-token noise (random pushes don't re-raise the same junk on the held-out half). The
floor is kept only as a reported secondary (`p_floored`).

## Results (no-floor primary)

| Layer | neurons beating all 8,192 nulls | top-200 overlap set1/set2 | control | control p | control rank |
|---|---|---|---|---|---|
| L8  | 111 | **0.85** | n908 (ocean +)      | 0.0052 | 438 / 3072 |
| L19 | 345 | **0.92** | n834 (ships/sea −)  | 0.171  | 1448 / 3072 |

- **FDR over the 6,144 tested neurons (m = 86,016): BH q=0.05 → 679 survivors** (p ≤ 3.66e-4). BY → 0
  (BY needs far stronger evidence at this null count; expected).
- **Stability** (top-200 overlap across disjoint context sets): 0.85 / 0.92, both clear the 0.70 bar.
- **Null scale differs by layer:** L8 null S mean +2.05 (p95 +2.64); L19 +4.56 (p95 +5.36) — the lift
  is in log-prob nats and scales with each layer's residual norm.

## Positive controls — a finding, not a pass
The two CPU-demo neurons are **real but below the strict FDR bar** against a *fair* column-space null:
- **n908 (L8, ocean):** beats 99.5 % of 8,192 random pushes (p = 0.005, GPD tail-fit agrees 5.1e-3),
  rank 438/3072 — clearly special, but not among the 679 FDR survivors.
- **n834 (L19, ships/sea):** p = 0.17, rank 1448 — not distinguishable from the null here.

This matches the map's P5 probe ("the atlas is the tail; column-space nulls are hard at L8/L19"). The
demo found them with *isotropic* controls and a single hand-picked prompt; against a fair null and
held-out text they do not clear 5 % FDR. We report this honestly rather than weaken the null.

## Projection to the full 28-layer sweep
- Scaling the 679 BH survivors (2 layers) gives **k ≈ 9,500** real neurons model-wide (an over-estimate:
  L8/L19 were chosen for known structure). By Mary & Roquain (2022), `n ≈ m/(αk)` then needs only
  **~180–520 nulls/layer**, not 8,192 — far cheaper and still valid.
- **Cost:** the pilot took 2,641 s for 2 layers at 8,192 nulls with an **unbatched per-unit GPU loop**.
  Extrapolated that is ~10 GPU-h (~$1.9–3.5) for the full sweep. **Batching the units** (the loop is the
  bottleneck, GPU was compute-light per step) plus the lower null count should bring the full sweep to
  **~1–2 GPU-h (~$0.4–1)**.

## Recommendation
The method is sound: faithful GPU replica, a frozen statistic that is stable across context sets, and a
healthy population of FDR survivors. **Before the full sweep:** (1) batch the screen, (2) drop to
~512 nulls/layer (Mary-Roquain), (3) decide how to present the positive-control finding in the post.
Then run all 28 layers. **Not started without Jake's explicit OK** (and he tops up RunPod first).
