# R9 results: a layout that groups neurons by meaning (2026-10-28)

Ticket #15, spec #11 I4. Everything ran on the cloud CPU: $0, about 25 minutes. Code: `atlas/layout.py`.

## Verdict: I4 fails as pre-registered
The co-activation layout does **not** beat the W_down-UMAP baseline on neighbour-label agreement. The
baseline wins by a small margin, and its CI is clear of 0 in the wrong direction.

| metric (k = 10 within-layer neighbours) | co-activation | W_down UMAP | paired diff, 95% CI |
|---|---|---|---|
| **neighbour-label agreement** (pre-registered; 2,945 labelled poles, L21–27) | 0.0030 | 0.0065 | **−0.0036 [−0.0046, −0.0025]** |
| neighbour co-activation on held-out text (no labels; all 86,008 live neurons) | 0.0108 | 0.0020 | +0.0088 [+0.0086, +0.0091] |

- Random same-layer pairs co-activate at r 0.0002, so co-activation neighbours co-fire about 50× more
  than random, and about 5× more than W_down neighbours.
- **Labels group by what a neuron writes, not by what fires with it.** Many R6 labels name the token a
  pole pushes ("writes 'library'"), and W_down columns are exactly that write direction.
- Both label effects are tiny: neighbours' labels are only 0.003–0.007 more cosine-similar than random
  ones. Most labelled poles are L25–27 subword-fragment neurons with near-identical label wording.
- **Cross-layer** (co-activation only; the baseline's axes aren't shared across layers): labelled
  neighbours from other layers agree +0.0062 [+0.0054, +0.0071] above random.
- Per layer, W_down wins on L22–27 and co-activation only on L21 (29 poles).
- **Exploratory, not pre-registered:** two other variants also lose to W_down (0.0065).
  - A per-layer co-activation UMAP: 0.0038 (diff −0.0027 [−0.0037, −0.0017]).
  - A reads+writes hybrid: 0.0035 (diff −0.0030 [−0.0040, −0.0020]).

## Method
- **Activations:** 16,384 random positions (≥ 1) from the R6 dev+test windows (FineWeb-Edu + UltraChat,
  never shown to labellers), times all 86,016 neurons. They are pole-aligned (activation × R3 sign) and
  stored as fp16.
- **Co-activation layout:** standardize each neuron, then take 64 shared PCA directions with a streamed
  randomized SVD. They explain 26.6% of the variance; co-activation is high-dimensional. Then UMAP to 2D
  (cosine, 15 neighbours, min_dist 0.1, seed 0).
- **Baseline:** per layer, UMAP of the pole-aligned, L2-normalized INT4 W_down columns (same settings).
- **Score** (fixed in `layout.py` before any layout existed):
  - Embed each label's "name. description" with bge-small-en-v1.5.
  - For each labelled pole, take its 10 nearest labelled poles of the same layer in (y, z), in layers
    with ≥ 20 labels.
  - Agreement = their mean cosine minus the mean cosine to all labelled poles of that layer.
  - Paired bootstrap over poles, 10,000 resamples.

## Files (for the UI)
- `coact_yz.f16`: the co-activation layout.
- `wdown_yz.f16`: the baseline.
- Both are raw float16 arrays of shape [28, 3072, 2] (layer, neuron, (y, z)), each axis in [−1, 1], with
  x = layer. Each file is 344 KB.
- `layout.json` holds the format, method and all scores.
