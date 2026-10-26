# 14-server-infer

Code reader report on lyramakesmusic/neuron-explorer @ f79b219. Analysis only; no code copied.

**1. What this slice does**
Lets you paste text and see which neurons fire on it. `/api/infer` runs the text once and caches every neuron's activation per token. `/api/infer/view` filters that cache. `/api/infer/paths` returns direct links among neurons active on one token. `/api/search` matches lens tokens and needs no model. `/api/fires` is an older variant that the UI never calls. This is from reading only; nothing was run.

**2. Key mechanics**
- Score, `server.py:371-375`: `pos = max(pos_cal, 0.1·−neg_cal, 1e-3)`, `neg = min(neg_cal, −0.1·pos_cal, −1e-3)`. Then `score = act/pos` if `act >= 0`, else `−act/neg`. Position 0 is zeroed (`:376-377`).
- Calibration, `runtime.py:485-505`: per-neuron max and min over 10k FineWeb-Edu tokens, positions ≥1.
- Input cap: `ids[:1024]` (`:331`, `:369`), silently.
- View filter, `:411-424`: keep `|score| ≥ min` (UI sends 1.0), apply sign and layer range, and optionally match a lens token in the top 6. Return the top 300 by |score|. `pos=-1` means each neuron's strongest token.
- Search, `:301-326`: substring match over the vocab. A neuron matches if one of its top-12 lens tokens matches. Sorted by best matched index, then hit count.
- Paths, `:448-464`: take the top-300 |score| neurons at one token, then keep precomputed `down_*` links (top-24 per neuron) whose both ends are active. Sign is `down_sign × sign(writer score)`. Hop is always 1. A "path" is a single edge, not a chain.

**3. Inputs -> outputs**
- Reads: `cache/act_scale.npz` (pos/neg `[32,9216]`), `lens_index.npz` and `steer_index.npz` (`[32,9216,12]` int32), `connections.npz` (`[32,9216,24]`), `vocab.json` (248,077 tokens). The live model is touched only via `full_activations` and `all_activations`.
- `POST /api/infer` → `{ready, tokens, pos}`. `pos` is each display unit's last token index.
- `GET /api/infer/view` → `{results, total, map, shape}`. `map` is base64 float16 `[L,N]`.
- `GET /api/infer/paths` → int16 `[k,6]`: src L, src N, dst L, dst N, `round(10z)×sign`, hop.

**4. Porting notes for Gemma-270M / Coralboard NPU**
- Transfers: search, view, paths, and the scoring math are lookups over small arrays. For Gemma, my estimate is about 3.5 MB of lens index, 11 MB of connections, and 0.3 MB of act scale.
- Must change: activation capture. `full_activations` needs every layer's `down_proj` input for T tokens. NPU graphs are fixed-shape, so pad to a fixed T and compile extra outputs. Calibrate on the quantized graph, not the fp model. Float32 `[18,1024,2048]` is about 151 MB before copies, so store float16 or reduce per neuron.
- Gemma differences, to verify in the HF code: a GELU-tanh gate, and a post-feedforward RMSNorm that breaks the linear `W_down` writer assumption in `build_connections.py:45,68`. Its tokenizer adds `<bos>` by default; Qwen's does not. `index.html:67` hard-codes 31 layers.
- Stays on the laptop: calibration (FineWeb download) and the lens and connection builds.

**5. Surprises, bugs, or open questions**
- `/api/fires` never zeroes position 0. `all_activations` defaults to `skip_first=False` (`runtime.py:472`), unlike `score_text`. The attention sink probably tops that list. Unverified.
- `/api/fires` has no model guard: without a model, `rt.tok` is None and it raises. `limit ≥ 9216` also crashes `argpartition` (`:349`).
- `_infer` is one global slot, so two tabs overwrite each other.
- `token_units` (quadratic decode, `runtime.py:213`) and search's Python loop run on the event loop, not in a thread. Broad queries may stall the SSE stream. Not timed.
- The link sign is `sign((gate·v)(up·v))`, a proxy rather than the derivative of `silu(g)·u`. Keeping only the top 24 links per neuron misses real co-firing pairs.
