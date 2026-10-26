# 12-layout-embeddings

Code reader report on lyramakesmusic/neuron-explorer @ f79b219. Analysis only; no code copied.

## 1. What this slice does
`build_layout.py` places each layer's 9,216 neurons on a 96×96 sheet so neurons with similar `down_proj` columns sit near each other. `build_embeddings.py` embeds each sweep rollout with bge-small and UMAPs it to 2D for the embeddings view. The two server handlers only serve cached files and compute nothing.

## 2. Key mechanics
- Writes input is `down_proj.weight.T`, shape [N, hidden] (build_layout.py:43). Reads concatenates gate and up rows (L45).
- Each row is L2-normalised (L46), columns are mean-centred (L47), then SVD gives `Z = U[:, :50] * S[:50]` (L48–49).
- `umap.UMAP(n_neighbors=30, min_dist=0.05, metric="cosine", random_state=0)` runs on Z (L50).
- Snap: sort by y, cut into 96 bands of 96 (L51–53), sort each band by x (L54), set `cell = r*96 + col` (L55). Each layer is its own sheet.
- Rollouts: `text[:600]` characters, `BAAI/bge-small-en-v1.5`, normalised (build_embeddings.py:19–20, 28). The prompt is not embedded.
- `umap.UMAP(n_neighbors=15, min_dist=0.08, metric="cosine", random_state=0)` (L31). x and y are min-max scaled to [0,1] per axis (L32).
- `/api/layout` (server.py:470–478) returns base64 int16 plus `shape`. `/api/embeddings` (L198–204) returns `<run>.umap.json`, or `{}` if missing.

## 3. Inputs -> outputs
- Layout in: Qwen3.5-4B safetensors, `model.language_model.layers.{l}.mlp.down_proj.weight`. Out: `explorer/cache/layout_writes.npz` with one array, `cell` int16 [32, 9216], 590 KB. I checked it with a stdlib parse: every layer is a permutation of 0–9215.
- Embeddings in: `runs/<run>.jsonl` (`text, kind, neuron, prompt, dose`). Out: `<run>.emb.npy` (float16) and `<run>.umap.json` with parallel arrays `x, y, row, kind, neuron, prompt, dose`. The frontend uses `row` for `/api/rollout` (app.js:125–130).
- The frontend decodes the layout at app.js:84–92 and places neurons with `gridSide = Math.ceil(Math.sqrt(N))` (map3d.js:73–77).

## 4. Porting notes for Gemma-270M / Coralboard
- Both scripts run on a host. Nothing here runs on the Coralboard. Only the layout table ships, about 74 KB for 18×2048.
- **Breaks as-is:** 2048 is not a square. `assert side*side == N` (build_layout.py:31) fails. The frontend's `ceil(sqrt(N))` would give 46 → 2116 cells. Both need a rectangular grid rule.
- Qwen-specific names: `cfg["text_config"]` (L28), `model.language_model.layers` (L41), and the default model (L20). Gemma 3 checkpoints probably use a top-level config and `model.layers.{l}`. I did not download anything, so verify this.
- PCA 50 is valid for hidden=640. Gemma's gated MLP still has gate, up and down, so both kinds apply.
- bge-small, sentence-transformers, and umap-learn stay on the host.

## 5. Surprises, bugs, or open questions
- **Metric mismatch:** the layout uses cosine on mean-centred PCA scores (L46–50). The validation in research_log.md:169–180 measures raw down-column cosine. I did not check whether centring matters.
- **Weak signal:** research_log.md:173–182 reports recall of true neighbours at 4.7% seriated vs 1.0% index (L8), and 12.9% vs 0.7% (L22). The log's own reading is "slightly related, sometimes."
- **Strip seriation:** columns are ranked by x only inside each y-band (L51–55). Vertical neighbours are not necessarily similar.
- `random_state=0` makes umap-learn single-threaded. That fits the ~50 s per layer in research_log.md:282. It is a speed cost, not a bug.
- `umap.json` repeats the prompt string on every row. Nothing ties it to the jsonl, so it can go stale silently.
- `.emb.npy` is written but nothing reads it, so re-running UMAP always re-embeds.
- server.py:200 calls `load_layer`, which parses the whole rollout file, just to find the path. `layer_runs()` would do.
