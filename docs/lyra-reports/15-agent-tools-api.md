# 15-agent-tools-api

Code reader report on lyramakesmusic/neuron-explorer @ f79b219. Analysis only; no code copied.

## 1. What this slice does
The Claude chat's tool backend: ten FastAPI handlers (server.py:589–825) that read cached neuron data and, when the live Qwen3.5-4B is loaded, run steering, activation traces, logprob diffs and multi-text screens. The MCP tools in `agent_mcp.py:171–220` map onto these endpoints. Read-only; I did not run anything, so behaviour claims come from the code.

## 2. Key mechanics
- Score: `acts/pos` if act ≥ 0, else `-acts/neg`. Position 0 is zeroed (server.py:375, 377).
- Add push: `r·dose·(resid_p50 / ‖down_col‖)`, with r = +1 up, 0 baseline, -1 down (runtime.py:174, 244–245). Steer defaults: dose 0.8, mode add, max_new 150 (server.py:666, 671).
- Mul push: up = `dose`, down = `-dose`, baseline = 1 (runtime.py:242).
- Pushes apply at the down_proj input. An activation read is captured before same-layer pushes (runtime.py:90–104).
- Logprobs: `kl = Σ p_base·(log p_base − log p_alt)` (server.py:750). Rose/fell are the top-12 of `exp(alt) − exp(base)` (server.py:746–747).
- Cone: first hop uses the precomputed top-24 links; later hops use k=2 with z ≥ 15, capped at 20,000 per side (server.py:481, 632).

## 3. Inputs -> outputs
Reads `cache/act_scale.npz` (pos/neg `[32,9216]`, tokens 10240), `steer_index.npz` (up/down `[32,9216,12]`, KL, dose 0.8), `lens_index.npz`, `connections.npz` (k=24), `names.json`, sweep runs, and the live model.

| Endpoint | Params | Returns | Experiment |
|---|---|---|---|
| GET overview | — | model, arch, sweep layers, live_model | orientation |
| GET names | — | `{"L14/N3262": name}` | — |
| GET neuron | layer, n | both-pole lenses, steer lens + KL, act range, sweep block, 24 links per side | profile |
| GET cone | layer, n, k=2, limit=120 | nodes: hop, abs z, sign, steer_up | multi-hop leads |
| POST steer | prompt, layer+n or neurons[], dose, mode, max_new | up / baseline / down text | causal generation |
| POST activations | layer, n, text, chat=false, neurons[], heads[] | [token, act] pairs, top 10, act_range | trace; does X drive N |
| POST logprobs | text, chat=true, prefix, k=15, targets[], neurons, heads | top, targets, kl, rose, fell | next-token effect |
| POST compare | neurons[], texts[], chat | max/min token and peak_score per neuron × text | screening |
| POST fires | text, token, limit=30, layers, chat | neurons: pole, score, act, steer | what fires on a token |
| POST ui | chat, action, payload | queued to browser (app.js:798–820) | map control |

## 4. Porting notes
- Transfers: the JSON contracts, score normalization, the three-row steer batch, KL/rose/fell, `_live_off` cache-only mode, and the down_proj hook (runtime.py:80–104), which fits any gated MLP.
- Must change: Qwen hard-codes at server.py:29 and 602, runtime.py:15/65, the `<|im_end|>` EOS at runtime.py:181, and the linear-attention head code at runtime.py:70–77. Layer and neuron ranges are hard-coded at agent_mcp.py:14.
- Cost: full activations are 18×2048×4 B ≈ 147 KB per token (Qwen ≈ 1.2 MB). Cap `MAX_INFER_TOKENS` (server.py:331) near 256 on 2 GB. Logprobs need the full vocab each step (Gemma 3 ≈ 262k; verify in its config).
- Off-device: the agent loop, sweeps, index builds, calibration and the 3D UI. The NPU runs only the forward pass. Steering needs a per-neuron push tensor as a graph input, since shapes are static.
- Recalibrate act_scale, steer KL and the residual-norm dose unit on the quantized graph. Host fp32 values will not match.

## 5. Surprises
- `mul` footgun: the default dose 0.8 also applies in mul mode (server.py:666). The "up" row then damps ×0.8 and "down" flips ×-0.8.
- `cone` `sign` is each link's own sign, not chained from the origin. `steer_up` is always the up pole (server.py:647–648). `web_summary` chains signs and picks the pole per node (server.py:549, 556), so deep cone nodes can be misread.
- `fires` with a `token` matching position 0 returns zero-score neurons with pole "-", because position 0 is zeroed (server.py:377, 796).
- `neuron.sweep` is non-null for any neuron in a swept layer, even when unranked (server.py:621).
- `overview` reports Qwen even when `EXPLORER_MODEL` is set (server.py:602 vs runtime.py:17).
- The `steer_up` fallback silently holds logit-lens tokens when steer_index is missing (runtime.py:370).
- All live calls share one lock (runtime.py:39), and `max_new` is uncapped, so one long steer blocks the rest.
- `compare` silently truncates to 300 neurons and 24 texts (server.py:765, 767).
- act_scale is measured on only 10,240 tokens (runtime.py:485), so "1 = usual max" is noisy.
