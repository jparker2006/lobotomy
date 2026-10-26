# 11-connections-web

Code reader report on lyramakesmusic/neuron-explorer @ f79b219. Analysis only; no code copied.

## 1. What this slice does
`build_connections.py` precomputes each neuron's 24 strongest weight-only "direct-path" links up and down the residual stream into `connections.npz`. `server.py` chains those links into a cone (`cone_links`, `/api/web`) and pools the cone's lens tokens (`/api/web/summary`). Every multi-hop path is a weight-only guess, and the agent prompt says so.

## 2. Key mechanics
- **Weight product** per layer pair i<j (`build_connections.py:3`): `v_A = (1+w_j) ⊙ W_down_i[:,A]`, `P[B,A] = (W_gate_j[B]·v_A)(W_up_j[B]·v_A)`. Code at :45, :67-68. `w_j` is `post_attention_layernorm`, with `1.0 +`.
- **Dropped:** SiLU on gate (:68), attention, intermediate layers, A's activation, and the RMS scale.
- **z:** |P| z-scored within each pair. Down: per column A over B (:71). Up: per row B over A (:76).
- **sign** = +1 if P>0 else −1 (:73, :78).
- **Merge:** running top-24 per neuron and side across all j (:53-60). Empty slots are clamped to −60000 (:85); the server treats z<−1e4 as empty (`server.py:496`).
- **cone_links** (`server.py:481-506`): hop 1 takes the top `first` (default 5, :510). Later hops take the top `k` (default 2) with z ≥ `zmin`=15 (:496). Down hops are +, up hops are − (:488, :500). Each side is capped at cap//2, truncated in BFS order (:505).
- **/api/web** packs `round(10·z)·sign` as 6 int16 per link (:513).
- **web_summary** (:520-577):
  - IDF base `(df+1)/(2N)` over top-8 tokens of both poles (:536-538)
  - `sign[far] = sign[near]·sgn(z)` (:549)
  - `w = z/zmax·0.8^(|hop|−1)` (:555)
  - `lift = (n/len(nodes)) / mean(base)` (:568)
  - `score = w·ln(lift)` if lift>1 (:569)

**Measured** on the shipped npz (L=32, N=9216, K=24; stdlib reader, scratch files in /tmp/scratch):
- 6.86M of 7.08M entries are valid per side.
- Down z: p50 15.8, p99 76.1; 55% are ≥15. Up: p50 14.8; 49% are ≥15.
- My reimplementation of `cone_links` with server defaults, on 300 random neurons: hop-1 links median 10, total links median 858 (max 10,011), distinct neurons median 700, deepest hop median 15.

## 3. Inputs -> outputs
- **In:** HF safetensors for Qwen/Qwen3.5-4B (`down_proj`, `gate_proj`, `up_proj`, `post_attention_layernorm`), on CUDA or MPS.
- **Out:** `explorer/cache/connections.npz`: `{down,up}_layer` int8 [L,N,24], `{down,up}_n` int16, `{down,up}_z` float16, `{down,up}_sign` int8, and `k`. 54 MB.
- **Read:** `runtime.py:302-313` (`con_index`); `/api/connections` (`runtime.py:353-368`); `/api/web` (binary); `/api/web/summary` (JSON with `toward`, `away`, `fed_by`, `held_by`); `/api/infer/paths` (`server.py:443-470`).

## 4. Porting notes for Gemma-270M / Coralboard
- **Transfers:** pure weight math, one pair at a time. Gemma 3 270M gives 153 pairs of 2048×2048 fp32 (16 MB each), so it runs on CPU in minutes. The output is about 4 MB per side, and the cone and summary are lookups that fit in 2 GB.
- **Must change:**
  - The MLP input norm in Gemma is `pre_feedforward_layernorm`. `post_attention_layernorm` is the attention-output norm.
  - Gemma 3 also has `post_feedforward_layernorm` on the MLP output, a nonlinear normalization over the whole layer. The write from A is then no longer linear. Check the key names in the checkpoint.
  - Gemma uses GeGLU, not SiLU.
- **Can't run on the NPU:** all of this is offline or host-side. Only the steered forward pass runs on the NPU.

## 5. Surprises, bugs, open questions
1. **Sign contradiction.** P is quadratic in v_A, so sign(P) does not flip when A's direction flips. `research_log.md:274` removed the "fires ±" toggle for that reason. Yet `server.py:464` multiplies link sign by sign(S[l,n]), which brings the flip back. `web_summary` (:549, :556) also reads sign(P) as the pole the chain drives. Under SwiGLU the first-order push is a·(silu'(g)·u·dg + silu(g)·du), which flips with a and depends on the reader's operating point. So sign(P) is not the push direction. Which behaviour is intended?
2. **zmin=15 barely prunes.** It cuts only about half the stored links. `web_summary` hardcodes 15 (:541) and ignores the `zmin` parameter of `/api/web`.
3. **Cap truncation** follows BFS order, not z (:505).
4. **Ceiling hits:** 12 stored links sit at z≈96, the single-outlier ceiling (n−1)/√n for n=9216. A few columns are dominated by one entry.
5. **Minor:** `first or ci["k"]` (:494) makes `first=0` mean all 24. Summary counts can double-count one node when `" the"` and `"The"` both fold to `the` (:559-566).
