# 13-server-core

Code reader report on lyramakesmusic/neuron-explorer @ f79b219. Analysis only; no code copied.

### 1. What this slice does
App setup, a localhost-only guard and a no-store middleware, cached loaders for sweep runs, and the endpoints that serve them, write names, and stream steered generations from `Runtime`. Pinned commit `f79b219`, read-only.

### 2. Key mechanics
- **Guard:** 403 unless `Host` is in `{localhost, 127.0.0.1, [::1]}:PORT`, `Origin` (if sent) matches, and `Sec-Fetch-Site` is not cross-site or same-site (server.py:40, 47-50). No token. Binds `127.0.0.1` (server.py:1075).
- **Loaders:** `layer_runs` takes the max of `(design_name=="main_rel", jsonl bytes)` per layer and skips `calib` (server.py:70, 75). `load_layer` caches on `(path, mtime(jsonl), mtime(units))` (server.py:89). It rebuilds `.units.jsonl` via `.tmp` + `os.replace` (server.py:95-100) and keeps 8 slim row fields (server.py:104).
- **`/api/steer`:** rows 0/1/2 = `+dose` / unsteered / `−dose` (runtime.py:174). SSE events are `data: {"row","t"}`. Errors are `{"row":null,"error"}`. The end marker is `{"row":null,"t":null}`. A disconnect sets `cancel`, which is checked per token (server.py:273-277; runtime.py:188).

### 3. Inputs -> outputs
Reads `runs/*.{meta.json,jsonl,units.jsonl,umap.json}`. Reads and writes `explorer/names.json`.
- `GET /api/index` → `{model, arch{n_layers,hidden,inter,layer_types,…}, layers{"L": file}, status}`
- `GET /api/units?layer` → `[{kind: neuron|random, n, z (=mass), null_pct, words[5], name}]`, or `[]`
- `GET /api/unit?layer&n&kind=neuron` → `{layer,n,kind,name,run,prompts,rollouts[],baselines[],stats,rank,n_ranked}`
- `POST /api/name {layer,n,name}` → `{ok:true}`. A blank name deletes the key.
- `GET /api/embeddings?layer` → umap.json file, or `{}`
- `GET /api/rollout?layer&i` → one row `{kind,neuron,prompt,dose,rel_push,text}`
- `GET /api/connections?layer&n&k=24&lens=5` → `{down[], up[]}`, items `{layer,n,z,sign,lens,lens_neg,name,words≤4}`
- `GET /api/hover?layer&n` → `{lens, lens_neg, steer, steer_neg, con}`
- `GET /api/lens?layer&n` → `{up, down}` (12 tokens), plus steer and KL fields when indexed
- `POST /api/reads {layer,n,text,chat}` → `{units:[{text,act,pieces}]}`
- `POST /api/steer {layer,n,dose,prompt,history,mode="add",max_new=300}` → SSE. Errors arrive in-stream.

### 4. Porting notes for Gemma-270M / Coralboard NPU
- **Transfers:** the endpoint contract, file formats, mtime caching, slim rows, the SSE protocol, and the guard.
- **Qwen-specific:** `AutoConfig.from_pretrained("Qwen/Qwen3.5-4B")` runs at import (server.py:29). `ARCH` has hybrid-Qwen fields (server.py:30-33). The model name is hard-coded (server.py:148). Drop `transformers` on the board. EOS `<|im_end|>` (runtime.py:181) becomes Gemma's `<end_of_turn>`.
- **Steering:** it is a hook on the down_proj input (runtime.py:80-104). On Torq, make the push a graph input that defaults to zeros, so changing neuron or dose needs no recompile.
- **Precompute:** `connections` and `lens` are weight math with caches (runtime.py:329-436). Compute them on a PC and ship the caches. Only `/api/steer` and `/api/reads` need NPU forward passes.
- **RAM:** `load_layer` holds a layer's full rows in memory (server.py:106-111). On 2 GB, ship only units plus slim rows.

### 5. Surprises, bugs, or open questions
- `/api/rollout` with no run raises `None["rows"]` and returns 500 (server.py:209). A negative `i` silently wraps.
- `/api/unit` with no run returns only `{layer,n,kind,name}` (server.py:166-168), so clients get missing keys.
- `names.json` is written in place (server.py:193-194). A crash mid-write breaks `names()` (server.py:142) and every endpoint that calls it.
- `z` is really `mass`. The rank.py:35 docstring says "sorted by z", but the code sorts by mass (rank.py:80).
- `steer_multi` holds `rt.lock` for the whole generation (runtime.py:169). Uncached `connections` (runtime.py:389) and `read` (runtime.py:255) stall behind it.
- Any mode other than `"mul"` silently acts as add (runtime.py:241). `max_new` defaults to 300 here and 200 in runtime (runtime.py:159).
- `is_disconnected()` is checked only after an event (server.py:273). Cancelling during an idle gap depends on Starlette closing the generator. I did not verify the pinned version.
- Hover's `con` lacks the `name`/`words` enrichment that `/api/connections` adds (server.py:219-221 vs 227-230).
