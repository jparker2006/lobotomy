# 10-runtime-reads

Code reader report on lyramakesmusic/neuron-explorer @ f79b219. Analysis only; no code copied.

**1. What this slice does**
Reads per-token neuron activations (`read`, `all_activations`, `full_activations`), computes next-token log-probs with and without pushes or head ablation (`next_logprobs`), and serves direct-path connections from a precomputed index or a live weight-only fallback (`connections`). It also defines the calibration file `act_scale.npz`. Every hook sits on the `down_proj` input.

**2. Key mechanics**
- Activation is the `down_proj` input, `silu(gate)*up` (runtime.py:4, 84). `read` captures before same-layer pushes (runtime.py:90-103), so only earlier-layer pushes can change its value.
- `next_logprobs` runs two full forwards with `use_cache=False` and a float32 `log_softmax` (runtime.py:286-300). Pushes apply as +dose only (`rows=(1,)`).
- Live connections: `mag=|g*u|`, `z=(mag-mean)/(std+1e-8)` within each layer, `sign=sign(g*u)`, with `(1+w)` norm scaling (runtime.py:394-413).
- `act_scale`: running max/min over positions ≥1 in 512-token chunks, starting from 0 (runtime.py:490-503). Server scoring (server.py:357-375) uses `acts/pos` or `-(acts/neg)` with floors `max(0.1*|other side|, 1e-3)`, and zeroes position 0.
- Precomputed path when `k ≤ 24` (runtime.py:363). Otherwise the live path, cached in sqlite (runtime.py:381-436).

**3. Inputs -> outputs**
- In: token ids, `text`/`chat`/`prefix`, pushes `(layer,n,dose,mode)`, heads `(layer,head,scale)`.
- Out: `read` gives `{units:[{text,act,pieces}]}`. `all_activations` gives per-layer (max, argmax, min, argmin) over N neurons. `full_activations` gives float32 [L,T,inter]. `next_logprobs` gives (base[V], alt[V] or None).
- Files: `connections.npz` (`{down,up}_{z f16, layer i8, n i16, sign i8}`, [32,9216,24]); `act_scale.npz` (pos/neg [32,9216], tokens=10240); `vocab.json` (248,077 entries).
- Checked with a stdlib parser, since numpy isn't installed here: rows are sorted by z, sign 0 appears only in sentinels, and max z is 96.00 ≈ (N−1)/√N.

**4. Porting notes**
- Transfers: the `down_proj` hook, the `(1+w)` norm form (Gemma's RMSNorm also uses `1+w`, from memory), precomputed connections (~10 MB for Gemma), and `act_scale` (~300 KB).
- Must change: the model paths `m.model.language_model`, `self.lm.norm`, and `lm_head` (runtime.py:66, 346). The MLP input norm here is `post_attention_layernorm` (runtime.py:394, 401). From memory, Gemma 3 uses `pre_feedforward_layernorm` there and applies `post_feedforward_layernorm` to the MLP output. Context7 returned no Gemma 3 code, so this needs checking against the installed transformers. If it holds, neuron writes are normalized rather than linear `d*a`, and the add-push and connection formulas no longer hold.
- z-max scales with N: about 45 at N=2048, so `zmin=15` (server.py:481) needs recalibrating.
- NPU: no forward hooks. Bake pushes in as per-layer input vectors and head scales as per-head inputs. Calibration and live connections stay on a desktop.

**5. Surprises, bugs, open questions**
- The cache is not keyed by model. `vocab.json` is reused whenever it exists (runtime.py:460-462), and sqlite keys omit the model. `EXPLORER_MODEL=Gemma` would silently use Qwen tokens.
- BOS mismatch: calibration uses `add_special_tokens=False` (runtime.py:493), while `read` and `next_logprobs` use the defaults (runtime.py:259, 282). Harmless for Qwen, but Gemma prepends `<bos>`.
- `/api/fires` (server.py:342) keeps position 0 and uses 1e-3 floors, unlike `score_text`. The bundled UI doesn't call it.
- `read` sets `_read` outside its try (runtime.py:260-263). A ValueError from runtime.py:239-240 leaves capture armed.
- `next_logprobs` builds pushes outside the lock (runtime.py:299). If stats are missing, `layer_resid_p50` runs a forward while another thread's pushes are live.
- `sign(g*u)` is weight-only. The research_log's 99.8% overlap compares two implementations of the same formula, so it doesn't test whether the metric predicts effects.
