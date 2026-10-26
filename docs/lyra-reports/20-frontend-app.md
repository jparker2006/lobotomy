# 20-frontend-app

Code reader report on lyramakesmusic/neuron-explorer @ f79b219. Analysis only; no code copied.

1. What this slice does

Vanilla-JS controller for the explorer page (`explorer/static/app.js`, 1018 lines). It boots from `/api/index`, holds UI state (`state`, :14), and drives the map and list views, the neuron panel, the 3-row steered chat, lens and fires search, and the Claude tab.

2. Key mechanics

- Hash `#L{l}/N{n}` is written with `replaceState` (:468) and read by `fromHash` (:400). The layer persists in `ne.layer` (:161).
- Dose ladders: ADD `0.05…1.5`, then `1.75 2 2.5 3 4 5` (:15). MUL `0…4` by 0.5, then `5 6 8 12 16` (:16). Defaults: add 0.8, mul 2 (:14).
- Fires marks: Float16 bitmap, `w = min(1, 0.25 + |v|/4)`, neuron `i % shape[1]` (:293-297), capped at 40,000 (:300).
- Name autosave: 500 ms debounce (:608), flushed on neuron switch (:464) and close (:436).
- Claude send aborts or queues; "send now" interrupts (:933-950). Edit and reroll rewind the session (:993-1003).

3. Inputs → outputs

- Boot: `/api/index` (:18), `/api/layout?kind=writes` (:94), `/api/units` (:164), `/api/status` poll (:717).
- Hover and scatter: `/api/hover` (:50), `/api/rollout` (:129). Embeddings: `/api/embeddings` (:176).
- Open neuron: `/api/unit` (:484), `/api/connections?k=24` (:541), `/api/lens` (:543).
- Web: `/api/web` as binary Int16 (:107), `/api/web/summary` (:111).
- Search: `/api/search` (:319). Fires: `POST /api/infer` (:235), `/api/infer/view` (:287), `/api/infer/paths` (:249).
- Rename: `POST /api/name` (:588). Reads: `POST /api/reads` (:704).
- Steer: `POST /api/steer` {layer,n,prompt,dose,mode,history} (:671). Returns SSE `{row,t,error}`. Row 0 is pushed up, 1 unsteered, 2 pushed down.
- Claude: `POST /api/agent/chat` (:866) streams NDJSON `chat|text|tool|tool_result|ui|error`. `ui` actions are handled at :797-820. Also `/api/agent/upload` (:963) and `/api/agent/rewind` (:998).
- Never called here: `/api/fires` and the `/api/agent/*` helpers.

4. Porting notes

- The JS transfers if the API contract holds. Qwen-only bits are the attention tooltips (:35) and the L22 default (:735).
- Gemma 270M has 18 × 2048 = 36,864 neurons, under the 40,000 cap.
- Each steer turn re-sends history, and the server replays it once per row (`explorer/server.py:251-257`). That means three growing contexts. On 2 GB RAM, cap history or run rows sequentially.
- The dose probably has to be a runtime tensor input, or every slider change forces a recompile. This is my inference; confirm with the compiler.
- Not runnable on the board: the lens index and rollouts (offline GPU work) and the Claude agent (spawns `claude -p` server-side). The on-device candidates are steered chat, reads, and fires.

5. Surprises, bugs, open questions

- Opening any neuron, including Claude's `select` (:800), aborts an in-flight steer and wipes the thread (:465, :476).
- Top-level `await` on `/api/index` (:18): if it fails, `pollStatus` never starts (:729).
- `conP` can reject before its handler is attached (:541-546). That fires the global handler (:9) into `#status`.
- :1002 checks `cls.includes("interp")`, but nothing sets that class. Looks dead.
- :511 prints prompt text unescaped. :762 saves the chat as `innerHTML`, and quota errors are swallowed. :980 `.a-msg.user` also matches queued bubbles.
- :101-102 read `webk` and `webhops` from the URL, but nothing writes them.
