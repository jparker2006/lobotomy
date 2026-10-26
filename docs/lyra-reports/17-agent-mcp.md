# 17-agent-mcp

Code reader report on lyramakesmusic/neuron-explorer @ f79b219. Analysis only; no code copied.

## 1. What this slice does
`explorer/agent_mcp.py` is a stdlib-only stdio MCP server that gives each `claude -p` chat 21 tools. Each tool is a thin HTTP client to the explorer's own server. Reads go to `/api/agent/*`, live-model calls go to POST `/api/agent/*`, and map actions go to `/api/agent/ui`, which queues events for the browser.

## 2. Key mechanics
- **Transport:** newline-delimited JSON-RPC on stdin/stdout (`:230-252`). `initialize` echoes the client's `protocolVersion` (default `2025-06-18`) with caps `{"tools": {}}` (`:237-239`). Notifications are ignored (`:235-236`). Unknown methods return `-32601` (`:252`).
- **Launch:** `server.py:973-974` sets `EXPLORER_URL` and `EXPLORER_CHAT`. `server.py:977-979` uses `--strict-mcp-config --allowedTools mcp__explorer`. The server strips the `mcp__explorer__` prefix (`server.py:1025`).
- **Tool errors:** any exception becomes `{"content":[{"type":"text","text":"error: …"}],"isError":true}` (`:247-248`). Success is `json.dumps(out)` in one text block (`:246`).
- **Timeouts:** `get` 60 s (`:161`), `post` 60 s (`:165`), and `steer`, `activations`, `compare`, `logprobs`, `top_firing` 900 s (`:204-212`).
- **Map actions:** `{"chat": CHAT, "action": …, **a}` (`:215`). `select_neuron` sends `select` and `highlight_neurons` sends `highlight` (`:216-219`).

## 3. Inputs -> outputs
`*` = required. Every tool that is not marked 900 s uses a 60 s timeout.

| Tool | Input | Endpoint | Timeout |
|---|---|---|---|
| overview | — | GET /api/agent/overview | 60 |
| neuron | layer*, n* | GET /api/agent/neuron | 60 |
| rollouts | layer*, n*, max_chars=500 | GET /api/unit (text cut) | 60 |
| cone | layer*, n*, k=2, limit=120 | GET /api/agent/cone | 60 |
| web_summary | layer*, n*, first=5, k=2 | GET /api/web/summary (top=20) | 60 |
| search_lens | query*, limit=40 | GET /api/search | 60 |
| top_units | layer*, limit=30 | GET /api/units (sorted by z) | 60 |
| set_name | layer*, n*, name* | POST /api/name, then /api/agent/ui `rename` | 60 |
| names | — | GET /api/agent/names | 60 |
| steer | prompt*, neurons[], dose=0.8, mode=add, max_new=150 | POST /api/agent/steer | 900 |
| activations | layer*, n*, text*, chat, neurons[], heads[] | POST /api/agent/activations | 900 |
| compare | neurons*, texts*, chat | POST /api/agent/compare | 900 |
| logprobs | text*, chat=true, prefix, k=15, targets[], neurons[], heads[] | POST /api/agent/logprobs | 900 |
| top_firing | text*, token, limit=30, layers[], chat | POST /api/agent/fires | 900 |
| camera | mode*, layer, n, closeness=0.22 | POST /api/agent/ui `camera` | 60 |
| annotate_neurons | items* | POST /api/agent/ui `annotate` | 60 |
| show_text | text*, token | POST /api/agent/ui `text` | 60 |
| show_search | query* | POST /api/agent/ui `search` | 60 |
| open_tab | tab* (examples/chat/details) | POST /api/agent/ui `tab` | 60 |
| select_neuron | layer*, n* | POST /api/agent/ui `select` | 60 |
| highlight_neurons | neurons* (sign +/-) | POST /api/agent/ui `highlight` | 60 |

Output is the endpoint's JSON, as text. The only write is `set_name`, which the server saves to the names file.

## 4. Porting notes for Gemma-270M / Coralboard NPU
- **Transfers unchanged:** the transport, dispatch, and error envelope. The read-only tools transfer as long as the new server returns the same JSON shapes.
- **Must change:** the `LN` ranges, "0-31" and "0-9215" (`:14`), become 0-17 and 0-2047 for Gemma-270M. The `heads` text (`:73-74`) describes Qwen's hybrid layers and needs rewriting from the Gemma config. The model name is hard-coded server-side (`server.py:602`).
- **Live tools need run-time steering:** `steer`, `activations`, `logprobs`, `compare`, and `top_firing` need steering pushes and per-layer activations at run time. Compiled Torq/IREE graphs fix weights ahead of time, so steering vectors would have to be graph inputs.
- **Runs on the host, not the board:** the `claude` CLI needs Node and network access to Anthropic. This file is stdlib-only, so it can run on a host beside the board with `EXPLORER_URL` pointing at the board. The UI-bridge tools need a browser on that host.

## 5. Surprises, bugs, or open questions
I checked these with a stub HTTP server and no model. The probe is at `/tmp/mcp-probe/probe.py`, outside the repo.
- **Crash on bad input:** a non-JSON stdin line exits the process (`:233`, rc 1). So does `tools/call` with no `params` (`:243`). One bad line ends every tool for that turn.
- **Opaque 500s:** `urllib` raises on 4xx/5xx and drops the body, so the model sees `error: HTTP Error 500: Internal Server Error`. The `steer` schema requires only `prompt` (`:68`). The server reads `b["layer"]` when `neurons` is absent (`server.py:667`), so that case gives this opaque 500.
- **Soft errors reported as success:** an unknown chat in `/api/agent/ui` returns `{"ok": False}` with HTTP 200 (`server.py:823`). `_live_off()` returns `{"error": …}` with HTTP 200 (`server.py:652-655`). The MCP passes both through with no `isError`.
- **`search_lens` drops `ready`** (`:193`). If the index is not built, the model gets `null, null, []`, which reads as "no match" (`server.py:299`).
- **`set_name` is not atomic** (`:198-199`). If the second POST fails, the name is saved but the tool reports an error.
- **Open:** Claude Code's MCP tool timeout is not in this repo, so the 900 s live timeouts may be cut short. I did not verify it.
- **Not a sandbox:** Claude also has Bash and `--add-dir ~` (`server.py:980`), so it can call the API directly with curl.
