# 16-agent-chat-loop

Code reader report on lyramakesmusic/neuron-explorer @ f79b219. Analysis only; no code copied.

**1. What this slice does**
This is the backend of the explorer's Claude chat: file uploads, attachment encoding, one `claude -p` subprocess per turn streamed back as NDJSON, rewind by truncating the CLI's session JSONL, and the startup preload thread. Each turn runs the CLI in `explorer/cache/agent` with the explorer mounted as an MCP server.

**2. Key mechanics**
- Spawn (`server.py:976-982`): `claude -p --input-format stream-json --output-format stream-json --verbose --include-partial-messages --system-prompt-file <prompt> --mcp-config <inline JSON> --strict-mcp-config --tools Read Glob Grep Write Edit Bash --allowedTools mcp__explorer <same tools> --add-dir $HOME --setting-sources project`. `--model $EXPLORER_CLAUDE_MODEL` is appended only if set (`:981-982`), otherwise the CLI default applies.
- MCP server: `sys.executable explorer/agent_mcp.py` with `EXPLORER_URL=http://127.0.0.1:{PORT}` and `EXPLORER_CHAT=<cid>` (`:973-974`).
- Resume: chat id is the session UUID (`:966`). `--resume` if `~/.claude/projects/*/<id>.jsonl` exists, else `--session-id` (`:988`). A prior turn is killed first: SIGTERM to the process group, wait 5 s, then SIGKILL (`:891-904`).
- Stream mapping (`:1013-1038`): `text_delta` → `text`; `tool_use` → `tool` (strips `mcp__explorer__`); `tool_result` → output cut at 1500 chars; `result` → `done` with `total_cost_usd`; non-zero exit → `error` with the last 600 stderr chars.
- Rewind (`:925-956`): finds the latest person-typed message (`_user_text`, `:912-922`) that matches `text`, copies the file to `.bak-<unix-seconds>`, truncates before it, and deletes the file if no person message is left. The client aborts the stream first (`app.js:995`), then calls rewind (`app.js:998`).
- Security: `local_only` middleware (`server.py:43-51`) blocks foreign Host/Origin and cross-site fetches, which covers browser CSRF and DNS rebinding. Inside the agent, Bash is auto-allowed (bare name in `--allowedTools`, `:979`) with no path or network limits, so `curl` works. `--add-dir ~` only widens the file tools. "Don't print this file" is a prompt rule (`agent_prompt.md:157`), not enforced.

**3. Inputs -> outputs**
- `POST /api/agent/upload` `{name, type, data(b64)}` → `cache/agent/uploads/<id>-<name>` + `<id>.json`. The 25 MB cap is checked after decode (`:839`).
- `POST /api/agent/chat` `{chat?, message, layer?, n?, attachments?}` → NDJSON `chat | text | tool | tool_result | ui | error | done`.
- `POST /api/agent/rewind` `{chat, text}` → `{ok, cut}`.
- `GET /api/status` → `{status}` from `runtime.py` (`idle`, `loading`, progress strings, `ready`, `no model`).
- Writes `cache/agent/system_prompt.md` every turn. Reads `~/.claude/projects/*/<id>.jsonl`.

**4. Porting notes for Gemma-270M / Coralboard NPU**
- Transfers: `local_only` middleware, the NDJSON relay and event mapping, base64 upload → content block, and the preload-thread pattern.
- Can't run on-device: the agent is a hosted Claude CLI and is not feasible on 2 GB RAM. Run it on a host and treat the board as an HTTP target, or drop it. Steering and activations need PyTorch forward hooks (`runtime.py`), so they stay host-side. The board can only run ahead-of-time compiled graphs.
- Don't copy the Bash grant. `claude --help` lists `--restricted`, which confines file tools and drops user/project settings. Bash stays unless `--tools` omits it.
- Rewind depends on Claude's JSONL format. Keep your own transcript list and truncate that instead.

**5. Surprises, bugs, or open questions**
- **Verified hang:** a stdout line over 16 MiB (`limit=1<<24`, `:992`) raises `ValueError` in `pump` (`:997-1001`). No `end` is queued, so `events()` blocks forever on `q.get()`. I reproduced this with the same asyncio pattern.
- **Rewind race:** rewind doesn't call `_stop_turn`. The killed CLI may still append to the file being rewritten.
- **Prompt edits:** `claude --help` says `--system-prompt-snapshot` defaults to on for `-p`, recording the prompt on a chat's first request and reusing it on resume. Edits to `agent_prompt.md` may not reach existing chats (not verified by running a turn).
- **CLAUDE.md:** the comment at `:584` says no CLAUDE.md applies. The `--bare` help says `--add-dir` supplies CLAUDE.md dirs, which suggests `~/CLAUDE.md` and possibly `~/.claude/CLAUDE.md` load. `--setting-sources project` also skips the user's settings, so their hooks and deny rules don't apply to this agent.
- Latent deadlocks: stderr is read only at EOF (`:1000`), and stdin is written before `pump` starts (`:993-997`).
- A missing `claude` binary raises before any event, so the client gets no error event.
- Image types outside the four listed (e.g. SVG, BMP) fall through to UTF-8 text (`:861`).
- Uploads are never cleaned. Same-second rewinds overwrite each other's `.bak`.
- Preload (`:1057-1071`) has no try/except. A failure leaves the last progress string on `/api/status`.
