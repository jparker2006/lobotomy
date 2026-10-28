# U5 sandbox renders + the ticket's check (#23)

Cloud sandbox (no GPU): Chromium, SwiftShader WebGPU, 1440 × 900 CSS px at DPR 2, CPU stand-in (Qwen3-0.6B, real server),
"Describe the ocean at night in three sentences." (91 tokens). The deck is composited from `deck.snapshot()` (this sandbox
never composites WebGPU canvases); the UI is the real DOM.

- `lock-words.png`: `/L8 908` while pinned at " sea": the readout prints the answer, each word lit by N908's value on it
  (amber fill = +, outline = −, current word underlined); `+ FIRES MOST ON` / `− FIRES AGAINST` (top 3% only, (Q) = in the question).
- `prompt-subtitle-no-markup.png`: pinned at `<think>` (token 18): the subtitles show only the question; template tape bars dimmed.
- `hover-overview.png`: pinned at " sea", hover on L8 N0908 (q8-hover). `lock-console.png`: click to lock, then `c`.

**The check** (`check.mjs` + `expect.py`; sandbox tooling, `playwright-core` installed outside the repo): records nothing
itself; it loads a recording made with `web/mockups/tools/record.py` against `server.py --backend cpu`, feeds it through
`app.beginSession` / `app.ingest`, and compares the readout with values `expect.py` computes from the same file with numpy
(f16 activation / p99). It checks: no template markup in the UI text or tooltips at any of the 91 tokens (Cinema + Console);
`/L8 908` locks while pinned at `<think>` (also typed as `/L8 908`); Esc order (command line, lock, pin); NOW, WORD, the
lit words and both fires lists equal the session's values, with " ocean", " sea", " water" under `+ FIRES MOST ON`;
readout, tape and subtitle clicks pin their token, `space` resumes; bad input says so; hover picks L8 N908; click locks on. Also passes on `?backend=webgl2`.

```sh
python server/server.py --backend cpu            # venv with torch + transformers
python web/mockups/tools/record.py "Describe the ocean at night in three sentences." /tmp/rec/ocean.bin --url http://127.0.0.1:8000
DISPLAY=:99 PY=<venv python> REC=/tmp/rec SHOTS=/tmp/rec node check.mjs   # from a folder with playwright-core installed
```

SwiftShader cannot keep up with the rAF loop and GPU readbacks (`pick`, `snapshot`) then never return, so after ingesting the
check freezes the clock and steps frames with `app.renderAt`. A real GPU does not need this.
