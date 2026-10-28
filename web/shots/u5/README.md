# U5 sandbox renders + the ticket's check (#23)

Cloud sandbox (no GPU): Chromium, SwiftShader WebGPU, 1440 × 900 CSS px at DPR 2, CPU stand-in (Qwen3-0.6B, real server),
"Describe the ocean at night in three sentences." (91 tokens). The deck is composited from `deck.snapshot()` (this sandbox
never composites WebGPU canvases); the UI is the real DOM.

- `hover-overview.png`: word replay pinned at " sea" (tape click), hover on L8 N0908: `+4.7× ITS P99 · TOP 0.1%` (q8-hover).
- `lock-slash-cmd.png`: `/L8 908` + Enter, then " sea" pinned: face-on L8, reticle, leader, docked readout (q8-lock).
- `lock-click.png`: the same lock reached by clicking the hovered dot. `lock-console.png`: `c` while locked.

**The check** (`check.mjs` + `expect.py`; sandbox tooling, `playwright-core` installed outside the repo): records nothing
itself; it loads a recording made with `web/mockups/tools/record.py` against `server.py --backend cpu`, feeds it through
`app.beginSession` / `app.ingest`, and compares the readout with values `expect.py` computes from the same file with numpy
(f16 activation / p99): NOW, WORD, FIRES MOST ON (token, ×p99) at the live end and pinned at " sea"; tape and subtitle
clicks pin their token, `space` resumes, invalid `/` input says so, `Esc` cancels / unlocks, hover picks L8 N908, click
locks on. Also passes on `?backend=webgl2`.

```sh
python server/server.py --backend cpu            # venv with torch + transformers
python web/mockups/tools/record.py "Describe the ocean at night in three sentences." /tmp/rec/ocean.bin --url http://127.0.0.1:8000
DISPLAY=:99 PY=<venv python> REC=/tmp/rec SHOTS=/tmp/rec node check.mjs   # from a folder with playwright-core installed
```

SwiftShader cannot keep up with the rAF loop and GPU readbacks (`pick`, `snapshot`) then never return, so after ingesting the
check freezes the clock and steps frames with `app.renderAt`. A real GPU does not need this.
