# Stats fixture (U3, #21)

A real CPU stand-in recording (Qwen3-0.6B, transformers fp32, prompt "Say hi.", 2026-10-28), cut to stay under 2 MB:
positions 0 (the attention sink), 3–5 (`Say`, ` hi`, `.`) and 15–20 (`Hello! How can I assist`) plus the done frame, whose
header still describes the whole answer (15 prompt + 11 generated tokens, 29.13 tok/s). Frames are byte-for-byte as served.

- `session.bin`: T2 frames; `calibration.f32` + `meta.json`: `/api/calibration` and `/api/meta` of the same server.
- `expected.json`: numpy's numbers for every token (top-1% threshold, counter, per-layer counts, 6 rarest) and the tape.
- Regenerate: `python server/server.py --backend cpu`, then
  `python web/mockups/tools/record.py "Say hi." /tmp/rec/hi.bin`,
  `python web/test/fixtures/stats_fixture.py trim /tmp/rec/hi.bin --keep 0,3-5,15-20`,
  `python web/test/fixtures/stats_fixture.py expected` (numpy only). `npm test` (`test/stats.test.ts`) compares.
