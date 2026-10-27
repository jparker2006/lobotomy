# Mockups (prototypes, not the product)

Canvas prototypes behind [docs/ui/design.md](../../docs/ui/design.md). They draw real recorded sessions; nothing is faked.

- **Data** (gitignored, in `data/`): run the CPU stand-in (`python server/server.py --backend cpu`), then
  `python web/mockups/tools/record.py "Describe the ocean at night in three sentences." web/mockups/data/ocean.bin`.
  This writes the raw T2 frame stream plus `calibration.f32` and `meta.json`.
- **Fonts** (gitignored, in `data/fonts/`): JetBrains Mono, Geist Mono, IBM Plex Mono, Space Mono, Martian Mono, VT323 and Share Tech Mono from Google Fonts (latin woff2), and Departure Mono from its GitHub release, listed in `data/fonts/fonts.css`. All OFL.
- **Serve**: `python -m http.server 8100` from `web/mockups/`, then open `q1/…` to `q9/…`. URL parameters set the session (`s`), token (`t`), shape, ramp, palette (`pal`), layout, `astro=1` and camera (`eye`, `tgt`, `fov`, `cxy`).
- **Render** (no GPU needed): headed Chromium on a virtual screen with `--use-angle=gl --ignore-gpu-blocklist`, screenshot when `window.__ready` is true. Animated pages expose `window.renderAt(t)` for frame-by-frame capture; frames go to MP4/GIF through ffmpeg.
- `lib/`: the session loader (`data.js`), the camera (`proj.js`), shapes (`shapes.js`), and the text-mode renderer with the rarity ladder and crowding correction (`ascii.js`).
