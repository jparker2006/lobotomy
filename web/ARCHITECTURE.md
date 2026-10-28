# web/: module map and contracts (v0.0.2)

The page is TypeScript + three.js r186 (`three/webgpu` + TSL), no bundler: `tsc` compiles `src/` to `dist/`, an import map in
`index.html` points `three/webgpu` and `three/tsl` at `dist/vendor/`, and `tools/build.mjs` copies pages, CSS, assets and
three's build files. The server (`server/server.py`) serves `web/dist`. Specs: [#18](https://github.com/jparker2006/lobotomy/issues/18),
look and rules: [docs/ui/design.md](../docs/ui/design.md).

**U3, U4 and U5 run in parallel.** Stay inside your folder. Nobody but U2 edits `main.ts`, `index.html`, `package.json`,
`tsconfig*.json`, `app/` (except U4: `app/camera.ts` internals), `proto/`, `data/` (except U3: `data/stats.ts`) or `render/`.
If you need something from another module, read the store or call the app; if that is not enough, say so in your report.

## Modules and owners

| Path | What | Owner |
|---|---|---|
| `src/main.ts` | bootstrap only: `createApp()`, then `initUI`, `initMotion`, `initInspect`, `initReplay`, `app.start()` | U2 |
| `src/app/state.ts` | the store: every state field, typed events | U2 |
| `src/app/app.ts` | the `App` context: protocol → session → rarity → deck, token pacing, pin, frame loop | U2 |
| `src/app/params.ts` | the one place `state` maps onto `deck.params` | U2 |
| `src/app/clock.ts` | time source (real time, manual/deterministic mode) | U2 |
| `src/app/camera.ts` | camera rig, **frozen API** below | U2 builds; **U4 owns the internals** |
| `src/app/provenance.ts` | backend labels (honesty rules) | U2 |
| `src/app/css.ts`, `app.css` | `loadCSS()`, page base and palette variables | U2 |
| `src/proto/` | `frames.ts` (T2 parser), `client.ts` (HTTP + long-poll), `session.ts` (frames of the current answer) | U2 |
| `src/data/` | `f16.ts`, `rarity.ts` (baked table, live F, `levelOf`) | U2; **U3 adds `data/stats.ts`** |
| `src/render/` | the WebGPU deck ([README](src/render/README.md)) | U1 built, U2 extended; **frozen**, U3–U6 only call it |
| `src/ui/` | Cinema UI: HUD, legend, subtitles, prompt line, tape, Console panels (`c`) | U3 (U2 left a minimal HUD there) |
| `src/motion/` | orbit / zoom / pan, idle drift, sweep pace, the tuning console (`~`) with the live fps / p95 readout, settings persistence (`localStorage`) | U4 |
| `src/inspect/` | hover card, lock-on, `/` jump, word replay clicks | U5 |
| `src/replay/` | record / replay files, deterministic clips | U6 |
| `src/plain/` + `plain.html` | T2's plain grid, kept as a debug view at `/plain.html` | U2 |
| `bench/` | render benchmark | U1 |
| `test/<module>.test.ts` | each ticket its own file (`render`: U1, `rarity` + `proto`: U2) | per ticket |

## The app context

Every module exports `init<Module>(app: App)`. `App` (`src/app/app.ts`):

| Member | Use |
|---|---|
| `store` | state + events (below) |
| `deck` | the renderer: `pick(x, y)`, `project(xyz)`, `layerLabel(l)`, `size`, `snapshot(t)`, `setGlow()` … ([render/README](src/render/README.md)). Don't write `deck.params` directly: change `store.settings` / `store.locked`; `app/params.ts` maps them |
| `camera` | the rig (frozen API below) |
| `clock` | `now()`, `frameTime`, `setManual(t)` |
| `client` | the protocol client (`meta`, `calibration`, `generate`, `abort`) |
| `canvas`, `ui` | the deck canvas and the overlay root `#ui` (add your DOM inside it; `#ui` has `pointer-events: none`, give interactive elements the class `interactive`) |
| `rarity` | the live pooled F (`rarity.table()`, `data/rarity.ts`) |
| `ask(prompt)` | asks the server (a newer question cancels the old one) |
| `beginSession(prompt, id?)`, `ingest(frames, t?)` | feed a session yourself (replay): same path as live frames |
| `pin(k)`, `unpin()` | word replay: show the map at session token k (CPU trail state `glowAt`, drawn via `deck.setGlow`), back to live |
| `renderAt(t)` | deterministic: freeze the clock at t and draw one frame (clips) |

## State (`app/state.ts`)

| Field | Meaning | Written by |
|---|---|---|
| `meta` | `/api/meta`: layers, neurons, backend, model, max_tokens | app |
| `backend` | the real backend name from `/api/meta` (`cpu`, `fake`, later `torq` / `npu`) | app; replay sets `replay` instead |
| `calibration` | per-neuron p99 (Float32Array, 86,016) | app |
| `session` | `Session`: `tokens[]` (`i`, `kind`, `text`, `r` = activation/p99 Float32Array, arrival `t`, `top1`, `top1ByLayer`), `done`, `text(k)` | app |
| `liveIndex` | session token on the deck now (-1 none). Lags arrival by the pacing | app |
| `pinnedIndex` | word replay token or `null` (use `app.pin` / `app.unpin`) | U3 / U5 via app |
| `hover` | `{layer, neuron}` under the pointer or null | U5 |
| `locked` | lock-on `{layer, neuron}` or null; while set only that layer is drawn | U5 |
| `consoleOpen` | Console panels shown | U3 |
| `settings` | every knob: `halfLifeTokens`, `bloomStrength`, `bloomScale`, `hazeStrength`, `cellW`, `cellH`, `sweepOn`, `driftOn`, `thresholds`, `inkHue` (use `store.setSettings(patch)`; U4 persists to localStorage) | U4 (tuning console), anyone may read |
| `tokS` | measured tokens/s (arrivals; the server's figure after done) | app |
| `elapsedSec` | seconds since the question | app |
| `replay` | null, or `{backend, recordedAt, power?}` while replaying; `power` = measured R12 record only | U6 |
| `status`, `error` | `connecting` / `ready` / `asking` / `generating` / `done` / `error` | app |

Events: `store.on(name, fn)` returns an unsubscribe function.
- `change:<field>` (new value) for every field, and `change` (list of changed fields) once per `set()`.
- `session` (a new session began), `arrive` `{index}` (a frame is in the session), `token` `{index}` (a token reached the deck),
  `done` (done header), `rarity` (F changed), `frame` `{t, dt}` (after every drawn frame: move DOM overlays that track the deck here).

## Shared rules for parallel tickets (set by the orchestrator)

- **Keys.** `c` = Console (U3), `~` = tuning console (U4), `/`, `Esc`, `space` = inspect (U5). Every key handler ignores events
  whose target (or `document.activeElement`) is an `input`, `textarea` or `[contenteditable]`, so typing a question never toggles
  panels. Register handlers in your own module only.
- **Clickable words.** U3 renders every word of the subtitles, the transcript and every tape bar with `data-token="<k>"` (the
  session token index) and the class `interactive`. U5 owns what a click does (pin that token, `space` back to live) by event
  delegation on `app.ui`; U3 adds no click behaviour to those elements.
- **Overlays.** Each module adds its own elements inside `app.ui` with its class prefix (`.ui-`, `.mot-`, `.insp-`, `.rep-`) and
  moves deck-tracking overlays on the `frame` event.

## Frozen APIs

**Camera rig** (`app/camera.ts`; U4 may change everything inside, not these signatures):
`pose` (current `{eye, target, fov, center}`), `overview(ms?)`, `faceLayer(l, ms?)`, `flyTo(pose, ms)` (promises resolve on
arrival or interruption), `onUserInput()` (stop tweens and drift now), `setFrame({l, r}, ms?, owner?)` (side panels' share of the width per owner, combined widest-per-side: `overview()` / `faceLayer()` then frame the deck between them; U3's Console uses owner `console`, U5's lock-on readout `inspect`), plus `update(t)` which the app loop calls each frame
before `deck.setView(pose)`. U5 calls `faceLayer` / `overview` on lock / unlock; U4 calls `onUserInput` from its input handlers
and reads `store.settings.driftOn`.

**Render** (`render/`): `createDeck`, `pushToken`, `setRarityTable`, `frame(t)` (deterministic in t), `setView` / `getView`,
`project`, `layerLabel`, `pick` (the neuron that set the dot), `snapshot`, `setGlow(state | null)`, `params.visibleLayers`,
and `render/cpu.ts` `glowAt(tokens, k, halfLife)`.

**Clock**: `now()`, `frameTime`, `tick(rafMs)`, `setManual(t | null)`, `manual`.

## Honesty rules (all modules)

Every per-element visual maps to a data field in design.md's channel table. Show the backend with `app/provenance.ts`
(`CPU STAND-IN`, `FAKE · SYNTHETIC DATA`, `REPLAY · …`). Never print "NPU", "1 W" or any power figure unless the data came from
the board **and** `state.replay.power` holds a measured R12 record. Synthetic data is labelled SYNTHETIC.

## CSS, assets, tests

- **CSS:** put `<module>.css` in your folder and load it from your init: `loadCSS(new URL("./ui.css", import.meta.url))`
  (`app/css.ts`). The build copies every `src/**/*.css` next to its JS. Prefix your classes (`.ui-…`, `.insp-…`) and use the
  palette variables from `app/app.css` (`--ink`, `--hot`, `--ui`, `--ui-bright`, `--font`, `--font-counter` …).
- **Fonts:** `assets/fonts/fonts.css` (linked by `index.html`): `"JetBrains Mono"` 300/400/700, `"Departure Mono"` 400. OFL licenses
  sit next to the files. Other assets go under `assets/<module>/`; they are copied to `dist/assets/`.
- **Tests:** `test/<module>.test.ts`, Node's `node:test` on the compiled output; `npm test` compiles every `test/**/*.ts`.
  The project has no `@types/node`: `test/node-shim.d.ts` declares what tests use (add members there if you need more).
- **No new dependencies** (only `three`, `@types/three`, `typescript`). Browser checks use tools outside the repo.

## Commands

```sh
cd web && npm install
npm run build          # dist/: the page (/), the plain debug view (/plain.html), the bench (/bench/)
npm test               # unit tests
python server/server.py --backend fake     # from the repo root: no model needed (SYNTHETIC data)
python server/server.py --backend cpu      # the CPU stand-in (needs a venv with torch + transformers)
```
