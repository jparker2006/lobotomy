# 19-frontend-3d

Code reader report on lyramakesmusic/neuron-explorer @ f79b219. Analysis only; no code copied.

## 1. What this slice does
Draws the model as a three.js r169 WebGL scene. Layers run left to right, each with a neuron "sheet" of points, a token-mixer slab, and link and web overlays. It also handles the orbit camera, zoom-fade, face-on layer view, and CPU ray picking. The embeddings view is a separate Canvas 2D scatter (`scatter.js`). There is no bundler; three.js loads from jsDelivr (`index.html:9-14`).

## 2. Key mechanics
- **Constants:** `SHEET=4.2, GAP=1.9, SHEET_Y0=0.7, ATTN_DX=-0.62, MLP_DX=0.18` (`map3d.js:9`). Layer l's sheet sits at x = 1.9l + 0.18 (`:78`).
- **Grid:** `gridSide=ceil(sqrt(N))`. Neuron y = 0.7 + j/(g−1)·4.2, z = (i/(g−1) − 0.5)·4.2 (`:73-78`). Qwen's 9216 gives a full 96×96; Gemma's 2048 gives 46×46 with the top row empty.
- **Points:** one `THREE.Points` holds all L×N vertices, so one draw call (`:82-93`). The `RawShaderMaterial` is additive with depthWrite off (`:22-47`). Size is `sizePx·min(DPR,2)·clamp(sqrt(ref/−mv.z),0.6,maxScale)` (`:24,39,41`). Discs use `discard` at r>0.5 and `1−smoothstep(0.38,0.5,r)` (`:43-44`).
- **Slabs:** BoxGeometry, thickness 0.3 for full attention and 0.1 for linear, opacity ×BRIGHT (default 2.5, `:18`, `:106-108`). Head stripes at `:117-120`.
- **Camera:** fov 32, near 0.02, far 2000 (`:61`). OrbitControls with zoomToCursor, min 1.6, max 400 (`:65-70`). Home distance `r/sin(min(vHalf,hHalf))·0.82` (`:307`). `setViewOffset` centres the target in the gap beside the panels (`:315`).
- **Zoom fade:** smoothstep on log-distance between 0.7·home and 0.07·home (`:406-408`).
- **Face-on:** `viewLayer` distance is 9.15 (`:451`). `enterLayer` is 8.2 from the −x side, with rotation off (`:378-380`).
- **Picking:** `Raycaster`, `Points.threshold=0.05` in world units (`:322`). That is about 0.54 of a Gemma cell at any zoom. Each pointermove scans every vertex, rAF-throttled (`:345-356`).
- **Links:** direct links are 1px `LineBasicMaterial` (`:150`). The web is 2px `LineSegments2` (`:161-163`). Web alpha is `0.5·(0.6+0.4|z|/zhi)·max(0.35,0.85^(hop−1))·(through?1:0.08)` (`:190`).
- **Rendering:** on demand. `render()` is rAF-coalesced, with no loop (`:287-295`).

## 3. Inputs → outputs
- **In:** `arch` (`n_layers, inter, layer_types, attn_heads, linear_v_heads`; built at `server.py:30-33`). `setLayout(Int16Array)`, `setUnits(layer, [{kind,n,z,name}])` (`map3d.js:465`), `setWeb(Int16Array, 6 per link)` (`:521`), `setConnections`, `setMarks`, `setNotes`, camera commands.
- **Out:** `onHover(p,x,y)`, `onPick(layer,n)`, `onLayerMode` (`app.js:24-35`).
- **Scatter:** typed arrays `x, y, kind, neuron, prompt, dose`. Callbacks `onPick(i)`, `onHover(i)`.

## 4. Porting notes
- **Transfers unchanged:** the single points buffer, sheet layout, glow shader, zoom fade and face-on math are generic in L and N. Gemma's 18×2048 is 36,864 points, which is trivial.
- **Must change:**
  - `server.py:30-33` reads Qwen-only config keys (`linear_num_key_heads`). This will likely raise on a Gemma config.
  - `map3d.js:106` treats every non-`full_attention` layer as linear. Gemma's sliding layers would render as linear, and `app.js:36` would print "undefined k / undefined v heads".
- **Dependency:** three.js comes from a CDN (`index.html:11-12`). Vendor it for offline use.
- **Can't run on the board:** the map and picking are host-side analysis. The 2 GB board only needs inference. Steering must be compiled into the ONNX graph, because NPU graphs have no per-token hooks.

## 5. Surprises, bugs, open questions
- Point colours are written raw. `RawShaderMaterial` has no colour-space chunk, so they likely don't match the sRGB-encoded lines and meshes. The palette was tuned by eye, so this may be intentional.
- `flyTo` and `viewLayer` (`:433-460`) don't cancel each other, so overlapping animations fight over `controls.target`.
- Dead or misleading code: `focusLayer(){}` (`:507`) is never called, `PAL.link` (`:14`) is unused, and `fitPerPx` actually stores `homeDist` (`:303,310`).
- `layerVisibility` (`:368-372`) returns `1−amt` for front layers. The shader (`:37`) uses `×(1−0.6·amt)`. Picking and rendering disagree slightly.
- `scatter.js`: `color(i)` builds strings per point per frame (`:31-37`). `nearest()` is O(n) per mouse move and recomputes layout for every point (`:26-30,68-75`).
- Picking is O(L·N), about 295k vertices for Qwen. A per-sheet ray-plane hit would be O(L).
