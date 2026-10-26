# lobotomy UI design (v0.0.2)

*Work in progress: the v0.0.2 UI grill with Jake, 2026-10-26. Each decision has a rendered mockup in [mockups/](mockups/). Mockups are drawn from a real recorded CPU session (Qwen3-0.6B, "Describe the ocean at night in three sentences.", token 52 " glow"), never from made-up data. Prototype code is in [web/mockups/](../../web/mockups/).*

## Decisions

| # | Question | Decision | Mockup |
|---|---|---|---|
| Q1 | Vibe | **ASCII render**: the 3D brain drawn through one character grid, with the UI on the same grid. Add B's glow (bloom on real spikes), A's bright amber highlights, and C's calm (a dim lattice, hairline UI, few mid-tones). | [q1-vibe-sheet.png](mockups/q1-vibe-sheet.png) |
| Q2 | Shape | **Deck**: 28 layer sheets side by side, L0 to L27. Within a layer, neurons sit by index for now (64 × 48); v0.1.0 swaps in the co-activation layout (R9). | [q2-shape-sheet.png](mockups/q2-shape-sheet.png) |
| Q3 | Glyphs | **Dots** `· ∙ • ● ◉`: bigger and brighter = stronger, readable without a legend (Jake: ASCII symbols look like they mean something). The ASCII feel stays in the grid, edge strokes and text UI. | [q3-glyph-sheet.png](mockups/q3-glyph-sheet.png) |
| — | Honesty fix | **Rarity ladder with a crowding correction.** A dot shows how rare the firing is for that neuron (top 10 / 3 / 1 / 0.3 / 0.1%). A character cell that holds k neurons uses u = F(max)^k, so a crowded far cell is no likelier to light than one neuron. Without this, far layers looked ablaze just from packing. | [q4-trails.gif](mockups/q4-trails.gif) |
| Q4 | Trails | **Half-life 2 tokens** (0.33 s at 6 tok/s), measured in tokens. Jake will tune it live, so it goes on a tuning knob. | [q4-trails.gif](mockups/q4-trails.gif) |
| — | Resolution fix | **Two grids.** The map uses a fine dot grid (5 × 8 px, about one neuron per cell at the overview); the UI uses the 9 × 16 text grid. With one coarse grid, each cell pooled ~100 neurons and the map went dark. | [q5-cinema.png](mockups/q5-cinema.png) |
| Q5 | Screen layout | **Cinema by default** (map full-bleed, answer as large subtitles, prompt line, token tape) **plus the Console panels on a toggle key** (transcript, top-1% count per layer, rarest neurons now). | [q5-layout-sheet.png](mockups/q5-layout-sheet.png) |
| Q6 | Colour and type | **Amber (+) / cyan (−)** (default; Jake didn't object). **JetBrains Mono** for all text, **Departure Mono** for the big counters. All fonts are OFL and bundled (the board serves the page offline). | [q6-palette-sheet.png](mockups/q6-palette-sheet.png), [q6-type.png](mockups/q6-type.png) |
| Q7 | Motion | *open* | [q7-motion.gif](mockups/q7-motion.gif) |
