# lobotomy UI design (v0.0.2)

*Work in progress: the v0.0.2 UI grill with Jake, 2026-10-26. Each decision has a rendered mockup in [mockups/](mockups/). Mockups are drawn from a real recorded CPU session (Qwen3-0.6B, "Describe the ocean at night in three sentences.", token 52 " glow"), never from made-up data. Prototype code is in [web/mockups/](../../web/mockups/).*

## Decisions

| # | Question | Decision | Mockup |
|---|---|---|---|
| Q1 | Vibe | **ASCII render**: the 3D brain drawn through one character grid, with the UI on the same grid. Add B's glow (bloom on real spikes), A's bright amber highlights, and C's calm (a dim lattice, hairline UI, few mid-tones). | [q1-vibe-sheet.png](mockups/q1-vibe-sheet.png) |
| Q2 | Shape | **Deck**: 28 layer sheets side by side, L0 to L27. Within a layer, neurons sit by index for now (64 × 48); v0.1.0 swaps in the co-activation layout (R9). | [q2-shape-sheet.png](mockups/q2-shape-sheet.png) |
| Q3 | Glyphs | *open* | [q3-glyph-sheet.png](mockups/q3-glyph-sheet.png) |
