# Mood board

Jake's references for the v0.0.2 UI (pasted 2026-10-27): three X posts, fetched with `api.fxtwitter.com`, and two *Project Hail Mary* stills. **Links and notes only.** The repo is public, so third-party images are never committed; local copies in this folder are gitignored.

| Reference | What it is | What we take |
|---|---|---|
| [@praveenisomer, "ASCII Web Section"](https://x.com/praveenisomer/status/2114680944196264294) (30 s video) | A web page where a globe and small 3D objects (shell, spring, figure) are rendered in periwinkle ASCII on white, next to serif headlines. | 3D form drawn by character density; one ink; editorial calm and white space around a dense object. |
| [@PranavOriginals, Voyager 1](https://x.com/PranavOriginals/status/2114390231558639635) (two images) | Voyager 1 twice: a white line drawing on black with a tiny exact caption ("September 5, 1977, 12:56:00 UTC"), and a 1-bit dithered pixel illustration. | Black negative space; a starfield of dots; metadata as typography (a precise timestamp under the hero); tone from dither density. |
| [@shridharathi, "why isn't agent thinking visual?"](https://x.com/shridharathi/status/2113255894423158914) (24 s video) | A chat UI where a halftone, dot-stippled 3D object changes with each thought step, beside a growing list of thoughts and a "Thinking 7.1s" timer. | Thinking made visible step by step; a running elapsed time; stipple dots as the medium. (Their floating particles are decoration; ours stay data.) |
| *Project Hail Mary* (2026 film, Amazon MGM Studios): the astrophage swarm (still pasted by Jake) | An astronaut on the hull, inside a swarm of thousands of glowing red sparks: pin-sharp far ones, soft bokeh near ones, and a faint haze behind the dense parts. Everything is lit by the swarm, in one ink. | Jake: "mono amber, like astrophage". One-ink sparks; a wide, faint **haze from the real top-1% dots only**, so dense activity glows like a nebula; depth from haze on the lattice, never from data-dot size. Our render of this: [q6c-astrophage-sheet](../mockups/q6c-astrophage-sheet.png). |
| *Project Hail Mary*: the green planet (still pasted by Jake) | A dark silhouette of the astronaut and ship against a luminous, swirling green-and-orange cloudscape. | Silhouette against glow: dark text on a soft scrim over the swarm. Turbulent flow is what v0.1.0's co-activation layout might look like when related neurons cluster. |

## Shared principles

1. **Small marks make form.** Dots, characters or dither, and density does the shading. This matches the dot-matrix deck and the lattice.
2. **One ink.** The X posts and the astrophage swarm each use a single colour on a plain ground. This prompted the palette re-check ([q6b-ink-sheet](../mockups/q6b-ink-sheet.png)) and the switch to mono amber.
3. **Negative space and tiny exact captions.** Lots of black, and small precise metadata: `L08 · N0908`, `token 052/256`, and an end-card caption with the real recording time, Voyager-style.
4. **Show the thinking as it happens.** Text accumulates with an elapsed timer; our subtitles, tape and `n/256 · tok/s` play that role.
5. **Calm around the hero.** One dense object, quiet type around it: the Cinema layout.
6. **A swarm that glows where it's busy** (astrophage). Sparks of one ink, and a soft haze that builds only where real rare activity clusters.

## Also in the air

- Lyra's neuron-explorer: the deck, additive glow, layers left to right ([docs/lyra-reports](../../lyra-reports/)).
- Eames, *Powers of Ten* (1977): storyboard C's pull-back from one neuron.
- Demoscene ASCII tunnels: storyboard B's fly-through.
- Datashader-style aggregation: the reason for the crowding correction (a crowded cell must not look busier than one neuron).
