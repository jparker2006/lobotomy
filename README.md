# lobotomy

Watch the neurons of a language model light up while it talks, live on a 1-watt chip.

Qwen3-0.6B runs offline on a Synaptics Coralboard (SL2619, 1 TOPS Torq NPU). The board also serves a web page, and your browser shows all 86,016 of the model's MLP neurons lighting up as each word is generated. Later, you'll be able to push single neurons and watch the answer bend.

Inspired by [lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer). No code is copied from it.

**Status:** v0.0.1 in progress. See [PLAN.md](PLAN.md).

## How it fits together
- **Coralboard (always on):** runs the model on the NPU and serves the page plus live activations over USB. Python, on Synaptics' Torq runtime.
- **Laptop browser:** draws the map. TypeScript + three.js.
- **Rented machines:** an x86 CPU pod compiles the model for the NPU. Later, a GPU tests every neuron once against random controls and writes an atlas of what the real ones do.

## Run it (v0.0.1, CPU stand-in for the board)
Needs Python 3.12 with numpy, and torch plus transformers for the `cpu` backend. Node is only for building the page.
```
(cd web && npm install && npm run build) && python server/server.py --backend cpu   # open http://localhost:8000
```
`--backend fake` needs no model. Tests: `python -m unittest discover server`.
