# brainbox

A live neuron explorer for Qwen3-0.6B running on a 1-watt NPU.

The model runs offline on a Synaptics Coralboard (SL2619, 1 TOPS Torq NPU). Physical knobs push individual neurons while it talks, and a 3D map in your browser shows every neuron lighting up as each token is generated.

Inspired by [lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer). No code is copied from it.

**Status:** day 0. See [PLAN.md](PLAN.md).

## How it fits together

- **GPU (once, a few hours):** pushes every one of Qwen3-0.6B's 86,016 MLP neurons, compares against random-direction controls, and writes an atlas of what each real neuron does.
- **Coralboard (always on):** runs the steered model on the NPU and streams tokens plus live activations over USB. Knobs push neurons from the atlas.
- **Laptop browser:** the 3D map.

Rust everywhere except model export and compile, where the Torq toolchain needs Python.
