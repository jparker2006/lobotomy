# lobotomy

Read [docs/CONTEXT.md](docs/CONTEXT.md) first. It has the decisions, board facts, toolchain limits, steering design and current status. The plan, v0.0.1 spec and tickets are in [PLAN.md](PLAN.md).

- Keep replies short. Jake skims long answers.
- Lyra's neuron-explorer has no license. Use [docs/lyra-reports/](docs/lyra-reports/) as reference, and never copy its code.
- Synaptics' torq-* repos are Apache 2.0. Adapting their code is fine; keep their license notices.
- Python on the board, TypeScript + three.js in the browser. No Rust for now.
- The Torq compiler only runs on x86_64 Linux. Use the RunPod CPU pod (Ubuntu 24.04), not the Mac or the cloud sandbox.
