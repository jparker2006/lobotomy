# 18-agent-prompts

Code reader report on lyramakesmusic/neuron-explorer @ f79b219. Analysis only; no code copied.

**1. What this slice does**
These two files are the system prompt for the explorer's in-app Claude agent. `server.py` joins `agent_prompt.md`, then `agent_methodology.md`, then an optional untracked `agent_prompt.local.md`, fills in `{ROOT}`, and passes the result to `claude -p --system-prompt-file` (server.py:876-885, 977). They tell the agent how to test a neuron's three views and how to report it.

**2. Key mechanics**
- **Unit is a pole:** `L22 N2742+` and `L22 N2742-`. Describe a verb, not a noun list (prompt:10-13).
- **Method, in order** (methodology:15-21): (1) read `neuron` and `web_summary`, and guess from the input/output pair; (2) steer open prompts at dose 0.6–0.8, then 1.5–2; (3) `compare` (max 24 texts, agent_mcp.py:92) to find a peak above `act_max`; (4) ablate upstream at the peak and measure the drop; (5) run a falsifier, e.g. swap the key word; (6) blind-max: write one sentence from the label alone; (7) close with `logprobs`.
- **Key rules:** "A neuron is a target, not a word list" (m:7). "Multi-hop paths are guesses from weights" (p:53). Weight links are "Leads to check, never circuits" (m:28).
- **Verdict:** `one job / two related jobs / not yet understood · confidence low / medium / high` (p:118).
- **Report** (p:113-126, about 120 words, "reply with this and nothing else"): **Name** (2–6 words, situation plus action) · **Verdict** · **+ pole** / **− pole** ("In <situation>, it <action>") · **Evidence** (one bullet each for Input, Direct, Downstream) · **Caveat** (only if sources disagree) · **Test**.

**3. Inputs -> outputs**
- **In:** the two `.md` files, plus tool results from `/api/agent/*` on `127.0.0.1:8791` (prompt:137-153) and `/api/web/summary`.
- **Out:** prose in the report format, and map actions such as `select_neuron` and `highlight_neurons`.
- **Writes:** only `explorer/cache/agent/system_prompt.md` (server.py:880-885). Scratch work goes under `explorer/cache/agent/work/` by instruction.

**4. Porting notes for Gemma-270M / Coralboard**
- **Transfers:** the method (poles, three views, falsifiers, blind max, logprob close), the verdict scale, and the Name/pole format.
- **Must change:** the Qwen facts (32 layers × 9216, hidden 2560, Qwen tokenizer, the Chinese-gloss rule, reference neurons, motifs like `</think>`, `作为`). The SwiGLU sign-regime paragraph (prompt:14-16) does not carry over. GELU's negative lobe is tiny (minimum about −0.17), so the "double negative echo" argument does not apply.
- **Foldable pushes:** `mul` is a scale on a W_down column, exact and free. `add` is a constant residual add of dose × median norm × unit column, which can be baked into the compiled graph.
- **Cannot run on-device:** the whole agent loop (Claude CLI, MCP, live steering, ablation, `compare`, `logprobs`). It runs on a host. The Coralboard can only run fixed, pre-baked steered generation.

**5. Surprises, bugs, open questions**
- **Calibration mismatch:** `act_max` comes from FineWeb-Edu with position 0 skipped and no chat template (runtime.py:472-503). The methodology still cites chat-only peaks. `L22 N3467` reaches 6.2 at `</think>` against a stored max of 3.14 (`act_scale.npz`, pos[22, 3467]), so the ratio is not comparable there.
- **Base rates:** the 90/7/2–4% split (prompt:7-8) comes from a preliminary Qwen3-32B audit per [secondary coverage](https://www.ibtimes.co.uk/ai-researcher-challenges-black-box-view-language-model-audit-1826618). Coverage also reports that Neel Nanda disputes the readability claim. The prompt states the split without that caveat, which invites anchoring. I could not verify the "Titanic-ish" quote (m:7).
- **Conflicting formats:** the report puts Name first (p:116), but p:103 says evidence comes before interpretation and m:49 says lead with the strongest firings. The methodology's "Writing it up" section describes a different, longer format.
- **Pole notation:** the prompt uses ASCII `-`, while the methodology and UI use `−` (app.js:12). Pick one.
- **Missing directory:** nothing in server.py creates `explorer/cache/agent/work/`, so scripts that write there may fail.
- **Verified:** every tool name in the prompt exists (agent_mcp.py:16-152). "Saturates near 96" holds: a z-score over 9216 entries cannot exceed √9215 ≈ 96 (build_connections.py:71).
