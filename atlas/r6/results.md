# R6 results: labels for the top 3,002 poles (2026-10-27)

Ticket #13, spec #11. All work ran on Cloudroom sub-agents at **$0** (Jake's plan), with no OpenRouter
and no paid APIs. Wall time was about 3 h.

## Headline
- **3,002 / 3,002 poles labelled in the locked format**: the top 3,000 by effect size S (all BH survivors)
  plus the controls L8n908 and L19n834. Each has a name (≤ 4 words), a one-line description, the pole,
  a verdict and 10 concepts. Every label validates and every pole has its full evidence.
- **E4 passes.** Haiku simulation on the held-out **test** set, Pearson r per pole, 95% bootstrap CI over poles:

| | n | real | shuffled | word list | token lookup | real − shuffled | real − word list | real − lookup |
|---|---|---|---|---|---|---|---|---|
| **all** | 3,002 | **0.381** | 0.058 | 0.237 | 0.477 | **+0.323 [0.309, 0.338]** | **+0.144 [0.133, 0.154]** | −0.096 [−0.108, −0.084] |
| clear | 222 | 0.591 | 0.033 | 0.438 | 0.509 | +0.558 [0.506, 0.609] | +0.153 [0.122, 0.185] | +0.082 [0.048, 0.113] |
| likely | 718 | 0.441 | 0.033 | 0.289 | 0.457 | +0.408 [0.380, 0.437] | +0.152 [0.131, 0.173] | −0.016 [−0.040, 0.008] |
| mixed | 2,062 | 0.338 | 0.070 | 0.198 | 0.481 | +0.268 [0.251, 0.285] | +0.140 [0.126, 0.154] | −0.143 [−0.157, −0.129] |
| L27 (p1–p3) | 1,383 | 0.318 | 0.072 | 0.185 | 0.484 | +0.246 [0.225, 0.265] | +0.133 [0.116, 0.148] | −0.166 [−0.183, −0.150] |
| L25–26 (p4–p5) | 1,084 | 0.391 | 0.057 | 0.244 | 0.498 | +0.334 [0.313, 0.356] | +0.147 [0.129, 0.164] | −0.107 [−0.125, −0.089] |
| L0–24 (p6) | 535 | 0.526 | 0.025 | 0.358 | 0.416 | +0.501 [0.468, 0.532] | +0.167 [0.146, 0.189] | +0.109 [0.085, 0.133] |

Real labels beat same-layer shuffled labels and the word-list baseline in every slice, every layer group
and every verdict, and every CI is clear of 0 (per-slice numbers are in each `out/pX/qc.md`).
- **The free token-lookup predictor beats the labels overall** (Bills-style: each token's mean activation
  on the labelling text; no LLM). It wins on the L25–27 subword-fragment neurons, which a token table
  captures well. Labels beat it on L0–24 and on `clear` poles. This is a known result (Bills et al.
  2023), reported here, not a pass criterion.
- **Verdicts:** clear 222 (7%), likely 718 (24%), mixed 2,062 (69%). The top-S tail is mostly
  structural: the 15 strongest poles by S are all `mixed`, mostly L26–27 subword-fragment neurons that
  write `$a`, `=A`, `+N` or closing brackets. So `labels.json` gives each pole a **`feature_rank`**: verdict
  first, then dev r (the QC score, never the test score), then S.

## How it was done
- **Evidence** (`build_evidence.py` → `evidence.json.gz`):
  - 15 top-activating 64-token snippets per pole with every token's activation binned 0–10, plus the
    pole's top tokens, 10 logit-lens and 10 steer-lens words, and 2 greedy steered samples at the pole's
    ±0.5 dose (story and Sunday prompts).
  - The labelling text is 6,001 windows from FineWeb-Edu (ODC-By) and UltraChat (MIT) on the INT4 replica.
  - Memory peaked at 3.3 GB. Four helper sandboxes generated the 6,004 steered samples.
- **Held-out text** (`heldout.json.gz`), never shown to labellers:
  - **dev** (1,500 windows) was used for QC and for picking between label versions; **test** (1,501) was used
    once, for the score above.
  - Both are split by document from the labelling stream, which also contains the first orchestrator's corpus.
  - Per pole: 10 top and 10 random windows, shuffled. The truth is the window's peak activation.
- **Fan-out** (`label_fanout.py`):
  - **Wave 1** was 10 Haiku labellers on 200 poles: 200/200 valid, no retries.
  - **Six Opus planners** each took about 500 poles in layer-aligned slices (`slices/`), so shuffles stay
    within a layer.
  - Each planner labelled with Haiku (batch 20) and spot-checked every batch.
  - It dev-scored the labels, re-ran its weakest batches as v2, and kept the better version per pole on
    dev (522 final labels come from v2). It also brought in 15 R6-pilot labels where they scored better.
  - It escalated ambiguous dev cases to a fresh Opus scorer, used only for re-run decisions, and then
    scored test once.
  - Every sub-agent ran in its **own sandbox** (no `--shared-sandbox`, the cause of the first
    orchestrator's OOM) and was archived. There were about 1,300 sub-agents in all: 208 labeller and
    1,083 scorer jobs, retries included.
- **Rules that keep the score honest:**
  - Planners never read the held-out text, and no label changed after test scoring started.
  - Each test condition was scored once. A flat prediction counts as r = 0 in every condition.
  - The script re-asks a whole batch when nearly every pole comes back flat (a lazy reply). With
    unparseable replies, that was 20 of 1,083 scorer jobs (1.8%). It does this the same way for all conditions, and keeps the last attempt either way.

## Spot-check (human read of the top 30 featured, plus the controls)
All 30 read as sensible against their snippets and lenses. Many `clear` poles are next-token predictors:
they fire on the context that predicts a token and write that token.
- **L8n908 + "Sea and ship words"** (clear; dev 0.90, test 0.89). It fires on 'off' in "waters off
  Christmas Island" and on 'sea' and 'dhows', and writes shore, reef and ocean. At +0.5 it says "the ocean
  waves gently kiss the shore". In R3 it sat at S rank 49,847, so its label is a control, not a top pole.
- **L19n834 − "Marine and sea words"** (clear; test 0.95). It fires on 'marine' and 'sea' and writes
  marine, fish and sailor. Its steered samples at −0.5 show no sea, as in the CPU demo.
- Others, with test r:
  - L24n2097 − "Digit before percent sign" (0.69): fires on the 0 in "100%" and writes `%`.
  - L22n3055 − "'Have access', easy availability" (0.84): fires on "I do not have" and writes 'access'.
  - L24n2693 − "Sorting algorithm names" (0.88).
  - L21n1527 + "Random and pick words" (0.91); R3 found the same neuron.
  - L23n2646 − "'on' before a date" (0.92).
  - L23n1439 − "Colour adjectives" (0.90).
  - L25n1627 − "Travel booking words" (0.82).
  - L24n1084 − "Taxonomic family and genus" (0.83).
- **Weak spots the planners flagged:**
  - Some `clear` verdicts are generous. L27n2319 "Word before 'Administration'" scores test 0.07.
  - Some labels describe only what a pole writes, for example 'button' or 'best' writers.
  - 52% of concept lists copy at least 8 of their 10 words from the TOP TOKENS. That's allowed, but some
    contradict their description.
  - 472 poles (15.7%) score under r 0.10 on test. The atlas keeps them, ranked low, and doesn't hide them.

## Other findings
- **Spawn failures were a detached process, not a quota.** One planner's spawns failed after it had made
  143 sub-agents. `room-cli` refuses to spawn from a process detached from its agent's tree ("Run cloudroom
  thread spawn from a Cloudroom cloud thread"), and stderr wasn't logged at first. I ran that slice's
  word-list test condition from my sandbox. Fan-outs must run as tracked background tasks.
- **Dev picking inflates dev r:** p4's final dev r was 0.408 against 0.387 on test. Test is the number to quote.
- **The orphaned planner2** from the first orchestrator never pushed anything. Its workers died with the
  shared sandbox, so all poles were labelled from scratch here.

## Files
- `labels.json`: the final record per pole, in feature order. It holds the label, layer/neuron/S/p/rank,
  version, planner, dev r and test r for every condition.
- `e4.json`: the global and per-group summary above.
- `evidence.json.gz` and `heldout.json.gz` are what labellers and scorers saw.
- `out/pX/`: per-planner label versions, raw scorer predictions, grades, `qc.md` and the fan-out state.
- `PLANNER.md` is the planner brief, and `slices/` holds the six slices.
- Reproduce the scores: `python3 atlas/label_fanout.py merge`.
