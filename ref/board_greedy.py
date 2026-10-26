# Run ON THE BOARD: Synaptics' stock Qwen3 (torq-examples qwen3 runner) on eval/prompts.json.
#
#   python3 board_greedy.py --examples ~/torq-examples -m <dir>/transformer.vmfb --out board_greedy.json
#   python3 board_greedy.py ... --tf ref_greedy.json --out board_tf.json   # teacher-forced argmax
#
# Greedy mode writes the same JSON shape as `int4_ref.py greedy`, so on the Mac or sandbox:
#   python ref/int4_ref.py compare eval/ref_greedy.json board_greedy.json --n 32
# Each prompt starts a fresh conversation (reset to the post-system-prompt state), unlike the chat loop.
import argparse, json, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--examples", required=True, help="torq-examples checkout")
ap.add_argument("-m", "--model", required=True, help="transformer.vmfb (lm_head.vmfb.trim etc. next to it)")
ap.add_argument("--prompts", default=str(Path(__file__).resolve().parent.parent / "eval/prompts.json"))
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--tf", help="ref_greedy.json: feed its tokens, record the board's argmax at each answer position")
ap.add_argument("--out", required=True)
a = ap.parse_args()

ex = Path(a.examples).resolve()
sys.path[:0] = [str(ex), str(ex / "qwen3" / "src")]
from runner import Qwen3Static  # noqa: E402
from utils.runtime import setup_npu_for_inference, cleanup_npu_after_inference  # noqa: E402

cfg = json.loads(Path(a.prompts).read_text())
setup_npu_for_inference()
q = Qwen3Static(a.model, instruct_model=True, thinking=False, temperature=0.0,
                sys_prompt=cfg["system"], max_gen_tokens=a.n)
warm = q._build_warmup_tokens()
res = {"variant": "board", "system": cfg["system"], "n": a.n, "prompts": []}
try:
    if a.tf:
        agree = total = 0
        for p in json.loads(Path(a.tf).read_text())["prompts"]:
            q.reset()
            seq, n_prompt = p["prompt_ids"] + p["gen_ids"], len(p["prompt_ids"])
            assert seq[:len(warm)] == warm, "system prompt tokens differ from the reference"
            preds = []
            for pos in range(len(warm), len(seq) - 1):
                pred = q.llm_step(seq[pos], pos, compute_logits=pos >= n_prompt - 1,
                                  sample_next=pos >= n_prompt - 1)
                if pos >= n_prompt - 1:
                    preds.append(pred)
            hit = sum(x == y for x, y in zip(preds, p["gen_ids"]))
            agree, total = agree + hit, total + len(preds)
            res["prompts"].append({"id": p["id"], "tf_pred_ids": preds, "ref_gen_ids": p["gen_ids"][:len(preds)],
                                   "agree": hit})
            print(f"[{p['id']}] teacher-forced top-1 agreement {hit}/{len(preds)}", flush=True)
        res["tf_top1_agreement"] = agree / max(total, 1)
        print(f"overall {agree}/{total}")
    else:
        for p in cfg["prompts"]:
            q.reset()
            prompt = q._build_prompt_tokens(p["text"])
            tok, pos = q.prefill_tokens(prompt)
            gen = [tok]
            while not q._should_stop(tok, gen) and len(gen) < a.n:
                tok = q.llm_step(tok, pos)
                pos += 1
                gen.append(tok)
            text = q._tokenizer.decode(gen, skip_special_tokens=True)
            res["prompts"].append({"id": p["id"], "prompt_ids": warm + prompt, "gen_ids": gen, "text": text})
            print(f"[{p['id']}] {text!r}", flush=True)
finally:
    cleanup_npu_after_inference()
Path(a.out).write_text(json.dumps(res, indent=1))
