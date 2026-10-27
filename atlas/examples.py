# Collect each target neuron's top-activating 64-token snippets from natural text (FineWeb-Edu +
# UltraChat), alongside its sweep lens words / S / p. This is the evidence a labeller sees (R6,
# Appendix D step 2), and the "show don't tell" material for the R6 grill. Honest: snippets come from
# natural corpus text, not sentences hand-picked to match the concept.
#
#   python atlas/examples.py --out atlas/examples.json [--windows 160]
import argparse, importlib.util, json
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parent.parent
torch.set_grad_enabled(False)
spec = importlib.util.spec_from_file_location("screen", ROOT / "atlas" / "screen.py")
SC = importlib.util.module_from_spec(spec); spec.loader.exec_module(SC)
IR = SC.IR

TARGETS = [(8, 908), (21, 1527), (22, 796), (23, 1482), (18, 2736)]  # vivid sweep survivors + control


def corpus(tok, n):
    from datasets import load_dataset
    out = []
    web = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True).shuffle(seed=7, buffer_size=2000)
    for row in web:
        ids = tok.encode(row["text"]).ids
        for s in range(32, min(len(ids) - 64, 32 + 64 * 3), 64):
            out.append(ids[s:s + 64])
        if len(out) >= n * 2 // 3:
            break
    chat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True).shuffle(seed=7, buffer_size=2000)
    sysp = json.loads((ROOT / "eval" / "prompts.json").read_text())["system"]
    for row in chat:
        ids = IR.chat_ids(sysp, row["prompt"].strip(), tok)
        ans = tok.encode(" " + row["messages"][1]["content"] if len(row["messages"]) > 1 else "").ids
        seq = (ids + ans)[:64]
        if len(seq) >= 40:
            out.append((seq + [tok.encode(".").ids[0]] * 64)[:64])
        if len(out) >= n:
            break
    return out[:n]


def main(a):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(IR.torq_file("tokenizer.json"))
    sweep = np.load(ROOT / "atlas" / "sweep_run.npz", allow_pickle=True)
    scr = SC.Screener(variant="int4-onnx", device="cpu")
    wins = corpus(tok, a.windows)
    print(f"corpus: {len(wins)} windows", flush=True)
    # per target: track top snippets by activation in the pole's direction
    best = {t: [] for t in TARGETS}
    sign = {(L, n): float(sweep[f"L{L}/sign"][n]) for L, n in TARGETS}
    for wi, ids in enumerate(wins):
        acts = scr.neuron_acts(torch.tensor(ids))         # [T,28,3072]
        for (L, n) in TARGETS:
            v = acts[:, L, n].float().numpy() * sign[(L, n)]   # pole direction
            pos = int(v.argmax()); best[(L, n)].append((float(v[pos]), wi, pos))
        if wi % 40 == 0:
            print(f"  {wi}/{len(wins)}", flush=True)
    res = {}
    for (L, n) in TARGETS:
        lens = [tok.decode([int(np.load(IR.torq_file("token_id_lut.npy"))[i])]) for i in sweep[f"L{L}/lens"][n]]
        top = sorted(best[(L, n)], reverse=True)[:3]
        snips = []
        for val, wi, pos in top:
            toks = wins[wi]
            lo, hi = max(0, pos - 12), min(64, pos + 6)
            txt = tok.decode(toks[lo:pos]) + " <<" + tok.decode([toks[pos]]) + ">> " + tok.decode(toks[pos + 1:hi])
            snips.append({"act": round(val, 2), "text": txt.replace("\n", " ")})
        res[f"L{L}n{n}"] = {"sign": "+" if sign[(L, n)] > 0 else "-", "S": round(float(sweep[f"L{L}/S_neu"][n]), 2),
                            "p": float(sweep[f"L{L}/p"][n]), "lens": [w.strip() for w in lens],
                            "snippets": snips}
        print(f"\nL{L} n{n} ({res[f'L{L}n{n}']['sign']}) S={res[f'L{L}n{n}']['S']} lens={res[f'L{L}n{n}']['lens'][:8]}")
        for s in snips:
            print(f"   [{s['act']:+.1f}] ...{s['text']}...")
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "atlas" / "examples.json")); ap.add_argument("--windows", type=int, default=160)
    main(ap.parse_args())
