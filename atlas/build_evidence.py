# Build labelling evidence for the top-N poles by effect size S (default 3000): top-15 activating
# 64-token snippets (reads) + logit-lens + steer-lens words (writes). Steered samples are added later
# for the featured neurons only (generating 3000 on CPU is too slow). Output feeds the Haiku workers.
#
#   python atlas/build_evidence.py --top 3000 --windows 2500 --out atlas/labels_evidence.json
import argparse, importlib.util, json
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parent.parent
torch.set_grad_enabled(False)
spec = importlib.util.spec_from_file_location("screen", ROOT / "atlas" / "screen.py")
SC = importlib.util.module_from_spec(spec); spec.loader.exec_module(SC)
IR = SC.IR


def top_poles(sweep, layers, top):
    a = [(float(sweep[f"L{L}/S_neu"][i]), L, i, int(sweep[f"L{L}/sign"][i]))
         for L in layers for i in range(3072)]
    a.sort(reverse=True)
    return a[:top]


def corpus(tok, n):
    from datasets import load_dataset
    out = []
    for name, cfg, split, field in [("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text"),
                                    ("HuggingFaceH4/ultrachat_200k", None, "train_sft", None)]:
        ds = load_dataset(name, cfg, split=split, streaming=True).shuffle(seed=3, buffer_size=3000)
        want = n * 2 // 3 if field else n
        for row in ds:
            if field:
                ids = tok.encode(row[field]).ids
                for s in range(32, min(len(ids) - 64, 32 + 64 * 5), 64):
                    out.append(ids[s:s + 64])
            else:
                sysp = json.loads((ROOT / "eval" / "prompts.json").read_text())["system"]
                msg = row["messages"][1]["content"] if len(row.get("messages", [])) > 1 else ""
                ids = IR.chat_ids(sysp, row["prompt"].strip(), tok) + tok.encode(" " + msg).ids
                out.append((ids + [tok.encode(".").ids[0]] * 64)[:64])
            if len(out) >= want:
                break
        if not field:
            break
    return out[:n]


def logit_lens_batch(scr, poles, lut, tok):
    """Words each pole's W_down column promotes (weights only). poles: list of (L,n,sign)."""
    out = {}
    normw = next(scr.hf.norm.parameters())
    for (L, n, sg) in poles:
        col = (scr.Wdown[L][:, n] * sg).to(normw.dtype)
        h = scr.hf.norm(col[None])
        lg = scr.ref.head(h).float()[0, :SC.N_KEPT]
        out[(L, n)] = [tok.decode([int(lut[i])]).strip() for i in lg.topk(10).indices.cpu()]
    return out


def main(a):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(IR.torq_file("tokenizer.json"))
    lut = np.load(IR.torq_file("token_id_lut.npy")).astype(np.int64)
    sweep = np.load(ROOT / "atlas" / "sweep_run.npz", allow_pickle=True)
    layers = json.loads(str(sweep["meta"]))["layers"]
    scr = SC.Screener(variant="int4-onnx", device="cpu")
    poles = top_poles(sweep, layers, a.top)
    Ls = np.array([L for _, L, _, _ in poles]); Ns = np.array([n for _, _, n, _ in poles])
    Sg = torch.tensor([float(s) for _, _, _, s in poles])
    print(f"evidence for {len(poles)} poles; building corpus...", flush=True)
    wins = corpus(tok, a.windows)
    print(f"corpus {len(wins)} windows; activation pass...", flush=True)
    K = 15
    heaps = [[] for _ in range(len(poles))]   # per pole: list of (act, wi, pos), kept top-K
    Lt, Nt = torch.tensor(Ls), torch.tensor(Ns)
    for wi, ids in enumerate(wins):
        acts = scr.neuron_acts(torch.tensor(ids))             # [T,28,3072]
        sel = acts[:, Lt, Nt].float() * Sg                    # [T, P] pole activations (pole dir)
        vmax, pmax = sel.max(0)                                # [P]
        vmax = vmax.numpy(); pmax = pmax.numpy()
        for p in range(len(poles)):
            h = heaps[p]
            if len(h) < K:
                h.append((float(vmax[p]), wi, int(pmax[p])))
                if len(h) == K: h.sort()
            elif vmax[p] > h[0][0]:
                h[0] = (float(vmax[p]), wi, int(pmax[p])); h.sort()
        if wi % 200 == 0:
            print(f"  {wi}/{len(wins)}", flush=True)
    print("logit lens...", flush=True)
    llens = logit_lens_batch(scr, [(L, n, s) for _, L, n, s in [(x[0], x[1], x[2], x[3]) for x in poles]], lut, tok)
    res = {}
    for pi, (S, L, n, sg) in enumerate(poles):
        snips = []
        for val, wi, pos in sorted(heaps[pi], reverse=True):
            t = wins[wi]; lo, hi = max(0, pos - 16), min(64, pos + 6)
            snips.append({"act": round(val, 2),
                          "text": (tok.decode(t[lo:pos]) + " <<" + tok.decode([t[pos]]) + ">> " + tok.decode(t[pos + 1:hi])).replace("\n", " ")})
        res[f"L{L}n{n}"] = {"layer": L, "neuron": n, "sign": "+" if sg > 0 else "-", "S": round(S, 2),
                            "p": float(sweep[f"L{L}/p"][n]), "rank": pi,
                            "steer_lens": [tok.decode([int(lut[i])]).strip() for i in sweep[f"L{L}/lens"][n]],
                            "logit_lens": llens[(L, n)], "snippets": snips}
    Path(a.out).write_text(json.dumps({"poles": res}, indent=1))
    print(f"wrote {a.out}: {len(res)} poles")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=3000); ap.add_argument("--windows", type=int, default=2500)
    ap.add_argument("--out", default=str(ROOT / "atlas" / "labels_evidence.json"))
    main(ap.parse_args())
