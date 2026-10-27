# Build the R6 pilot evidence bundle for ~50 poles: top-activating snippets (reads), logit-lens +
# steer-lens words (writes), and one steered sample (effect). Selects the pilot poles from the top
# ~3,000 by effect size S (plus the two controls), using the sweep. Output feeds the Haiku labellers.
#
#   python atlas/label_pilot.py --out atlas/label_pilot_evidence.json [--windows 160] [--n 50]
import argparse, importlib.util, json
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parent.parent
torch.set_grad_enabled(False)
spec = importlib.util.spec_from_file_location("screen", ROOT / "atlas" / "screen.py")
SC = importlib.util.module_from_spec(spec); spec.loader.exec_module(SC)
IR = SC.IR
CONTROLS = [(8, 908), (19, 834)]


def pick_poles(sweep, layers, n):
    """Top ~3,000 poles by S; pilot = controls + a stratified spread across S and layers."""
    allp = [(float(sweep[f"L{L}/S_neu"][i]), L, i, int(sweep[f"L{L}/sign"][i]))
            for L in layers for i in range(3072)]
    allp.sort(reverse=True)
    top3000 = allp[:3000]
    picks = [(L, i) for L, i in CONTROLS]
    # strata: ranks 0-9, then even spread through the top-3000, plus a few near the cutoff
    idx = list(range(10)) + list(np.linspace(10, 2999, n - len(picks) - 10).astype(int))
    for j in idx:
        L, i = top3000[j][1], top3000[j][2]
        if (L, i) not in picks:
            picks.append((L, i))
    return picks[:n], {(L, i): rank for rank, (_, L, i, _) in enumerate(top3000)}


def corpus(tok, n):
    from datasets import load_dataset
    out = []
    web = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True).shuffle(seed=5, buffer_size=2000)
    for row in web:
        ids = tok.encode(row["text"]).ids
        for s in range(32, min(len(ids) - 64, 32 + 64 * 4), 64):
            out.append(ids[s:s + 64])
        if len(out) >= n:
            break
    return out[:n]


def logit_lens(scr, L, n, sign, tok, lut):
    """Words the neuron's W_down column promotes through the final norm + head (weights only)."""
    col = scr.Wdown[L][:, n] * sign
    h = scr.hf.norm(col.to(scr.device)[None].to(next(scr.hf.norm.parameters()).dtype))
    logits = scr.ref.head(h).float()[0, :SC.N_KEPT]
    return [tok.decode([int(lut[i])]).strip() for i in logits.topk(10).indices.cpu()]


def steered_sample(scr, L, n, sign, prompt_ids, med, dose=0.5, k=24):
    """Greedy sample with neuron n pushed sign*dose at every position (trimmed-vocab head)."""
    push = torch.zeros(scr.model.config.intermediate_size, device=scr.device)
    push[n] = sign * dose * med / scr.colnorm[L][n].clamp_min(1e-20)
    dp = scr.hf.layers[L].mlp.down_proj
    h = dp.register_forward_pre_hook(lambda m, a: (a[0] + push.to(a[0].dtype),) + a[1:])
    try:
        gen, _ = scr.ref.greedy(list(prompt_ids), k)
    finally:
        h.remove()
    from tokenizers import Tokenizer  # decode via the board tokenizer
    return gen


def main(a):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(IR.torq_file("tokenizer.json"))
    lut = np.load(IR.torq_file("token_id_lut.npy")).astype(np.int64)
    sweep = np.load(ROOT / "atlas" / "sweep_run.npz", allow_pickle=True)
    layers = json.loads(str(sweep["meta"]))["layers"]
    scr = SC.Screener(variant="int4-onnx", device="cpu")
    poles, ranks = pick_poles(sweep, layers, a.n)
    print(f"pilot poles: {len(poles)} (incl controls); corpus {a.windows} windows", flush=True)
    wins = corpus(tok, a.windows)
    sign = {(L, n): float(sweep[f"L{L}/sign"][n]) for (L, n) in poles}
    best = {(L, n): [] for (L, n) in poles}
    for wi, ids in enumerate(wins):
        acts = scr.neuron_acts(torch.tensor(ids))
        for (L, n) in poles:
            v = acts[:, L, n].float().numpy() * sign[(L, n)]
            p = int(v.argmax()); best[(L, n)].append((float(v[p]), wi, p))
        if wi % 40 == 0:
            print(f"  acts {wi}/{len(wins)}", flush=True)
    # steered sample shared prompt + per-pole med
    sys0 = json.loads((ROOT / "eval" / "prompts.json").read_text())["system"]
    pr = IR.chat_ids(sys0, "Tell me a short story.", tok)
    base_med = scr.base(torch.tensor([pr]))["med"]
    uns, _ = scr.ref.greedy(list(pr), 24)
    res = {"unsteered_baseline": tok.decode(uns, skip_special_tokens=True), "poles": {}}
    for k, (L, n) in enumerate(poles):
        sg = steered_sample(scr, L, n, sign[(L, n)], pr, base_med[L])
        top = sorted(best[(L, n)], reverse=True)[:15]
        snips = []
        for val, wi, pos in top:
            t = wins[wi]; lo, hi = max(0, pos - 16), min(64, pos + 6)
            snips.append({"act": round(val, 2),
                          "text": (tok.decode(t[lo:pos]) + " <<" + tok.decode([t[pos]]) + ">> " + tok.decode(t[pos + 1:hi])).replace("\n", " ")})
        key = f"L{L}n{n}"
        res["poles"][key] = {
            "layer": L, "neuron": n, "sign": "+" if sign[(L, n)] > 0 else "-",
            "S": round(float(sweep[f"L{L}/S_neu"][n]), 2), "p": float(sweep[f"L{L}/p"][n]),
            "rank_in_top3000": ranks.get((L, n), -1), "is_control": (L, n) in CONTROLS,
            "steer_lens": [w.strip() for w in [tok.decode([int(lut[i])]) for i in sweep[f"L{L}/lens"][n]]],
            "logit_lens": logit_lens(scr, L, n, sign[(L, n)], tok, lut),
            "snippets": snips, "steered_sample": tok.decode(sg, skip_special_tokens=True)}
        if k % 10 == 0:
            print(f"  evidence {k}/{len(poles)}", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}: {len(res['poles'])} poles")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "atlas" / "label_pilot_evidence.json"))
    ap.add_argument("--windows", type=int, default=160); ap.add_argument("--n", type=int, default=50)
    main(ap.parse_args())
