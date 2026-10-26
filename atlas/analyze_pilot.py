# Turn a screen run (atlas/pilot_run.npz or atlas/sweep_run.npz) into the report numbers (spec #11):
# per-layer BH & BY survivors, the statistic vs the nulls, the positive controls, top-200 stability
# across the two context sets, an estimate of k and the null count it implies (Mary & Roquain 2022),
# a GPD tail-fit cross-check, cost projection, and the top neurons per layer with their lens words.
#
#   python atlas/analyze_pilot.py --run atlas/sweep_run.npz [--pilot-seconds S] [--topk 8]
import argparse, json
import numpy as np

M = 28 * 3072                      # 86,016 neurons
N_KEPT = 99477
TORQ, REV = "Synaptics/Qwen3-0.6B-GPTQ-int4-torq", "v2.3.0"
CONTROL = {8: (908, "ocean +"), 19: (834, "ships/sea -")}


def bh_by(p, q=0.05, m=M):
    order = np.argsort(p); ps = p[order]; k = np.arange(1, len(ps) + 1)
    def step(c):
        ok = ps <= q * k / (m * c)
        kmax = (np.where(ok)[0].max() + 1) if ok.any() else 0
        mask = np.zeros(len(p), bool)
        if kmax:
            mask[order[:kmax]] = True
        return mask, (ps[kmax - 1] if kmax else 0.0)
    cBY = np.log(m) + np.euler_gamma + 1 / (2 * m)
    return step(1.0), step(cBY)


def gpd_tail_p(S_null, S_val, tail_frac=0.1):
    try:
        from scipy.stats import genpareto
    except Exception:
        return None
    thr = np.quantile(S_null, 1 - tail_frac); exc = S_null[S_null > thr] - thr
    if len(exc) < 20 or S_val <= thr:
        return None
    c, loc, scale = genpareto.fit(exc, floc=0)
    return float(tail_frac * genpareto.sf(S_val - thr, c, loc, scale))


def make_decoder():
    """Return f(ids[10]) -> list[str] decoding trimmed-vocab indices to tokens, or None if unavailable."""
    try:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(hf_hub_download(TORQ, "tokenizer.json", revision=REV))
        lut = np.load(hf_hub_download(TORQ, "token_id_lut.npy", revision=REV)).astype(np.int64)
        return lambda ids: [tok.decode([int(lut[i])]) for i in ids if 0 <= i < N_KEPT]
    except Exception as e:
        print(f"(lens decode unavailable: {e})")
        return None


def main(a):
    d = np.load(a.run, allow_pickle=True)
    meta = json.loads(str(d["meta"])); layers = meta["layers"]; nulls = meta["nulls"]
    print(f"run: {len(layers)} layers, {nulls} nulls/layer, {meta['n_neurons']} neurons, "
          f"{meta['n_contexts']} contexts/set, variant {meta['variant']}")
    allp = np.concatenate([d[f"L{L}/p"] for L in layers])
    (bh, bht), (by, byt) = bh_by(allp)
    dec = make_decoder() if a.topk else None
    off = 0
    tot_bh = tot_by = 0
    for L in layers:
        p = d[f"L{L}/p"]; nL = len(p); sl = slice(off, off + nL); off += nL
        Sn, Snull, S2 = d[f"L{L}/S_neu"], d[f"L{L}/S_null"], d[f"L{L}/S_neu_set2"]
        bhL, byL = int(bh[sl].sum()), int(by[sl].sum()); tot_bh += bhL; tot_by += byL
        ov = len(set(np.argsort(-Sn)[:200]) & set(np.argsort(-S2)[:200])) / 200.0
        print(f"\n--- L{L} ---  BH {bhL}  BY {byL}  beat-all-nulls {int((p<=1/(nulls+1)+1e-12).sum())}"
              f"  top200-overlap {ov:.2f}")
        print(f"  null S mean {Snull.mean():+.3f} p95 {np.quantile(Snull,0.95):+.3f};  "
              f"neuron S p50 {np.median(Sn):+.3f} max {Sn.max():+.3f}")
        if L in CONTROL and CONTROL[L][0] < nL:
            cn, tag = CONTROL[L]
            gp = gpd_tail_p(Snull, Sn[cn])
            print(f"  control n{cn} ({tag}): S={Sn[cn]:+.3f} p={p[cn]:.4f} rank {int((Sn>Sn[cn]).sum())+1}"
                  + (f" GPD p={gp:.2e}" if gp else "") + ("  [BH survivor]" if bh[sl][cn] else ""))
        if dec is not None and f"L{L}/lens" in d:
            lens, sign = d[f"L{L}/lens"], d[f"L{L}/sign"]
            top = np.argsort(-Sn)[:a.topk]
            for n in top:
                words = ", ".join(w.strip() for w in dec(lens[n])[:8] if w.strip())
                print(f"    n{n:<4d} S={Sn[n]:+.2f} p={p[n]:.1e} {'+' if sign[n]>0 else '-'}  [{words}]")
    print(f"\n=== FDR over {len(allp)} tested neurons (m={M}) ===  BH {tot_bh}  BY {tot_by}  "
          f"(BH p<= {bht:.2e})")
    k = max(1, tot_bh); n_needed = int(np.ceil(M / (0.05 * k)))
    print(f"k (BH survivors) = {k}  ->  Mary-Roquain nulls/layer n >= m/(alpha*k) = {n_needed}")
    if a.pilot_seconds:
        up = len(layers) * (meta["n_neurons"] + nulls)
        print(f"  {a.pilot_seconds:.0f}s for {up} unit-layers; this run's rate scales linearly")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="atlas/sweep_run.npz")
    ap.add_argument("--pilot-seconds", type=float, default=0)
    ap.add_argument("--topk", type=int, default=8, help="top neurons per layer to show with lens words")
    main(ap.parse_args())
