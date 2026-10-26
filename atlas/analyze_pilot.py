# Turn the R3 pilot's raw output (atlas/pilot_run.npz) into the report numbers (spec #11, R3):
# survivors per layer under FDR (BH + BY), the statistic vs the nulls, whether the positive controls
# survive, top-200 stability across the two context sets, an estimate of k (real neurons) and the
# full-sweep null count it implies (Mary & Roquain 2022), and a GPD tail-fit cross-check.
#
#   python atlas/analyze_pilot.py --run atlas/pilot_run.npz [--pilot-seconds S]
import argparse, json
import numpy as np

M = 28 * 3072                      # 86,016 neurons in the whole model
CONTROL = {8: (908, "ocean +"), 19: (834, "ships/sea -")}


def bh_by(p, q=0.05, m=M):
    order = np.argsort(p); ps = p[order]; k = np.arange(1, len(ps) + 1)
    def step(c):
        ok = ps <= q * k / (m * c)
        kmax = (np.where(ok)[0].max() + 1) if ok.any() else 0
        mask = np.zeros(len(p), bool); mask[order[:kmax]] = kmax > 0 and True
        if kmax:
            mask[order[:kmax]] = True
        return mask, (ps[kmax - 1] if kmax else 0.0)
    cBY = np.log(m) + np.euler_gamma + 1 / (2 * m)
    return step(1.0), step(cBY)


def gpd_tail_p(S_null, S_val, tail_frac=0.1):
    """Generalized-Pareto tail fit (Knijnenburg 2009): p for S_val beyond the empirical floor."""
    try:
        from scipy.stats import genpareto
    except Exception:
        return None
    thr = np.quantile(S_null, 1 - tail_frac)
    exc = S_null[S_null > thr] - thr
    if len(exc) < 20 or S_val <= thr:
        return None
    c, loc, scale = genpareto.fit(exc, floc=0)
    return tail_frac * genpareto.sf(S_val - thr, c, loc, scale)


def main(a):
    d = np.load(a.run, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    layers = meta["layers"]; nulls = meta["nulls"]
    print(f"R3 pilot: layers {layers}, {nulls} nulls/layer, {meta['n_neurons']} neurons, "
          f"{meta['n_contexts']} contexts/set, variant {meta['variant']}, floor {meta['floor']}")
    pfloor = 1.0 / (nulls + 1)
    allp, per_layer_k = [], {}
    for L in layers:
        Sn, Snull = d[f"L{L}/S_neu"], d[f"L{L}/S_null"]
        p = d[f"L{L}/p"]                                   # primary = pre-registered no-floor lift
        p_floored = d[f"L{L}/p_floored"]                   # secondary (base-prob floor), for the record
        S2 = d[f"L{L}/S_neu_set2"]
        allp.append(p)
        beat_all = int((p <= pfloor + 1e-12).sum())
        per_layer_k[L] = beat_all
        t1 = set(np.argsort(-Sn)[:200]); t2 = set(np.argsort(-S2)[:200])
        ov = len(t1 & t2) / 200.0
        print(f"\n--- L{L} ---")
        print(f"  neurons beating all {nulls} nulls (p={pfloor:.1e}): {beat_all}  min p={p.min():.1e}")
        print(f"  null S: mean {Snull.mean():+.3f}  p95 {np.quantile(Snull,0.95):+.3f}  "
              f"max {Snull.max():+.3f};  neuron S: p50 {np.median(Sn):+.3f} max {Sn.max():+.3f}")
        print(f"  top-200 overlap set1 vs set2: {ov:.2f}")
        if L in CONTROL:
            cn, tag = CONTROL[L]
            gp = gpd_tail_p(Snull, Sn[cn])
            print(f"  control n{cn} ({tag}): S={Sn[cn]:+.3f}  p={p[cn]:.4f}  p_floored={p_floored[cn]:.4f}"
                  f"  rank {int((Sn > Sn[cn]).sum())+1}/3072" + (f"  GPD p={gp:.2e}" if gp else ""))
    allp = np.concatenate(allp)
    (bh_m, bh_t), (by_m, by_t) = bh_by(allp)
    print(f"\n=== FDR over {len(allp)} tested neurons (m={M}) ===")
    print(f"  BH q=0.05: {int(bh_m.sum())} survivors (p<= {bh_t:.2e})")
    print(f"  BY q=0.05: {int(by_m.sum())} survivors (p<= {by_t:.2e})")
    # k estimate and the full-sweep null count it implies (n ~ m/(alpha*k), Mary & Roquain 2022)
    k_tested = int(bh_m.sum())
    k_full = max(1, round(k_tested * (28 / len(layers))))
    n_needed = int(np.ceil(M / (0.05 * max(k_full, 1))))
    print(f"\n=== projection ===")
    print(f"  tested-layer BH survivors: {k_tested}  ->  scaled k_hat (28 layers) ~ {k_full}")
    print(f"  implied full-run nulls/layer n >= m/(alpha*k) = {n_needed}")
    if a.pilot_seconds:
        units_pilot = len(layers) * (meta["n_neurons"] + nulls)
        units_full = 28 * (3072 + max(n_needed, nulls))
        hrs = a.pilot_seconds / 3600 * units_full / units_pilot
        print(f"  pilot {a.pilot_seconds}s for {units_pilot} unit-layers  ->  full sweep ~{hrs:.1f} GPU-h "
              f"(~${hrs*0.18:.2f} at $0.18/h, ~${hrs*0.34:.2f} at $0.34/h)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="atlas/pilot_run.npz")
    ap.add_argument("--pilot-seconds", type=float, default=0, help="wall time of the pilot screen, for cost projection")
    main(ap.parse_args())
