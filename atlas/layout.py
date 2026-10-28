# R9 layout (#15, spec #11 I4): a 3D map where x = layer and (y, z) = a co-activation embedding, so neurons
# that fire together on real text sit together. Scored against a weight-only baseline (a per-layer UMAP of
# normalized W_down columns, like Lyra's) by neighbour-label agreement on the R6 labels.
#
# Method: 16,384 random held-out positions (R6 dev+test windows, positions >= 1) x all 86,016 neurons,
# pole-aligned (activation x the neuron's R3 sign); standardize each neuron; 64 shared PCA directions
# (randomized SVD, streamed from disk); UMAP to 2D. Baseline: per layer, UMAP of the pole-aligned,
# L2-normalized W_down columns (INT4 replica weights). Both UMAPs: n_neighbors 15, min_dist 0.1, cosine,
# seed 0.
#
# Pre-registered score (fixed before any layout was computed):
#  - neighbours: for each labelled pole, its k = 10 nearest labelled poles of the SAME layer by 2D (y, z)
#    distance, in layers with >= 20 labelled poles (L21-L27);
#  - agreement_i = mean cos(e_i, e_j) over those neighbours - mean cos(e_i, e_j) over all other labelled
#    poles of the layer, e = bge-small-en-v1.5 embedding of "name. description";
#  - pass: mean paired difference (co-activation - W_down) > 0 with a 95% bootstrap CI (10,000 resamples
#    over poles) that excludes 0.
#  Also reported: neighbour co-activation on 4,096 other held-out positions (no labels), and cross-layer
#  label agreement for the co-activation layout (the baseline has no shared axes across layers).
#
#   OMP_NUM_THREADS=2 MALLOC_ARENA_MAX=2 python atlas/layout.py all --work /code/work/r9
import argparse, ctypes, gzip, importlib.util, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "atlas" / "layout"
NL, NN = 28, 3072
NU = NL * NN
N_LAYOUT, N_HELD, K, MIN_LABELLED, SEED = 16384, 4096, 10, 20, 0
UMAP_KW = dict(n_components=2, n_neighbors=15, min_dist=0.1, random_state=SEED)


def guard(msg):
    rss = int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) / 1e6
    print(f"{msg}  [rss {rss:.2f} GB]", flush=True)
    if rss > 5.0:
        raise SystemExit(f"rss {rss:.2f} GB is over the 5 GB budget")


def signs():
    sw = np.load(ROOT / "atlas" / "sweep_run.npz", allow_pickle=True)
    return np.concatenate([sw[f"L{L}/sign"].astype(np.float32) for L in range(NL)])   # [86016]


def cmd_acts(a, W):
    """Sample positions and write pole-aligned activations of all 86,016 neurons to two fp16 memmaps."""
    import torch
    torch.set_grad_enabled(False); torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    spec = importlib.util.spec_from_file_location("screen", ROOT / "atlas" / "screen.py")
    SC = importlib.util.module_from_spec(spec); spec.loader.exec_module(SC)
    scr = SC.Screener(variant="int4-onnx", device="cpu")
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    guard("model loaded")
    C = np.load(Path(a.evidence_work) / "corpus.npz")
    held_windows = np.concatenate([C["dev_ids"], C["test_ids"]])        # never shown to labellers
    rng = np.random.default_rng(SEED)
    sets = {"layout": (held_windows, N_LAYOUT), "heldout": (C["label_ids"][:820], N_HELD)}
    sg = torch.from_numpy(signs())
    for name, (wins, n) in sets.items():
        pairs = rng.choice(len(wins) * 63, n, replace=False)
        wi, pos = pairs // 63, pairs % 63 + 1                            # positions 1..63
        order = np.lexsort((pos, wi)); wi, pos = wi[order], pos[order]
        np.savez(W / f"{name}_index.npz", window=wi, pos=pos)
        mm = np.lib.format.open_memmap(W / f"{name}_acts.npy", mode="w+", dtype=np.float16, shape=(n, NU))
        uw = np.unique(wi)
        t0 = time.time()
        for b0 in range(0, len(uw), a.batch):
            ws = uw[b0:b0 + a.batch]
            scr.hf(input_ids=torch.from_numpy(wins[ws].astype(np.int64)), use_cache=False)
            A = torch.stack(scr.ref.acts, dim=2).reshape(len(ws), 64, NU)     # [B,64,86016] bf16
            for j, w in enumerate(ws):
                rows = np.nonzero(wi == w)[0]
                mm[rows] = (A[j, torch.from_numpy(pos[rows])].float() * sg).half().numpy()
            if (b0 // a.batch) % 100 == 0:
                mm.flush()
                guard(f"  {name}: {b0 + len(ws)}/{len(uw)} windows, {time.time() - t0:.0f}s")
        mm.flush(); del mm


def stream(path, chunk=1024):
    mm = np.load(path, mmap_mode="r")
    for r0 in range(0, mm.shape[0], chunk):
        yield r0, np.asarray(mm[r0:r0 + chunk], dtype=np.float32)


def cmd_pca(a, W):
    """64 shared PCA directions of the standardized neuron profiles (neurons are the points), by a
    randomized SVD (Halko et al. 2011) that streams the [16384 x 86016] matrix from disk."""
    path = W / "layout_acts.npy"
    s1 = np.zeros(NU); s2 = np.zeros(NU); n = 0
    for _, X in stream(path):
        s1 += X.sum(0); s2 += (X.astype(np.float64) ** 2).sum(0); n += len(X)
    mu = s1 / n
    sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0))
    inv = np.where(sd > 1e-6, 1 / np.maximum(sd, 1e-12), 0).astype(np.float32)
    mu32 = mu.astype(np.float32)
    np.savez(W / "standardize.npz", mu=mu32, inv=inv)
    print(f"standardized: {int((inv == 0).sum())} constant neurons", flush=True)
    rng = np.random.default_rng(SEED)
    r, q = 64 + 10, 3
    om = rng.standard_normal((n, r)).astype(np.float32)
    Y = np.zeros((NU, r), np.float32)
    for r0, X in stream(path):                                    # Y = Xs^T Omega  (Xs: neurons x positions)
        Y += ((X - mu32) * inv).T @ om[r0:r0 + len(X)]
    for it in range(q):                                           # power iterations, one fused pass each
        Y, _ = np.linalg.qr(Y)
        Y2 = np.zeros_like(Y)
        for _, X in stream(path):
            Z = (X - mu32) * inv
            Y2 += Z.T @ (Z @ Y)
        Y = Y2
        print(f"  power iteration {it + 1}/{q}", flush=True)
    Q, _ = np.linalg.qr(Y)
    B = np.zeros((r, n), np.float32)
    for r0, X in stream(path):                                    # B = Q^T Xs  -> [r x positions]
        B[:, r0:r0 + len(X)] = (((X - mu32) * inv) @ Q).T
    Ub, S, _ = np.linalg.svd(B, full_matrices=False)
    coords = (Q @ Ub[:, :64]) * S[:64]                              # [86016, 64]
    total = float((inv > 0).sum()) * n                              # sum of squared standardized entries
    np.save(W / "pca64.npy", coords.astype(np.float32))
    ev = (S[:64] ** 2) / total
    print(f"PCA: top-64 explain {ev.sum():.3f} of variance (first 5: {np.round(ev[:5], 4).tolist()})", flush=True)
    json.dump({"explained": ev.tolist()}, open(W / "pca64.json", "w"))


def umap2d(X, metric):
    import umap
    return umap.UMAP(metric=metric, **UMAP_KW).fit_transform(X).astype(np.float32)


def cmd_umap(a, W):
    P = np.load(W / "pca64.npy")
    nrm = np.linalg.norm(P, axis=1, keepdims=True)
    P = np.where(nrm > 0, P / np.maximum(nrm, 1e-12), 0)            # euclidean on unit rows = cosine, dead rows ok
    t0 = time.time()
    Y = umap2d(P, "euclidean")
    np.save(W / "coact_yz.npy", Y)
    print(f"co-activation UMAP: {time.time() - t0:.0f}s", flush=True)


def cmd_wdown(a, W):
    """Baseline: per layer, UMAP of the pole-aligned, L2-normalized W_down columns (bf16-rounded INT4)."""
    import torch
    spec = importlib.util.spec_from_file_location("int4_ref", ROOT / "ref" / "int4_ref.py")
    IR = importlib.util.module_from_spec(spec); spec.loader.exec_module(IR)
    from safetensors.numpy import load_file
    ckpt = load_file(IR.torq_file("gptq/model.safetensors"))
    sg = signs().reshape(NL, NN)
    out = np.zeros((NL, NN, 2), np.float32)
    for L in range(NL):
        q, s, zp = IR.load_gptq(ckpt, f"model.layers.{L}.mlp.down_proj")
        n, k = q.shape
        w = (q.astype(np.float32).reshape(n, -1, 32) - zp[..., None]) * IR.bf16_trunc(s)[..., None]
        w = torch.from_numpy(w.reshape(n, k)).bfloat16().float().numpy()   # [1024, 3072]
        cols = w.T * sg[L][:, None]
        cols /= np.maximum(np.linalg.norm(cols, axis=1, keepdims=True), 1e-12)
        out[L] = umap2d(cols, "cosine")
        print(f"  W_down UMAP L{L}", flush=True)
    np.save(W / "wdown_yz.npy", out)


def cmd_embed(a, W):
    """bge-small-en-v1.5 embeddings of each R6 label's 'name. description' (CLS pooling, unit norm)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    labels = json.loads((ROOT / "atlas" / "r6" / "labels.json").read_text())
    keys = list(labels)
    texts = [f'{labels[k]["name"]}. {labels[k]["description"]}' for k in keys]
    tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    m = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            b = tok(texts[i:i + 64], padding=True, truncation=True, max_length=128, return_tensors="pt")
            e = m(**b).last_hidden_state[:, 0]
            embs.append(torch.nn.functional.normalize(e, dim=-1).numpy())
    np.savez(W / "label_emb.npz", keys=np.array(keys), emb=np.concatenate(embs).astype(np.float32))
    print(f"embedded {len(keys)} labels", flush=True)


def knn_same(xy, k):
    """Indices of the k nearest other points (2D) for every point of one group."""
    d = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1)[:, :k]


def boot(v, n=10000, seed=SEED):
    """Mean and 95% bootstrap CI; resamples in small chunks so memory stays flat for long vectors."""
    rng = np.random.default_rng(seed)
    v = np.asarray(v, np.float64)
    m = np.concatenate([v[rng.integers(0, len(v), (min(100, n - i), len(v)), dtype=np.int32)].mean(1)
                        for i in range(0, n, 100)])
    return float(v.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def cmd_score(a, W):
    E = np.load(W / "label_emb.npz"); keys = list(E["keys"]); emb = E["emb"]
    labels = json.loads((ROOT / "atlas" / "r6" / "labels.json").read_text())
    layer = np.array([labels[k]["layer"] for k in keys]); neuron = np.array([labels[k]["neuron"] for k in keys])
    coact = np.load(W / "coact_yz.npy").reshape(NL, NN, 2)
    wdown = np.load(W / "wdown_yz.npy")
    res = {"k": K, "min_labelled": MIN_LABELLED, "layers": {}}
    diffs, agree = [], {"coact": [], "wdown": []}
    for L in range(NL):
        ix = np.nonzero(layer == L)[0]
        if len(ix) < MIN_LABELLED:
            continue
        e = emb[ix]; cos = e @ e.T
        base = (cos.sum(1) - 1) / (len(ix) - 1)                     # mean cos to the other labelled poles
        per = {}
        for name, lay in (("coact", coact), ("wdown", wdown)):
            nb = knn_same(lay[L, neuron[ix]], K)
            per[name] = np.take_along_axis(cos, nb, 1).mean(1) - base
            agree[name] += per[name].tolist()
        d = per["coact"] - per["wdown"]; diffs += d.tolist()
        res["layers"][f"L{L}"] = {"n": int(len(ix)), "coact": float(per["coact"].mean()),
                                  "wdown": float(per["wdown"].mean()), "diff": float(d.mean())}
    m, lo, hi = boot(diffs)
    res["label_agreement"] = {"n": len(diffs), "coact": float(np.mean(agree["coact"])),
                              "wdown": float(np.mean(agree["wdown"])),
                              "diff": {"mean": m, "ci95": [lo, hi]}, "pass": lo > 0}
    print(json.dumps(res["label_agreement"]), flush=True)
    # cross-layer agreement (co-activation only): neighbours among labelled poles of OTHER layers
    xy = coact[layer, neuron]; cos = emb @ emb.T
    d2 = ((xy[:, None] - xy[None]) ** 2).sum(-1)
    d2[layer[:, None] == layer[None, :]] = np.inf
    nb = np.argsort(d2, 1)[:, :K]
    other = layer[:, None] != layer[None, :]
    rand = (cos * other).sum(1) / other.sum(1)
    cl = np.take_along_axis(cos, nb, 1).mean(1) - rand
    res["cross_layer_coact"] = dict(zip(("mean", "lo", "hi"), boot(cl)))
    print("cross-layer:", res["cross_layer_coact"], flush=True)
    # neighbour co-activation on held-out positions (no labels): every neuron, both layouts
    H = np.load(W / "heldout_acts.npy", mmap_mode="r")
    co = {"coact": [], "wdown": []}; rnd = []
    for L in range(NL):
        sl = slice(L * NN, (L + 1) * NN)
        Z = np.asarray(H[:, sl], np.float32)
        Z -= Z.mean(0)
        nz = np.linalg.norm(Z, axis=0)
        live = nz > 1e-6
        Z /= np.maximum(nz, 1e-12)
        corr = Z.T @ Z                                                 # [3072, 3072] held-out Pearson r
        rnd += ((corr.sum(1) - 1) / (NN - 1))[live].tolist()
        for name, lay in (("coact", coact), ("wdown", wdown)):
            nb = knn_same(lay[L], K)
            co[name] += np.take_along_axis(corr, nb, 1).mean(1)[live].tolist()
    res["neighbour_coactivation"] = {"coact": float(np.mean(co["coact"])), "wdown": float(np.mean(co["wdown"])),
                                     "random_same_layer": float(np.mean(rnd)),
                                     "diff": dict(zip(("mean", "lo", "hi"),
                                                      boot(np.array(co["coact"]) - np.array(co["wdown"]))))}
    print("neighbour co-activation:", json.dumps(res["neighbour_coactivation"]), flush=True)
    json.dump(res, open(W / "score.json", "w"), indent=1)


def norm11(v):
    lo, hi = v.min(0), v.max(0)
    return (2 * (v - lo) / np.maximum(hi - lo, 1e-9) - 1).astype(np.float16)


def cmd_export(a, W):
    """Compact files for the UI: [28, 3072, 2] float16 (y, z) in [-1, 1], neuron id = layer * 3072 + n."""
    OUT.mkdir(parents=True, exist_ok=True)
    coact = norm11(np.load(W / "coact_yz.npy")).reshape(NL, NN, 2)
    wdown = np.stack([norm11(x) for x in np.load(W / "wdown_yz.npy")])
    coact.tofile(OUT / "coact_yz.f16"); wdown.tofile(OUT / "wdown_yz.f16")
    score = json.load(open(W / "score.json"))
    pca = json.load(open(W / "pca64.json"))
    meta = {"files": {"coact_yz.f16": "the layout: co-activation embedding, shared across layers",
                      "wdown_yz.f16": "baseline: per-layer UMAP of W_down columns (axes not shared across layers)"},
            "format": "raw little-endian float16, shape [28, 3072, 2] = (layer, neuron, (y, z)), each axis in "
                      "[-1, 1]; x = layer",
            "method": {"positions": N_LAYOUT, "source": "R6 dev+test windows (FineWeb-Edu, UltraChat), positions >= 1",
                       "pole_aligned": True, "pca_dims": 64, "pca_explained": round(sum(pca["explained"]), 4),
                       "umap": UMAP_KW | {"metric": "cosine"}},
            "score": score}
    (OUT / "layout.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {OUT}: coact_yz.f16 ({coact.nbytes} B), wdown_yz.f16, layout.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["acts", "pca", "umap", "wdown", "embed", "score", "export", "all"])
    ap.add_argument("--work", default="/code/work/r9")
    ap.add_argument("--evidence-work", default="/code/work/r6", help="R6 work dir with corpus.npz")
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    W = Path(a.work); W.mkdir(parents=True, exist_ok=True)
    stages = ["acts", "pca", "umap", "wdown", "embed", "score", "export"]
    done = {"acts": "heldout_acts.npy", "pca": "pca64.npy", "umap": "coact_yz.npy", "wdown": "wdown_yz.npy",
            "embed": "label_emb.npz", "score": "score.json"}
    if a.stage == "all":                                          # one process per stage keeps memory flat
        for st in stages:
            if st in done and (W / done[st]).exists():
                print(f"{st}: cached", flush=True); continue
            subprocess.run([sys.executable, __file__, st] + sys.argv[2:], check=True)
    else:
        t0 = time.time()
        {"acts": cmd_acts, "pca": cmd_pca, "umap": cmd_umap, "wdown": cmd_wdown, "embed": cmd_embed,
         "score": cmd_score, "export": cmd_export}[a.stage](a, W)
        print(f"{a.stage}: done in {time.time() - t0:.0f}s", flush=True)
