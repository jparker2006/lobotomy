# R6 evidence for the top-N poles by effect size S plus the two positive controls (#13), streamed so
# memory stays flat (~3 GB, guarded at 5). Per pole: 15 top-activating 64-token snippets with every
# token's activation binned 0-10 (Bills 2023), 10 logit-lens + 10 steer-lens words, 2 steered samples
# at the pole's 0.5 dose, and a token-lookup table (the word-list and lookup baselines). Plus two
# held-out sets (dev for QC and picking between labels, test for the final E4 score). Labelling, dev
# and test windows come from one shuffled stream split by document, so they never share text, and
# the labelling stream is the first orchestrator's (seed 3), so its labels are scored on unseen text
# too. Stages cache to --work (outside git) and resume; `assemble` writes atlas/r6/*.json.gz.
#
#   OMP_NUM_THREADS=2 python atlas/build_evidence.py all --top 3000 --work /code/work/r6
import argparse, ctypes, gzip, importlib.util, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parent.parent
torch.set_grad_enabled(False)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
spec = importlib.util.spec_from_file_location("screen", ROOT / "atlas" / "screen.py")
SC = importlib.util.module_from_spec(spec); spec.loader.exec_module(SC)
IR = SC.IR
CONTROLS = [(8, 908), (19, 834)]
K, T, GEN, DOSE = 15, 64, 32, 0.5            # snippets per pole, window tokens, sample tokens, dose
SETS = ("label", "dev", "test")
SPLIT = {"web": (4000, 1000, 1000), "chat": (2000, 500, 500)}   # windows per set
PROMPTS = {"story": "Tell me a short story.", "sunday": "Describe your ideal Sunday in three sentences."}
MAX_RSS_GB = 5.0


def guard(msg):
    rss = int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) / 1e6
    print(f"{msg}  [rss {rss:.2f} GB]", flush=True)
    if rss > MAX_RSS_GB:
        raise SystemExit(f"rss {rss:.2f} GB is over the {MAX_RSS_GB} GB budget")


def screener():
    scr = SC.Screener(variant="int4-onnx", device="cpu")
    ctypes.CDLL("libc.so.6").malloc_trim(0)                   # hand the load-time peak back to the OS
    guard("model loaded")
    return scr


def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(IR.torq_file("tokenizer.json"))


def pole_list(top):
    """Top-`top` poles by S (one per neuron, its winning sign) plus the controls, best first."""
    sw = np.load(ROOT / "atlas" / "sweep_run.npz", allow_pickle=True)
    allp = sorted(((float(sw[f"L{L}/S_neu"][n]), L, n) for L in range(SC.LAYERS) for n in range(3072)),
                  reverse=True)
    rank = {(L, n): r for r, (_, L, n) in enumerate(allp)}
    keep = [(L, n) for _, L, n in allp[:top]] + [c for c in CONTROLS if rank[c] >= top]
    return [{"key": f"L{L}n{n}", "layer": L, "neuron": n, "sign": int(sw[f"L{L}/sign"][n]),
             "S": round(float(sw[f"L{L}/S_neu"][n]), 3), "p": float(sw[f"L{L}/p"][n]), "rank": rank[(L, n)],
             "is_control": (L, n) in CONTROLS, "lens_ids": sw[f"L{L}/lens"][n].tolist()} for L, n in keep]


def cmd_corpus(a, W):
    from datasets import load_dataset
    tok = tokenizer()
    sysp = json.loads((ROOT / "eval" / "prompts.json").read_text())["system"]
    out = {s: {"ids": [], "kind": [], "doc": []} for s in SETS}
    srcs = {"web": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train"),
            "chat": ("HuggingFaceH4/ultrachat_200k", None, "train_sft")}
    for kind, (name, cfg, split) in srcs.items():
        ds = load_dataset(name, cfg, split=split, streaming=True).shuffle(seed=3, buffer_size=3000)
        si, have = 0, 0
        for doc, row in enumerate(ds):
            if kind == "web":
                ids = tok.encode(row["text"]).ids
                wins = [ids[s:s + T] for s in range(32, min(len(ids) - T, 32 + T * 5), T)]
            else:
                msg = row["messages"][1]["content"] if len(row.get("messages", [])) > 1 else ""
                ids = IR.chat_ids(sysp, row["prompt"].strip(), tok) + tok.encode(" " + msg).ids
                wins = [ids[:T]] if len(ids) >= T else []
            for w in wins:                      # a document's windows all go to one set
                d = out[SETS[si]]; d["ids"].append(w); d["kind"].append(kind); d["doc"].append(f"{kind}{doc}")
            have += len(wins)
            if have >= max(1, int(SPLIT[kind][si] * a.frac)):
                print(f"  {kind} {SETS[si]}: {have} windows (stream doc {doc})", flush=True)
                si, have = si + 1, 0
                if si == len(SETS):
                    break
    np.savez(W / "corpus.npz", **{f"{s}_{f}": np.array(out[s][f]) for s in SETS for f in ("ids", "kind", "doc")})
    print("corpus: " + ", ".join(f"{s} {len(out[s]['ids'])}" for s in SETS), flush=True)


def pole_acts(scr, ids, Lt, Nt, Sg):
    """[B,T] windows -> pole-direction activations [B,T,P] (float32), position 0 (sink) = -inf."""
    scr.hf(input_ids=ids, use_cache=False)                    # the Ref's hooks keep every layer's input
    A = torch.stack(scr.ref.acts, dim=2)                      # [B,T,28,3072] bf16
    sel = A[:, :, Lt, Nt].float() * Sg
    sel[:, 0] = float("-inf")
    return sel


def cmd_acts(a, W, P):
    scr = screener()
    C = np.load(W / "corpus.npz")
    Lt = torch.tensor([p["layer"] for p in P]); Nt = torch.tensor([p["neuron"] for p in P])
    Sg = torch.tensor([float(p["sign"]) for p in P])
    n = len(P)
    # logit lens: the words each pole's W_down column promotes through the final norm + head (weights only)
    tok, lut = tokenizer(), np.load(IR.torq_file("token_id_lut.npy")).astype(np.int64)
    dt = next(scr.hf.norm.parameters()).dtype
    lens = []
    for i in range(0, n, 128):
        cols = torch.stack([scr.Wdown[p["layer"]][:, p["neuron"]] * p["sign"] for p in P[i:i + 128]])
        lg = scr.ref.head(scr.hf.norm(cols.to(dt))).float()[:, :SC.N_KEPT]
        lens += lg.topk(10, dim=-1).indices.tolist()
    np.save(W / "logit_lens.npy", np.array(lens))
    # labelling windows: running top-K per pole (with each window's per-token acts) + token-lookup sums
    ids = torch.from_numpy(C["label_ids"].astype(np.int64))
    uniq = np.unique(C["label_ids"][:, 1:])
    top_v = torch.full((n, K), float("-inf")); top_w = torch.zeros((n, K), dtype=torch.long)
    top_a = torch.zeros((n, K, T), dtype=torch.float16)
    sums = torch.zeros((n, len(uniq))); uniq_t = torch.from_numpy(uniq)
    t0 = time.time()
    for b0 in range(0, len(ids), a.batch):
        x = ids[b0:b0 + a.batch]; B = x.shape[0]
        sel = pole_acts(scr, x, Lt, Nt, Sg)                    # [B,T,P]
        pk = sel.max(1).values.T                               # [P,B] window peaks
        cv = torch.cat([top_v, pk], 1)
        cw = torch.cat([top_w, torch.arange(b0, b0 + B).expand(n, B)], 1)
        ca = torch.cat([top_a, sel.permute(2, 0, 1).half()], 1)
        top_v, j = cv.topk(K, dim=1)
        top_w = cw.gather(1, j); top_a = ca.gather(1, j[..., None].expand(n, K, T))
        tix = torch.from_numpy(np.searchsorted(uniq, x[:, 1:].reshape(-1).numpy()))
        sums.index_add_(1, tix, sel[:, 1:].reshape(-1, n).T)
        if (b0 // a.batch) % 50 == 0:
            guard(f"  label {b0 + B}/{len(ids)} windows, {time.time() - t0:.0f}s")
    counts = np.bincount(np.searchsorted(uniq, C["label_ids"][:, 1:].reshape(-1)), minlength=len(uniq))
    np.savez(W / "label_top.npz", top_v=top_v.numpy(), top_w=top_w.numpy(), top_a=top_a.numpy())
    np.savez(W / "lookup.npz", uniq=uniq, counts=counts, sums=sums.numpy())
    del sums, top_a
    # held-out windows: each pole's peak per window
    for s in ("dev", "test"):
        hid = torch.from_numpy(C[f"{s}_ids"].astype(np.int64))
        peak = np.zeros((len(hid), n), np.float32)
        for b0 in range(0, len(hid), a.batch):
            peak[b0:b0 + a.batch] = pole_acts(scr, hid[b0:b0 + a.batch], Lt, Nt, Sg).max(1).values.numpy()
        np.save(W / f"{s}_peak.npy", peak)
        guard(f"  {s}: {len(hid)} windows done")


def cmd_steer(a, W, P):
    """Greedy samples with each pole pushed at sign*0.5 (median residual norm units) on every position,
    batched with a KV cache; pushes for all 28 layers ride in one tensor (zeros elsewhere). --shard i/n
    takes every n-th pole (so helper sandboxes can split the work); --budget stops after that many
    seconds, saving progress, and a rerun resumes. Prints STEER DONE when the shard is complete."""
    from transformers import DynamicCache
    i_sh, n_sh = (int(x) for x in a.shard.split("/"))
    mine = P[i_sh::n_sh]
    outf = W / ("steer.json" if n_sh == 1 else f"steer_{i_sh}.json")
    out = json.loads(outf.read_text()) if outf.exists() else {"_baseline": {}}
    todo = {nm: [p for p in mine if nm not in out.get(p["key"], {})] for nm in PROMPTS}
    if not any(todo.values()):
        print(f"STEER DONE {outf.name}: {len(mine)} poles", flush=True); return
    scr = screener()
    tok = tokenizer()
    sysp = json.loads((ROOT / "eval" / "prompts.json").read_text())["system"]
    push = None
    hooks = [scr.hf.layers[L].mlp.down_proj.register_forward_pre_hook(
        lambda m, args, L=L: (args[0] + push[L][:, None, :].to(args[0].dtype),) + args[1:])
        for L in range(SC.LAYERS)]

    def greedy(pr, B):
        cache = DynamicCache()
        h = scr.hf(input_ids=torch.tensor([pr] * B), past_key_values=cache, use_cache=True).last_hidden_state
        gen, done = [], torch.zeros(B, dtype=torch.bool)
        for _ in range(GEN):
            nxt = torch.from_numpy(scr.ref.lut[scr.ref.head(h[:, -1]).float().argmax(-1).numpy()])
            gen.append(nxt)
            done |= (nxt == 151643) | (nxt == 151645)
            if done.all():
                break
            h = scr.hf(input_ids=nxt[:, None], past_key_values=cache, use_cache=True).last_hidden_state
        rows = torch.stack(gen, 1).tolist()
        return [tok.decode(r[:next((i for i, t in enumerate(r) if t in IR.STOP), len(r))]) for r in rows]

    t0 = time.time()
    for name, text in PROMPTS.items():
        pr = IR.chat_ids(sysp, text, tok)
        push = torch.zeros(SC.LAYERS, 1, 3072)
        med = scr.base(torch.tensor([pr]))["med"]
        out["_baseline"][name] = greedy(pr, 1)[0]
        for b0 in range(0, len(todo[name]), a.gen_batch):
            chunk = todo[name][b0:b0 + a.gen_batch]
            push = torch.zeros(SC.LAYERS, len(chunk), 3072)
            for i, p in enumerate(chunk):
                L, nn_ = p["layer"], p["neuron"]
                push[L, i, nn_] = p["sign"] * DOSE * med[L] / scr.colnorm[L][nn_].clamp_min(1e-20)
            for p, smp in zip(chunk, greedy(pr, len(chunk))):
                out.setdefault(p["key"], {})[name] = smp
            tmp = outf.with_suffix(".tmp"); tmp.write_text(json.dumps(out)); tmp.replace(outf)
            guard(f"  [{name}] {b0 + len(chunk)}/{len(todo[name])} poles, {time.time() - t0:.0f}s")
            if a.budget and time.time() - t0 > a.budget:
                print(f"STEER PARTIAL {outf.name}: rerun the same command to continue", flush=True); return
    for h in hooks:
        h.remove()
    print(f"STEER DONE {outf.name}: {len(mine)} poles", flush=True)


def byte_decoder():
    """GPT-2 byte-level BPE alphabet -> bytes (Qwen's tokenizer.json uses it)."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, k = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + k); k += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def units(tok, BD, ids):
    """Tokens -> printable units; a multi-token UTF-8 character stays whole. Each unit: (text, token idxs)."""
    out, buf, idx = [], b"", []
    for i, t in enumerate(ids):
        s = tok.id_to_token(int(t))
        buf += s.encode() if int(t) >= 151643 else bytes(BD.get(c, 63) for c in s)
        idx.append(i)
        try:
            txt = buf.decode("utf-8")
        except UnicodeDecodeError:
            if len(buf) < 8 and i < len(ids) - 1:
                continue
            txt = buf.decode("utf-8", "replace")
        out.append((txt.replace("\n", "⏎").replace("\t", "⇥"), idx)); buf, idx = b"", []
    return out


def render(tok, BD, ids, bins=None, peak=None):
    """Window text; with bins, every token at bin >= 2 gets ⟦bin⟧ and the peak is <<marked>>."""
    s = []
    for txt, ix in units(tok, BD, ids):
        if bins is None:
            s.append(txt); continue
        b = max(bins[i] for i in ix)
        s.append((f"<<{txt}>>" if peak in ix else txt) + (f"⟦{b}⟧" if b >= 2 else ""))
    return "".join(s)


def cmd_assemble(a, W, P):
    tok, BD = tokenizer(), byte_decoder()
    lut = np.load(IR.torq_file("token_id_lut.npy")).astype(np.int64)
    dec = lambda i: tok.decode([int(i)]).strip()
    C = np.load(W / "corpus.npz")
    top, lk = np.load(W / "label_top.npz"), np.load(W / "lookup.npz")
    ll = np.load(W / "logit_lens.npy")
    steer = {}
    for f in sorted(W.glob("steer*.json")):                    # steer.json or the helpers' steer_<i>.json
        for k, v in json.loads(f.read_text()).items():
            steer.setdefault(k, {}).update(v)
    uniq, counts, mean = lk["uniq"], lk["counts"], lk["sums"]
    gmean = mean.sum(1) / max(counts.sum(), 1)                 # [P] each pole's mean act per token
    mean /= np.maximum(counts, 1)                              # [P,U] mean pole act per token
    ev, ho = {}, {"note": "Held-out windows for the simulation scorer, never shown to labellers. Per pole "
                          "and set: 10 top-firing + 10 random windows in a shuffled order; true = the pole's "
                          "peak activation in the window (positions >= 1); lookup = Bills-style token-lookup "
                          "prediction (max over the window's tokens of that token's mean activation on the "
                          "labelling text)."}
    for s in ("dev", "test"):
        ho[s] = {"windows": [render(tok, BD, w) for w in C[f"{s}_ids"]], "kind": C[f"{s}_kind"].tolist()}
    ho["poles"] = {}
    peaks = {s: np.load(W / f"{s}_peak.npy") for s in ("dev", "test")}
    hid_ix = {s: np.searchsorted(uniq, C[f"{s}_ids"][:, 1:]) for s in ("dev", "test")}
    hid_ok = {s: uniq[np.minimum(hid_ix[s], len(uniq) - 1)] == C[f"{s}_ids"][:, 1:] for s in ("dev", "test")}
    for pi, p in enumerate(P):
        k = p["key"]
        mx = float(top["top_v"][pi, 0])
        snips = []
        for j in range(K):
            w = int(top["top_w"][pi, j]); acts = top["top_a"][pi, j].astype(np.float32)
            bins = np.clip(np.rint(10 * np.maximum(acts, 0) / max(mx, 1e-6)), 0, 10).astype(int)
            bins[0] = 0
            snips.append({"act": round(float(top["top_v"][pi, j]), 2), "kind": str(C["label_kind"][w]),
                          "text": render(tok, BD, C["label_ids"][w], bins, int(np.argmax(acts)))})
        m = np.where(counts >= 3, mean[pi], -np.inf)
        words = []
        for u in np.argsort(-m):
            wd = dec(uniq[u])
            if wd and wd not in words:
                words.append(wd)
            if len(words) == 10:
                break
        ev[k] = {kk: p[kk] for kk in ("layer", "neuron", "S", "p", "rank", "is_control")}
        ev[k].update({"sign": "+" if p["sign"] > 0 else "-", "max_act": round(mx, 3),
                      "logit_lens": [dec(lut[i]) for i in ll[pi]], "steer_lens": [dec(lut[i]) for i in p["lens_ids"]],
                      "top_tokens": words, "snippets": snips,
                      "steered": [{"prompt": n, "dose": DOSE * p["sign"], "text": steer[k][n]}
                                  for n in PROMPTS if n in steer.get(k, {})]})
        rng = np.random.default_rng(1000 + pi)
        ho["poles"][k] = {}
        for s in ("dev", "test"):
            pk = peaks[s][:, pi]; order = np.argsort(-pk)
            sel = np.concatenate([order[:10], rng.choice(order[10:], 10, replace=False)])
            rng.shuffle(sel)
            lk_pred = np.where(hid_ok[s][sel], mean[pi][np.minimum(hid_ix[s][sel], len(uniq) - 1)], gmean[pi]).max(1)
            ho["poles"][k][s] = {"idx": sel.tolist(), "true": [round(float(v), 3) for v in pk[sel]],
                                 "lookup": [round(float(v), 3) for v in lk_pred]}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"poles": len(P), "top": a.top, "controls": [f"L{L}n{n}" for L, n in CONTROLS],
            "corpus": {s: len(C[f"{s}_ids"]) for s in SETS}, "split": SPLIT, "window_tokens": T,
            "snippets": K, "steer": {"prompts": PROMPTS, "dose": DOSE, "tokens": GEN,
                                     "baseline": steer.get("_baseline", {})},
            "sources": "FineWeb-Edu sample-10BT (ODC-By) and UltraChat 200k (MIT), stream seed 3",
            "format": "snippet text: ⟦b⟧ = that token's activation 0-10 (pole direction, scaled to the pole's "
                      "max; unmarked tokens are 0-1), <<x>> = the window's peak, ⏎ = newline; position 0 "
                      "(attention sink) unscored"}
    with gzip.open(out / "evidence.json.gz", "wt") as f:
        json.dump({"meta": meta, "poles": ev}, f)
    with gzip.open(out / "heldout.json.gz", "wt") as f:
        json.dump(ho, f)
    print(f"wrote {out}/evidence.json.gz and heldout.json.gz: {len(ev)} poles", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["corpus", "acts", "steer", "assemble", "all"])
    ap.add_argument("--top", type=int, default=3000)
    ap.add_argument("--work", default="/code/work/r6")
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--gen-batch", type=int, default=32)
    ap.add_argument("--frac", type=float, default=1.0, help="scale the corpus split (smoke tests)")
    ap.add_argument("--shard", default="0/1", help="steer: i/n takes every n-th pole")
    ap.add_argument("--budget", type=int, default=0, help="steer: stop (resumably) after this many seconds")
    ap.add_argument("--out", default=str(ROOT / "atlas" / "r6"))
    a = ap.parse_args()
    W = Path(a.work); W.mkdir(parents=True, exist_ok=True)
    P = pole_list(a.top)
    print(f"{len(P)} poles (top {a.top} by S + controls)", flush=True)
    done = {"corpus": "corpus.npz", "acts": "test_peak.npy", "steer": "steer.json"}
    if a.stage == "all":                                      # one process per stage keeps memory flat
        for st in ("corpus", "acts", "steer", "assemble"):
            if st in done and (W / done[st]).exists():
                print(f"{st}: cached", flush=True); continue
            subprocess.run([sys.executable, __file__, st] + sys.argv[2:], check=True)
    else:
        t0 = time.time()
        {"corpus": cmd_corpus, "acts": cmd_acts, "steer": cmd_steer, "assemble": cmd_assemble}[a.stage](
            *((a, W) if a.stage == "corpus" else (a, W, P)))
        print(f"{a.stage}: done in {time.time() - t0:.0f}s", flush=True)
