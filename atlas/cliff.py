# R8 safe slider ranges (#14, spec #11 I2): for every neuron and sign, the dose where text breaks (its
# coherence cliff), predicted by a cheap teacher-forced signal, KL(steered || base) along a dose ladder.
#
# Calibration: 100 BH survivors spread over the 28 layers (plus the L19n834 control), both signs. Sampled
# generations at ±{0.25, 0.5, 1, 2, 4} x 2 prompts x 3 seeds (temperature 0.7, the same Gumbel noise for
# every row of a prompt/seed, 48 tokens) are judged coherent/broken by blind Haiku sub-agents; the unsteered
# model's log-prob per token and distinct-2 are cross-checks. Judged cliff = the first ladder dose where at
# least half of a neuron-sign's 6 samples are broken (none -> beyond the ladder). KL cliff = the first dose
# whose teacher-forced KL exceeds kappa. kappa is fit to maximize the share of neuron-signs whose two cliffs
# are within one ladder step (pass: >= 80%; leave-one-out reported too). Then the KL ladder runs for every
# BH survivor and sign, and the slider range per sign is 80% of the KL cliff (interpolated, KL(0) = 0).
# Dose unit: dose x the layer's median residual norm on the 16 R3 set-1 contexts (stored with the ranges).
#
#   python atlas/cliff.py select                                    # calibration neurons
#   python atlas/cliff.py ladder --set calib|survivors --device cuda   # KL table (GPU for survivors)
#   python atlas/cliff.py sample --device cuda                      # calibration generations
#   python atlas/cliff.py metrics --device cuda                     # log-prob/token, distinct-2
#   python3 atlas/cliff.py judge                                    # Haiku judges (Cloudroom sub-agents)
#   python3 atlas/cliff.py fit                                      # kappa, pass check
#   python3 atlas/cliff.py ranges                                   # slider ranges for all survivors
import argparse, importlib.util, json, os, random, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "atlas" / "cliff"
DOSES = [0.25, 0.5, 1.0, 2.0, 4.0]
SIGNS = [1.0, -1.0]                          # index 0 = + pole, 1 = - pole
PROMPTS = {"story": "Tell me a short story.", "sunday": "Describe your ideal Sunday in three sentences."}
SEEDS, GEN, TEMP, N_CTX = 3, 48, 0.7, 16
CONTROLS = [(8, 908), (19, 834)]


def load_sc(device):
    import torch
    torch.set_grad_enabled(False)
    if device == "cpu":
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    spec = importlib.util.spec_from_file_location("screen", ROOT / "atlas" / "screen.py")
    SC = importlib.util.module_from_spec(spec); spec.loader.exec_module(SC)
    return SC, SC.Screener(variant="int4-onnx", device=device)


def survivors():
    sw = np.load(ROOT / "atlas" / "sweep_run.npz", allow_pickle=True)
    p = np.concatenate([sw[f"L{L}/p"] for L in range(28)])
    ps = np.sort(p); k = np.arange(1, len(ps) + 1); ok = ps <= 0.05 * k / len(ps)
    thr = ps[np.max(np.nonzero(ok)[0])]
    return [(int(i // 3072), int(i % 3072)) for i in np.nonzero(p <= thr)[0]]


def cmd_select(a):
    rng = np.random.default_rng(8)
    surv = survivors(); by = {}
    for L, n in surv:
        by.setdefault(L, []).append(n)
    picks = []
    for L in range(28):                                            # 4 per layer for L0-15, 3 for L16-27
        ctrl = [n for (cl, n) in CONTROLS if cl == L]               # controls ride along (n834 is no survivor)
        pool = [n for n in by.get(L, []) if n not in ctrl]
        k = (4 if L < 16 else 3) - sum(n in by.get(L, []) for n in ctrl)
        picks += [(L, n) for n in ctrl] + [(L, int(n)) for n in rng.choice(pool, k, replace=False)]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calib.json").write_text(json.dumps([{"layer": L, "neuron": n} for L, n in picks]))
    print(f"{len(picks)} calibration neurons, {len(surv)} survivors")


def units_for(a):
    if a.set == "calib":
        return [(u["layer"], u["neuron"]) for u in json.loads((OUT / "calib.json").read_text())]
    return survivors()


def contexts(SC, scr):
    import torch
    ids = torch.from_numpy(np.load(ROOT / "atlas" / "pilot_contexts.npz")["ids"][:N_CTX].astype(np.int64))
    return scr.base(ids)


def cmd_ladder(a):
    """KL(steered || base), mean over contexts and positions >= 1, for each unit x sign x dose."""
    import torch, torch.nn.functional as F
    SC, scr = load_sc(a.device)
    base = contexts(SC, scr)
    med = [base["med"][L] for L in range(28)]
    units = units_for(a)
    if a.limit:
        units = units[:a.limit]
    by = {}
    for i, (L, n) in enumerate(units):
        by.setdefault(L, []).append((i, n))
    kl = np.full((len(units), 2, len(DOSES)), np.nan, np.float32)
    lpb = base["logp_base"]                                       # [B,T,V]
    B, T, V = lpb.shape
    t0 = time.time(); done = 0
    for L in sorted(by):
        pos_emb, mask, position_ids = scr._ctx(base["ids"].repeat(a.ubatch, 1))
        scr._attach(L)
        jobs = [(i, n, si, di) for (i, n) in by[L] for si in range(2) for di in range(len(DOSES))]
        h_in = base["h_in"][L]
        for j0 in range(0, len(jobs), a.ubatch):
            chunk = jobs[j0:j0 + a.ubatch]; u = len(chunk)
            push = torch.zeros(u, 3072, device=scr.device)
            for r, (i, n, si, di) in enumerate(chunk):
                push[r, n] = SIGNS[si] * DOSES[di] * med[L] / scr.colnorm[L][n].clamp_min(1e-20)
            if u < a.ubatch:
                pos_emb, mask, position_ids = scr._ctx(base["ids"].repeat(u, 1))
            h = h_in.unsqueeze(0).expand(u, B, T, h_in.shape[-1]).reshape(u * B, T, -1)
            scr._push = push[:, None, None, :].expand(u, B, 1, 3072).reshape(u * B, 1, 3072)
            lp = F.log_softmax(scr._rerun_from(L, h, pos_emb, mask, position_ids), dim=-1).view(u, B, T, -1)
            k = (lp.exp() * (lp - lpb.unsqueeze(0))).sum(-1)[:, :, 1:].mean(dim=(1, 2))   # [u]
            for r, (i, n, si, di) in enumerate(chunk):
                kl[i, si, di] = float(k[r])
            del lp, k
            if u < a.ubatch:
                pos_emb, mask, position_ids = scr._ctx(base["ids"].repeat(a.ubatch, 1))
        scr._push = None
        done += len(by[L])
        print(f"L{L}: {len(by[L])} units, {done}/{len(units)} done, {time.time() - t0:.0f}s", flush=True)
        np.savez(a.out, layer=np.array([u[0] for u in units]), neuron=np.array([u[1] for u in units]),
                 kl=kl, med=np.array(med), doses=np.array(DOSES))
    print(f"wrote {a.out}", flush=True)


def cmd_sample(a):
    """Calibration generations: every calib neuron x sign x dose, plus unsteered rows, for each prompt and
    seed, with shared Gumbel noise per (prompt, seed, step) so rows differ only by the push."""
    import torch
    SC, scr = load_sc(a.device)
    IR = SC.IR
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(IR.torq_file("tokenizer.json"))
    sysp = json.loads((ROOT / "eval" / "prompts.json").read_text())["system"]
    med = [contexts(SC, scr)["med"][L] for L in range(28)]
    from transformers import DynamicCache
    rows = [(None, None, 0.0)] + [(L, n, s * d) for (L, n) in units_for(argparse.Namespace(set="calib"))
                                  for s in SIGNS for d in DOSES]
    if a.limit:
        rows = rows[:a.limit]
    push = None
    hooks = [scr.hf.layers[L].mlp.down_proj.register_forward_pre_hook(
        lambda m, args, L=L: (args[0] + push[L][:, None, :].to(args[0].dtype),) + args[1:]) for L in range(28)]
    lut = torch.from_numpy(scr.ref.lut).to(scr.device)
    out = []
    t0 = time.time()
    for pname, ptext in PROMPTS.items():
        pr = IR.chat_ids(sysp, ptext, tok)
        for seed in range(SEEDS):
            g = torch.Generator(device="cpu").manual_seed(1000 * seed + len(pname))
            noise = [(-torch.log(-torch.log(torch.rand(len(scr.ref.lut), generator=g).clamp(1e-10, 1 - 1e-10))))
                     .to(scr.device) for _ in range(GEN)]
            for r0 in range(0, len(rows), a.batch):
                chunk = rows[r0:r0 + a.batch]; b = len(chunk)
                push = torch.zeros(28, b, 3072, device=scr.device)
                for i, (L, n, dose) in enumerate(chunk):
                    if L is not None:
                        push[L, i, n] = dose * med[L] / scr.colnorm[L][n].clamp_min(1e-20)
                cache = DynamicCache()
                h = scr.hf(input_ids=torch.tensor([pr] * b, device=scr.device), past_key_values=cache,
                           use_cache=True).last_hidden_state
                gen = []
                for t in range(GEN):
                    logits = scr.ref.head(h[:, -1]).float()
                    nxt = lut[(logits / TEMP + noise[t]).argmax(-1)]
                    gen.append(nxt)
                    h = scr.hf(input_ids=nxt[:, None], past_key_values=cache, use_cache=True).last_hidden_state
                gen = torch.stack(gen, 1).tolist()
                for (L, n, dose), ids in zip(chunk, gen):
                    stop = next((i for i, t in enumerate(ids) if t in IR.STOP), len(ids))
                    out.append({"layer": L, "neuron": n, "dose": dose, "prompt": pname, "seed": seed,
                                "ids": ids[:stop], "text": tok.decode(ids[:stop])})
            print(f"[{pname} seed {seed}] {len(out)} samples, {time.time() - t0:.0f}s", flush=True)
    for h in hooks:
        h.remove()
    for i, s in enumerate(out):
        s["id"] = f"s{i}"
    Path(a.out).write_text(json.dumps({"med": med, "samples": out}))
    print(f"wrote {a.out}: {len(out)} samples", flush=True)


def cmd_metrics(a):
    """Unsteered log-prob per generated token (teacher-forced on prompt + text) and distinct-2."""
    import torch, torch.nn.functional as F
    SC, scr = load_sc(a.device)
    IR = SC.IR
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(IR.torq_file("tokenizer.json"))
    sysp = json.loads((ROOT / "eval" / "prompts.json").read_text())["system"]
    inv = {}
    for i, t in enumerate(scr.ref.lut[:SC.N_KEPT].tolist()):
        inv.setdefault(t, i)
    S = json.loads(Path(a.samples).read_text())
    prompt_ids = {p: IR.chat_ids(sysp, t, tok) for p, t in PROMPTS.items()}
    for s in S["samples"]:
        ids = s["ids"]
        bigrams = list(zip(ids, ids[1:]))
        s["distinct2"] = len(set(bigrams)) / max(len(bigrams), 1)
        if not ids:
            s["logp"] = None; continue
        pr = prompt_ids[s["prompt"]]
        x = torch.tensor([pr + ids], device=scr.device)
        h = scr.hf(input_ids=x, use_cache=False).last_hidden_state[0]
        lp = F.log_softmax(scr.ref.head(h).float()[:, :SC.N_KEPT], -1)
        tgt = [inv.get(t) for t in ids]
        vals = [float(lp[len(pr) - 1 + j, ti]) for j, ti in enumerate(tgt) if ti is not None]
        s["logp"] = float(np.mean(vals)) if vals else None
    Path(a.samples).write_text(json.dumps(S))
    print(f"metrics added to {a.samples}", flush=True)


JUDGE_HEAD = """You check short text samples from a small language model. Answer directly: do not use any tools, files or commands.

For EACH numbered sample, answer 1 if it is coherent and 0 if it is broken.
- coherent (1): readable natural language a person could have written. Odd, off-topic, unfinished, cut off or bland text still counts as coherent.
- broken (0): gibberish; a word, token or phrase repeating in a loop; strings of symbols or code fragments with no meaning; words glued or split wrongly; mostly non-English noise.

Output ONLY one JSON object mapping each sample id to 0 or 1, covering all {n} ids. No prose, no code fences.

"""


def cmd_judge(a):
    """Blind Haiku judges via the R6 fan-out engine (own sandboxes, ramped). 100 shuffled samples per job;
    10% of samples are judged twice, in different jobs, to measure judge agreement."""
    spec = importlib.util.spec_from_file_location("label_fanout", ROOT / "atlas" / "label_fanout.py")
    lf = importlib.util.module_from_spec(spec); spec.loader.exec_module(lf)
    S = json.loads(Path(a.samples).read_text())["samples"]
    text = {s["id"]: s["text"].replace("\n", " ⏎ ") for s in S}
    out_p = OUT / "judgments.json"
    judged = lf.load_json(out_p, {})
    items_p = OUT / "judge_items.json"                              # opaque id -> sample id (duplicates hidden)
    if items_p.exists():
        items = json.loads(items_p.read_text())
    else:
        rng = random.Random(14)
        ids = [s["id"] for s in S]
        src = ids + rng.sample(ids, len(ids) // 10)
        tags = rng.sample(range(10 ** 5, 10 ** 6), len(src))
        items = {f"t{t}": i for t, i in zip(tags, src)}
        items_p.write_text(json.dumps(items))
    order = list(items); random.Random(15).shuffle(order)

    def make_prompt(ks, attempt):
        return JUDGE_HEAD.format(n=len(ks)) + "\n".join(f"{k}: {text[items[k]][:600]}" for k in ks)

    def accept(ks, reply, attempt):
        d = lf.lr.parse_json_reply(reply) or {}
        good, bad = {}, []
        for k in ks:
            v = d.get(k)
            (good.__setitem__(k, int(v)) if v in (0, 1, "0", "1") else bad.append(k))
        judged.update(good); lf.save_json(out_p, judged)
        return good, bad

    a.dir = OUT; a.planner = "r8"
    fl = lf.Fleet(a, "judge", make_prompt, accept)
    fl.log(f"queued {lf.queue(fl, [k for k in order if k not in judged], a.batch)} samples")
    fl.run()


def ladder_index(vals, thr):
    """First ladder index whose value exceeds thr (len(DOSES) = beyond the ladder)."""
    for i, v in enumerate(vals):
        if v > thr:
            return i
    return len(DOSES)


def cmd_fit(a):
    S = json.loads(Path(a.samples).read_text())["samples"]
    items = json.loads((OUT / "judge_items.json").read_text())
    raw = json.loads((OUT / "judgments.json").read_text())
    J, dup = {}, []                                                 # first judgment per sample; repeats -> agreement
    for t, v in raw.items():
        sid = items[t]
        (dup.append((J[sid], v)) if sid in J else J.__setitem__(sid, v))
    lad = np.load(OUT / "ladder_calib.npz")
    pos = {(int(L), int(n)): i for i, (L, n) in enumerate(zip(lad["layer"], lad["neuron"]))}
    broken = {}
    for s in S:
        if s["layer"] is None or s["id"] not in J:
            continue
        si = 0 if s["dose"] > 0 else 1
        di = DOSES.index(abs(s["dose"]))
        broken.setdefault((s["layer"], s["neuron"], si), np.zeros((len(DOSES), 2)))[di] += [1 - J[s["id"]], 1]
    keys = sorted(broken)
    frac = np.array([broken[k][:, 0] / np.maximum(broken[k][:, 1], 1) for k in keys])      # [N, 5]
    first = lambda m: np.where(m.any(-1), m.argmax(-1), len(DOSES))                        # first True, else beyond
    judged = first(frac >= 0.5)
    kl = np.array([lad["kl"][pos[(k[0], k[1])], k[2]] for k in keys])                     # [N, 5]
    cands = np.unique(np.concatenate([kl.ravel(), [np.inf]]))
    pred = first(kl[None, :, :] > cands[:, None, None])                                   # [C, N]
    ok = np.abs(pred - judged[None]) <= 1
    tot = ok.sum(1)
    best = int(np.argmax(tot)); kappa = float(cands[best])
    loo = [ok[int(np.argmax(tot - ok[:, i])), i] for i in range(len(keys))]               # refit without pole i
    ins, cv = float(ok[best].mean()), float(np.mean(loo))
    unst = [s for s in S if s["layer"] is None]
    lab = [(1 - J[s["id"]], s.get("logp"), s["distinct2"]) for s in S if s["id"] in J and s.get("logp") is not None]
    br = np.array([x[0] for x in lab]); lp = np.array([x[1] for x in lab]); d2 = np.array([x[2] for x in lab])
    res = {"n_neuron_signs": len(keys), "kappa": kappa, "in_sample": ins, "leave_one_out": cv,
           "pass": ins >= 0.8 and cv >= 0.8,
           "exact_step": float((pred[best] == judged).mean()),
           "judge_agreement_on_repeats": float(np.mean([x == y for x, y in dup])) if dup else None,
           "n_repeats": len(dup),
           "judged_cliff_hist": np.bincount(judged, minlength=len(DOSES) + 1).tolist(),
           "kl_cliff_hist": np.bincount(pred[best], minlength=len(DOSES) + 1).tolist(),
           "unsteered_broken_rate": float(np.mean([1 - J[s["id"]] for s in unst if s["id"] in J])),
           "metrics_vs_judge": {"broken_rate": float(br.mean()),
                                "logp_coherent": float(lp[br == 0].mean()), "logp_broken": float(lp[br == 1].mean()),
                                "distinct2_coherent": float(d2[br == 0].mean()),
                                "distinct2_broken": float(d2[br == 1].mean())}}
    (OUT / "calib_results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def cliff_dose(kl, kappa):
    """Dose where KL crosses kappa, by linear interpolation along the ladder with KL(0) = 0."""
    xs = [0.0] + DOSES; ys = [0.0] + list(kl)
    for i in range(1, len(xs)):
        if ys[i] > kappa:
            y0, y1 = ys[i - 1], ys[i]
            return xs[i - 1] + (xs[i] - xs[i - 1]) * (kappa - y0) / max(y1 - y0, 1e-12)
    return None                                                   # no cliff inside the ladder


def cmd_ranges(a):
    res = json.loads((OUT / "calib_results.json").read_text()); kappa = res["kappa"]
    lad = np.load(OUT / "ladder_survivors.npz")
    rng = np.full((28, 3072, 2), np.nan, np.float32)               # [layer, neuron, (+, -)] max safe |dose|
    beyond = 0
    for i, (L, n) in enumerate(zip(lad["layer"], lad["neuron"])):
        for si in range(2):
            c = cliff_dose(lad["kl"][i, si], kappa)
            beyond += c is None
            rng[L, n, si] = 0.8 * (c if c is not None else DOSES[-1])
    rng.astype(np.float16).tofile(OUT / "ranges.f16")
    meta = {"format": "raw little-endian float16 [28, 3072, 2] = (layer, neuron, (+ pole, - pole)): the largest safe "
                      "|dose| per sign, in units of the layer's median residual norm; NaN = not a BH survivor",
            "rule": "80% of the dose where teacher-forced KL(steered || base) crosses kappa (linear interpolation "
                    "along the ladder, KL(0) = 0); capped at 80% of 4 when KL stays under kappa",
            "kappa": kappa, "ladder": DOSES, "median_resid_norm": [float(x) for x in lad["med"]],
            "survivors": int(len(lad["layer"])), "signs_without_cliff_in_ladder": int(beyond),
            "calibration": res}
    (OUT / "ranges.json").write_text(json.dumps(meta, indent=1))
    v = rng[~np.isnan(rng)]
    print(f"ranges for {len(lad['layer'])} survivors x 2 signs: median {np.median(v):.2f}, p10 {np.quantile(v, .1):.2f}, "
          f"p90 {np.quantile(v, .9):.2f}; {beyond} signs never cross kappa in the ladder")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select")
    s = sub.add_parser("ladder"); s.add_argument("--set", choices=["calib", "survivors"], required=True)
    s.add_argument("--device", default="cpu"); s.add_argument("--ubatch", type=int, default=4)
    s.add_argument("--limit", type=int, default=0); s.add_argument("--out", required=True)
    s = sub.add_parser("sample"); s.add_argument("--device", default="cpu"); s.add_argument("--batch", type=int, default=64)
    s.add_argument("--limit", type=int, default=0); s.add_argument("--out", required=True)
    s = sub.add_parser("metrics"); s.add_argument("--device", default="cpu"); s.add_argument("--samples", required=True)
    s = sub.add_parser("judge"); s.add_argument("--samples", required=True); s.add_argument("--batch", type=int, default=100)
    for k, v in (("model", "claude-haiku-5-5"), ("reasoning", "high")):
        s.add_argument(f"--{k}", default=v)
    for k, v in (("max_live", 8), ("ramp_to", 25), ("ramp_after", 8), ("poll", 30), ("timeout", 40), ("budget", 0)):
        s.add_argument(f"--{k.replace('_', '-')}", type=int, default=v)
    s = sub.add_parser("fit"); s.add_argument("--samples", required=True)
    sub.add_parser("ranges")
    a = ap.parse_args()
    {"select": cmd_select, "ladder": cmd_ladder, "sample": cmd_sample, "metrics": cmd_metrics, "judge": cmd_judge,
     "fit": cmd_fit, "ranges": cmd_ranges}[a.cmd](a)
