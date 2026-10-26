# R3 atlas screen: push every neuron +/-0.5 and keep those that beat same-layer random
# (column-space) pushes, with FDR held to 5% across all 86,016 (spec #11, map Appendix D).
#
# The statistic is frozen (map Q8): cross-fitted steer-lens lift at dose +/-0.5, symmetric
# 2-fold, S = the larger of the two signs' lifts, on the board's trimmed 99,477-token vocab.
# Numerics come from the INT4 CPU replica ref/int4_ref.py (`npu` mode); this module only adds
# steering (a push vector at a layer's down_proj input) and the lift/p-value math. The same
# code runs on CPU (correctness) and CUDA (the pilot); --device picks.
#
#   python atlas/screen.py selftest                 tiny CPU run; checks the positive control
#   python atlas/screen.py screen --layers 8,19 --nulls 8192 --device cuda --out run.npz
import argparse, importlib.util, json, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
torch.set_grad_enabled(False)


def _load_ref_module():
    spec = importlib.util.spec_from_file_location("int4_ref", ROOT / "ref" / "int4_ref.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


IR = _load_ref_module()
N_KEPT = IR.N_KEPT          # 99477 unique trimmed-vocab ids
LAYERS = IR.LAYERS          # 28
FLOOR = 1e-4                # base-prob floor on steer-lens candidate tokens (2026-10-26 fix, spec #11)


class Screener:
    """Wraps ref.int4_ref.Ref with steering at one layer and the frozen lift statistic.

    A push adds a vector `a` (shape [3072], neuron space) to layer L's down_proj input at every
    position. For neuron n: a = dose*med(L)/||W_down[:,n]|| * e_n. For a column-space null drawn
    from r ~ N(0,I): a = dose*med(L) * r/||W_down r||, so the residual moves by dose*med along
    W_down r/||W_down r|| -- the same displacement size as a neuron push (map P5, Appendix D)."""

    def __init__(self, variant="int4", device="cpu", dtype=None):
        self.device = torch.device(device)
        self.ref = IR.Ref(variant)
        self.model = self.ref.model.to(self.device).eval()
        # the LM head is held on the Ref, not as a model submodule, so .to() above misses it
        if isinstance(self.ref.head, torch.nn.Module):
            self.ref.head = self.ref.head.to(self.device)
        self.hf = self.model.model                      # Qwen3Model
        self.nlayers = len(self.hf.layers)
        self.hidden = self.model.config.hidden_size
        # dequantized down_proj weight per layer [out=1024, in=3072] (bf16, the board's weight),
        # used only for push directions and column norms -- not for the forward numerics.
        self.Wdown, self.colnorm = [], []
        for layer in self.hf.layers:
            dp = layer.mlp.down_proj
            w = self._dequant(dp).to(self.device).float()
            self.Wdown.append(w)
            self.colnorm.append(w.norm(dim=0))          # ||W_down[:,n]|| for every neuron [3072]
        self._push = None                               # [3072] add vector for the active layer
        self._Lhook = None

    @staticmethod
    def _dequant(linear):
        """Reconstruct the dequantized weight [out,in] of an Int4Linear (any mode) or nn.Linear."""
        if hasattr(linear, "weight"):
            return linear.weight.detach()
        if hasattr(linear, "w"):                        # onnx/fp32 modes keep w [N,K]
            return linear.w.detach().float()
        qT, s, nzs = linear.qT.float(), linear.s.float(), linear.nzs.float()  # npu mode
        g, _, n = qT.shape                              # [G,32,N]
        k = g * 32
        q = qT.permute(2, 0, 1).reshape(n, k)           # [N,K] signed int4 values as floats
        scale = s.t().reshape(n, g, 1)                  # [N,G,1]
        zp = -(nzs.t().reshape(n, g, 1)) / scale.clamp_min(1e-30)  # recover zp from nzs = -s*zp
        w = (q.reshape(n, g, 32) - zp) * scale
        return w.reshape(n, k)                          # [N=out, K=in]

    # ---- base pass: logp over the trimmed vocab, and the per-layer capture for reruns ----
    def base(self, ids):
        """ids [B,T] -> dict with logp_base [B,T,N_KEPT], and per layer the input hidden state
        h_in[L] [B,T,H] (to rerun from L) and med(L) the dose unit."""
        ids = ids.to(self.device)
        cap_in, cap_out = {}, {}
        hooks = []
        for L, layer in enumerate(self.hf.layers):
            hooks.append(layer.register_forward_pre_hook(
                lambda mod, args, kwargs, L=L: cap_in.__setitem__(L, args[0].detach()), with_kwargs=True))
            hooks.append(layer.register_forward_hook(
                lambda mod, args, out, L=L: cap_out.__setitem__(
                    L, (out[0] if isinstance(out, tuple) else out).detach())))
        out = self.hf(input_ids=ids, use_cache=False)
        for h in hooks:
            h.remove()
        h_last = out.last_hidden_state
        logp_base = F.log_softmax(self.ref.head(h_last).float()[..., :N_KEPT], dim=-1)
        floor = logp_base.exp().mean(dim=(0, 1)) > FLOOR   # [V] tokens the chip plausibly emits
        med = {}
        for L in range(self.nlayers):
            r = cap_out[L].float()                      # residual leaving layer L
            med[L] = r[:, 1:].norm(dim=-1).median().item()  # positions >=1 (skip attention sink)
        return {"logp_base": logp_base, "floor": floor, "h_in": cap_in, "med": med, "ids": ids}

    def neuron_acts(self, ids):
        """Teacher-forced neuron activations (down_proj inputs) [T,28,3072] for one sequence [T]."""
        ids = ids.to(self.device)
        acts = [None] * self.nlayers
        hooks = [self.hf.layers[L].mlp.down_proj.register_forward_pre_hook(
            lambda m, args, L=L: acts.__setitem__(L, args[0].detach())) for L in range(self.nlayers)]
        self.hf(input_ids=ids.unsqueeze(0), use_cache=False)
        for h in hooks:
            h.remove()
        return torch.stack([a[0] for a in acts], dim=1)   # [T,28,3072]

    def _attach(self, L):
        if self._Lhook is not None:
            self._Lhook.remove()
        dp = self.hf.layers[L].mlp.down_proj

        def pre(mod, args):
            x = args[0]
            if self._push is not None:
                x = x + self._push.to(x.dtype)
            return (x,) + args[1:]
        self._Lhook = dp.register_forward_pre_hook(pre)

    def _rerun_from(self, L, h_in, pos_emb, mask, position_ids):
        """Run layers[L:] + norm + head on a captured hidden state, with the active push."""
        h = h_in
        for i in range(L, self.nlayers):
            h = self.hf.layers[i](h, attention_mask=mask, position_embeddings=pos_emb,
                                  position_ids=position_ids, use_cache=False)
            if isinstance(h, tuple):
                h = h[0]
        h = self.hf.norm(h)
        return self.ref.head(h).float()[..., :N_KEPT]

    def _ctx(self, ids):
        """position embeddings + causal mask for a teacher-forced batch (shared across layers)."""
        B, T = ids.shape
        emb = self.hf.embed_tokens(ids)
        position_ids = torch.arange(T, device=self.device).unsqueeze(0)
        pos_emb = self.hf.rotary_emb(emb, position_ids)
        from transformers.masking_utils import create_causal_mask
        mask = create_causal_mask(config=self.model.config, inputs_embeds=emb,
                                  attention_mask=None, past_key_values=None, position_ids=position_ids)
        return pos_emb, mask, position_ids

    def lifts_for(self, L, h_inL, pushes, ids, logp_base, halfA, floor, ubatch=4):
        """pushes [U,Dn]. Scores `ubatch` units per forward pass (tiled hidden states + per-unit push),
        reducing each chunk to A/B means so peak memory is ~[ubatch*B,T,V]. Returns four tensors: the
        raw (no-floor) lift Sr (primary), the floored lift Sf (reported secondary), KL(steered||base),
        and the top-10 steer-lens token ids per unit [U,10] (by overall mean delta, for the report)."""
        B, T, Hn = h_inL.shape
        U, Dn = pushes.shape
        self._attach(L)
        Sf = torch.empty(U, device=self.device); Sr = torch.empty(U, device=self.device)
        KL = torch.empty(U, device=self.device)
        lens = torch.full((U, 10), -1, dtype=torch.long, device=self.device)
        for u0 in range(0, U, ubatch):
            uc = pushes[u0:u0 + ubatch]; u = uc.shape[0]
            pos_emb, mask, position_ids = self._ctx(ids.repeat(u, 1))   # batch u*B, causal mask
            h = h_inL.unsqueeze(0).expand(u, B, T, Hn).reshape(u * B, T, Hn)
            self._push = uc[:, None, None, :].expand(u, B, 1, Dn).reshape(u * B, 1, Dn)
            lp = F.log_softmax(self._rerun_from(L, h, pos_emb, mask, position_ids), dim=-1).view(u, B, T, -1)
            d = lp - logp_base.unsqueeze(0)             # [u,B,T,V]
            A = d[:, halfA].mean(dim=(1, 2)); Bm = d[:, ~halfA].mean(dim=(1, 2))   # [u,V]
            for i in range(u):
                Sf[u0 + i] = self._lift_one(A[i], Bm[i], floor)
                Sr[u0 + i] = self._lift_one(A[i], Bm[i], None)
                KL[u0 + i] = (lp[i].exp() * d[i]).sum(-1).mean()
            lens[u0:u0 + u] = (0.5 * (A + Bm)).topk(10, dim=-1).indices   # overall mean-delta top tokens
            del lp, d, A, Bm
        self._push = None
        return Sr, Sf, KL, lens

    # ---- the frozen statistic (map Q8 + the 2026-10-26 base-prob floor fix) ----
    @staticmethod
    def _lift_one(A, Bm, cand=None):
        """Cross-fitted symmetric 2-fold lift from a unit's A/B mean-delta vectors [V]: pick the 10
        top-delta tokens on A and average their delta on B, and vice versa; average. `cand` (a bool
        [V] mask) restricts the pick to tokens the chip plausibly emits, killing rare-token noise."""
        def pick_score(pick, score):
            if cand is not None:
                pick = pick.masked_fill(~cand, -1e30)
            return score[pick.topk(10).indices].mean()
        return 0.5 * (pick_score(A, Bm) + pick_score(Bm, A))


def build_pushes(scr, L, units, med, dose):
    """units: list of ('neuron', n) or ('null', rvec[3072]). Returns pushes [U,3072] for +dose."""
    H = scr.hidden
    out = torch.zeros(len(units), scr.model.config.intermediate_size, device=scr.device)
    W = scr.Wdown[L]
    for i, (kind, val) in enumerate(units):
        if kind == "neuron":
            out[i, val] = dose * med / scr.colnorm[L][val].clamp_min(1e-20)
        else:
            r = val.to(scr.device).float()
            out[i] = dose * med * r / (W @ r).norm().clamp_min(1e-20)
    return out


def null_dirs(n_nulls, dim, seed, device):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_nulls, dim, generator=g)      # r ~ N(0,I); normalized inside build_pushes


def score_units(scr, L, base, units, dose, ubatch=4):
    """Score every unit at +dose and -dose. Returns S_raw (primary no-floor lift, the winning sign's),
    S (floored secondary, max over signs), KL, lens token ids [U,10] (winning sign), and sign [U]."""
    h_inL, ids = base["h_in"][L], base["ids"]
    B = ids.shape[0]
    halfA = torch.zeros(B, dtype=torch.bool, device=scr.device); halfA[: B // 2] = True
    logp_base, floor = base["logp_base"], base["floor"]
    sr, sf, kl, ln = [], [], [], []
    for sign in (+1.0, -1.0):
        pushes = build_pushes(scr, L, units, base["med"][L], sign * dose)
        Sr, Sf, KL, lens = scr.lifts_for(L, h_inL, pushes, ids, logp_base, halfA, floor, ubatch)
        sr.append(Sr); sf.append(Sf); kl.append(KL); ln.append(lens)
    win = sr[0] >= sr[1]                                    # winning sign per unit = larger raw lift
    return {"S_raw": torch.where(win, sr[0], sr[1]),
            "S": torch.maximum(sf[0], sf[1]),
            "KL": torch.maximum(kl[0], kl[1]),
            "lens": torch.where(win[:, None], ln[0], ln[1]),
            "sign": torch.where(win, torch.ones_like(sr[0]), -torch.ones_like(sr[0]))}


def bh_by(pvals, q=0.05):
    """Benjamini-Hochberg and Benjamini-Yekutieli rejections over m=86,016. Returns (bh_mask, by_mask,
    bh_thresh, by_thresh) aligned to pvals; m is the full model, not just the tested units."""
    m = LAYERS * 3072
    p = np.asarray(pvals)
    order = np.argsort(p)
    ps = p[order]
    k = np.arange(1, len(ps) + 1)
    def step(c):
        thr = q * k / (m * c)
        ok = ps <= thr
        kmax = np.max(np.where(ok)[0]) + 1 if ok.any() else 0
        mask = np.zeros(len(p), bool)
        if kmax:
            mask[order[:kmax]] = True
        return mask, (ps[kmax - 1] if kmax else 0.0)
    bh_mask, bh_t = step(1.0)
    by_mask, by_t = step(np.log(m) + np.euler_gamma + 1 / (2 * m))  # harmonic sum ~ ln m + gamma
    return bh_mask, by_mask, bh_t, by_t


# ---------- contexts ----------
def demo_contexts(scr, n=4, T=64):
    """Quick contexts for the self-test: tokenize fixed text, pad/trim to T. Real pilot uses
    FineWeb-Edu + UltraChat (atlas/corpus.py, built next)."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(IR.torq_file("tokenizer.json"))
    text = ["The ocean waves crashed on the shore as the ship sailed past the reef and the sea.",
            "Photosynthesis lets plants turn sunlight, water and carbon dioxide into sugar and oxygen.",
            "The committee reviewed the quarterly budget and moved the vote to next Tuesday afternoon.",
            "She boarded the train, opened her novel, and read quietly until the city came into view.",
            "Volcanoes form where molten rock rises through the crust and erupts as lava and ash.",
            "He debugged the login page, fixed the off-by-one error, and pushed the patch to main."]
    ids = []
    for t in text[:n]:
        e = tok.encode(t).ids
        e = (e + [tok.encode(".").ids[0]] * T)[:T]
        ids.append(e)
    return torch.tensor(ids)


def cmd_selftest(a):
    scr = Screener(variant=a.variant, device=a.device)
    ids = demo_contexts(scr, n=a.contexts)
    t0 = time.time()
    base = scr.base(ids)
    L = 8
    # positive control n908 (+, ocean) plus a spread of other neurons and 64 column-space nulls
    neurons = [908] + list(range(0, 3072, 160))
    units = [("neuron", n) for n in neurons]
    nulls = null_dirs(a.nulls, scr.model.config.intermediate_size, seed=L, device=a.device)
    units += [("null", nulls[i]) for i in range(a.nulls)]
    out = score_units(scr, L, base, units, dose=0.5, ubatch=a.ubatch)
    S, Sraw = out["S"].cpu().numpy(), out["S_raw"].cpu().numpy()
    n_neu = len(neurons)
    print(f"L{L}  base+score {time.time()-t0:.1f}s  med={base['med'][L]:.1f}  floor>{FLOOR:g} "
          f"keeps {int(base['floor'].sum())}/{N_KEPT} tokens")
    for name, arr in [("raw (primary)", Sraw), ("floored (secondary)", S)]:
        Sneu, Snull = arr[:n_neu], arr[n_neu:]
        p_ctrl = (1 + int((Snull >= arr[0]).sum())) / (a.nulls + 1)
        print(f"  [{name}] n908 S={arr[0]:+.3f} p={p_ctrl:.3f}  "
              f"null mean {Snull.mean():+.3f} p95 {np.quantile(Snull,0.95):+.3f} max {Snull.max():+.3f}  "
              f"({'beats' if p_ctrl < 0.05 else 'FAILS'} at p<.05)")


def cmd_lens(a):
    """Diagnostic: decode the 10 tokens a push raises most (the frozen statistic's picks), with their
    baseline probability, so we can see whether the lift tracks a concept or rare-token log-prob noise."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(IR.torq_file("tokenizer.json"))
    scr = Screener(variant=a.variant, device=a.device)
    ids = demo_contexts(scr, n=a.contexts)
    base = scr.base(ids)
    L = a.layer
    nulls = null_dirs(4, scr.model.config.intermediate_size, seed=L, device=a.device)
    probes = [("neuron", a.neuron, "neuron n%d" % a.neuron)] + \
             [("null", nulls[i], "null %d" % i) for i in range(2)]
    base_ctx = scr._ctx(base["ids"]); h = base["h_in"][L]; med = base["med"][L]
    pos_emb, mask, position_ids = base_ctx
    scr._attach(L)
    for kind, val, name in probes:
        push = build_pushes(scr, L, [(kind, val)], med, a.dose)[0]
        scr._push = push
        lg = scr._rerun_from(L, h, pos_emb, mask, position_ids)
        d = (F.log_softmax(lg, -1) - base["logp_base"]).mean(dim=(0, 1))   # [V] mean delta
        idx = d.topk(10).indices
        base_p = base["logp_base"].exp().mean(dim=(0, 1))[idx]             # mean baseline prob
        toks = [tok.decode([scr.ref.lut[int(i)]]) for i in idx.cpu()]
        print(f"\n{name} (dose {a.dose:+g}): top-10 raised tokens")
        for t, dd, bp in zip(toks, d[idx].tolist(), base_p.tolist()):
            print(f"   {t!r:16} dlogp={dd:+.2f}  base_prob={bp:.2e}")
    scr._push = None


def cmd_faithful(a):
    """R3 gate: the GPU replica must match the CPU replica. Compares neuron activations on --device
    vs CPU, teacher-forced on eval/ref_greedy.json; pass = per-layer cosine >= 0.999. Also prints
    int4(npu)-vs-int4-onnx cosine on CPU so the screen's cheaper numerics are justified."""
    seqs = IR.tf_sequences(a.tokens)
    dev = Screener(variant=a.variant, device=a.device)
    cpu = dev if a.device == "cpu" else Screener(variant=a.variant, device="cpu")
    # position 0 is the attention sink (huge activations, numerically sensitive) -> report with/without
    mean0, meanN, worstN = [[] for _ in range(LAYERS)], [[] for _ in range(LAYERS)], [1.0] * LAYERS
    for _, ids, _ in seqs:
        t = torch.tensor(ids)
        ad, ac = dev.neuron_acts(t).float().cpu(), cpu.neuron_acts(t).float().cpu()
        c = F.cosine_similarity(ad, ac, dim=-1)                 # [T,28]
        for L in range(LAYERS):
            mean0[L].append(c[:, L]); meanN[L].append(c[1:, L])
            worstN[L] = min(worstN[L], c[1:, L].min().item())
    m_all = [torch.cat(x).mean().item() for x in mean0]
    m_np0 = [torch.cat(x).mean().item() for x in meanN]
    print(f"GPU({a.device})-vs-CPU {a.variant} neuron-act cosine, per layer:")
    print("  mean (all pos):  " + " ".join(f"{c:.4f}" for c in m_all))
    print("  mean (excl pos0):" + " ".join(f"{c:.4f}" for c in m_np0))
    mn = min(m_np0)
    print(f"min-over-layers mean (excl pos0) = {mn:.5f}; worst single pos (excl pos0) = {min(worstN):.4f}")
    print(f"  ->  {'PASS (mean>=0.999 every layer)' if mn >= 0.999 else 'mean below 0.999 on some layer'}")
    # justify the screen's cheaper int4-onnx numerics against the board kernel (int4/npu), on CPU
    if a.variant != "int4":
        npu = Screener(variant="int4", device="cpu")
        c2 = [1.0] * LAYERS
        for _, ids, _ in seqs:
            t = torch.tensor(ids)
            cc = F.cosine_similarity(cpu.neuron_acts(t).float().cpu(),
                                     npu.neuron_acts(t).float().cpu(), dim=-1).min(0).values
            c2 = [min(c2[L], cc[L].item()) for L in range(LAYERS)]
        print(f"{a.variant}-vs-int4(npu kernel) cosine, min over layers = {min(c2):.5f} "
              f"(screen uses {a.variant}; this shows it tracks the board kernel)")


CONTROL = {8: (908, "+", "ocean"), 19: (834, "-", "ships/sea")}  # CPU-demo positive controls


def emp_p(S_neu, S_null):
    """Phipson-Smyth empirical p-value for each neuron against the layer's nulls."""
    order = np.argsort(S_null)
    ranks = np.searchsorted(S_null[order], S_neu, side="left")   # nulls strictly greater start here
    n_ge = len(S_null) - ranks
    return (1 + n_ge) / (len(S_null) + 1)


def cmd_screen(a):
    layers = [int(x) for x in a.layers.split(",")]
    npz = np.load(a.contexts)
    ids_all = torch.from_numpy(npz["ids"].astype(np.int64))      # [32,64]
    nctx = a.n_contexts                                          # per set (16 full; fewer for smoke)
    set1, set2 = ids_all[:nctx], ids_all[16:16 + nctx]
    scr = Screener(variant=a.variant, device=a.device)
    dim = scr.model.config.intermediate_size
    n_neu = dim if a.max_neurons == 0 else a.max_neurons
    res, summary = {}, []
    for L in layers:
        t0 = time.time()
        nulls = null_dirs(a.nulls, dim, seed=L, device=a.device)
        units = [("neuron", n) for n in range(n_neu)] + [("null", nulls[i]) for i in range(a.nulls)]
        o1 = score_units(scr, L, scr.base(set1), units, dose=0.5, ubatch=a.ubatch)
        o2 = score_units(scr, L, scr.base(set2), units, dose=0.5, ubatch=a.ubatch)
        # PRIMARY = the pre-registered cross-fitted lift with NO floor (S_raw). The base-prob floor
        # (S) is kept only as a reported comparison: the pilot showed it removes concept tokens and
        # buries the positive controls, so it is NOT used for FDR.
        S1, S1flo, KL1 = (o1[k].cpu().numpy() for k in ("S_raw", "S", "KL"))
        lens1, sign1 = o1["lens"].cpu().numpy(), o1["sign"].cpu().numpy()
        S2 = o2["S_raw"].cpu().numpy()
        Sn1, Snull1 = S1[:n_neu], S1[n_neu:]
        Sn2 = S2[:n_neu]
        p = emp_p(Sn1, Snull1)                                   # primary (no-floor) empirical p
        p_floored = emp_p(S1flo[:n_neu], S1flo[n_neu:])          # secondary, for the record
        pfloor = 1.0 / (a.nulls + 1)
        at_floor = int((p <= pfloor + 1e-12).sum())             # neurons that beat every null
        top1 = set(np.argsort(-Sn1)[:200].tolist()); top2 = set(np.argsort(-Sn2)[:200].tolist())
        overlap = len(top1 & top2) / 200.0
        ctrl = CONTROL.get(L)
        cinfo = ""
        if ctrl and ctrl[0] < n_neu:
            cn = ctrl[0]
            cinfo = (f"  control n{cn}({ctrl[2]}): p={p[cn]:.4f} p_floored={p_floored[cn]:.4f} "
                     f"{'beats-all-nulls' if p[cn] <= pfloor + 1e-12 else ('<0.05' if p[cn]<=0.05 else 'misses')}")
        res[f"L{L}/S_neu"] = Sn1; res[f"L{L}/S_null"] = Snull1; res[f"L{L}/p"] = p
        res[f"L{L}/p_floored"] = p_floored; res[f"L{L}/KL_neu"] = KL1[:n_neu]
        res[f"L{L}/S_neu_set2"] = Sn2
        res[f"L{L}/lens"] = lens1[:n_neu]; res[f"L{L}/sign"] = sign1[:n_neu]
        summary.append((L, at_floor, overlap, p.min(), cinfo))
        print(f"L{L}: {time.time()-t0:.0f}s  neurons beating all {a.nulls} nulls (p-floor {pfloor:.1e}): "
              f"{at_floor}  min p={p.min():.1e}  top200 set1/set2 overlap={overlap:.2f}{cinfo}", flush=True)
        np.savez_compressed(a.out, **res, meta=json.dumps({"layers": layers[:layers.index(L)+1],
            "nulls": a.nulls, "n_neurons": n_neu, "n_contexts": nctx, "variant": a.variant, "floor": FLOOR}))
    # BH / BY over all tested neurons (m = full model), then survivors per layer
    allp = np.concatenate([res[f"L{L}/p"] for L in layers])
    bh, by, bht, byt = bh_by(allp, q=0.05)
    print(f"\nFDR over {len(allp)} tested neurons (m={LAYERS*3072}): BH survivors={int(bh.sum())} "
          f"(thresh {bht:.2e}), BY survivors={int(by.sum())} (thresh {byt:.2e})")
    off = 0
    for L in layers:
        nL = len(res[f"L{L}/p"])
        print(f"  L{L}: BH {int(bh[off:off+nL].sum())}  BY {int(by[off:off+nL].sum())}")
        off += nL
    res["meta"] = json.dumps({"layers": layers, "nulls": a.nulls, "n_neurons": n_neu,
                              "n_contexts": nctx, "variant": a.variant, "floor": FLOOR,
                              "summary": [(L, af, ov, float(mp)) for L, af, ov, mp, _ in summary]})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **res)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selftest")
    s.add_argument("--device", default="cpu"); s.add_argument("--contexts", type=int, default=4)
    s.add_argument("--nulls", type=int, default=64); s.add_argument("--ubatch", type=int, default=4)
    s.add_argument("--variant", default="int4-onnx", help="int4=board kernel; int4-onnx=faster bf16 (CPU test)")
    s = sub.add_parser("lens")
    s.add_argument("--layer", type=int, default=8); s.add_argument("--neuron", type=int, default=908)
    s.add_argument("--dose", type=float, default=0.5); s.add_argument("--contexts", type=int, default=4)
    s.add_argument("--device", default="cpu"); s.add_argument("--variant", default="int4-onnx")
    s = sub.add_parser("faithful")
    s.add_argument("--device", default="cuda"); s.add_argument("--variant", default="int4-onnx")
    s.add_argument("--tokens", default=str(ROOT / "eval" / "ref_greedy.json"))
    s = sub.add_parser("screen")
    s.add_argument("--layers", default="8,19"); s.add_argument("--nulls", type=int, default=8192)
    s.add_argument("--device", default="cuda"); s.add_argument("--ubatch", type=int, default=4)
    s.add_argument("--variant", default="int4-onnx",
                   help="int4-onnx=literal bf16 INT4 graph (fast, default); int4=slow board kernel")
    s.add_argument("--contexts", default=str(ROOT / "atlas" / "pilot_contexts.npz"))
    s.add_argument("--n-contexts", type=int, default=16, help="contexts per set (<=16)")
    s.add_argument("--max-neurons", type=int, default=0, help="0=all 3072; smaller for a smoke test")
    s.add_argument("--out", default=str(ROOT / "atlas" / "pilot_run.npz"))
    a = ap.parse_args()
    {"selftest": cmd_selftest, "lens": cmd_lens, "faithful": cmd_faithful,
     "screen": cmd_screen}[a.cmd](a)
