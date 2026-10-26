# CPU replica of Synaptics' INT4 Qwen3-0.6B (Synaptics/Qwen3-0.6B-GPTQ-int4-torq v2.3.0).
#
# The INT4 math is adapted from Synaptics' Apache-2.0 code (torq-compiler 2.3.0
# python/torq/lab/quantization/onnx/weights/{_gptq,_quantization}.py and the NPU kernel
# compiler/torq/Conversions/*/MixedQuantMatMulPattern.cpp). Copyright (c) 2025-2026
# Synaptics Incorporated, Apache License 2.0. See ref/README.md for the facts and sources.
#
#   int4_ref.py greedy  [--n 32] [--out eval/ref_greedy.json]   chat-template greedy decode
#   int4_ref.py dump    --tokens eval/ref_greedy.json --out acts.npz   teacher-forced dump
#   int4_ref.py gap     [--out eval/int4_vs_fp32.json]          INT4 vs fp32 on the test prompts
#   int4_ref.py synaptics [--out ...]                           redo compare_static_quant.py
#   int4_ref.py compare A.json B.json [--n 32]                  first-n greedy tokens agree?
import argparse, gc, json, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

BASE, BASE_REV = "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"
TORQ, TORQ_REV = "Synaptics/Qwen3-0.6B-GPTQ-int4-torq", "v2.3.0"
ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "eval" / "prompts.json"
N_KEPT = 99477          # token_id_lut[:99477] are the unique kept ids; the rest pads with 151668
STOP = {151643, 151645}  # <|endoftext|>, <|im_end|> (torq-examples Qwen3Static._should_stop)
LAYERS = 28
torch.set_grad_enabled(False)


def torq_file(name):
    return hf_hub_download(TORQ, name, revision=TORQ_REV)


def bf16_trunc(a):
    """fp32 -> bf16 by dropping the low 16 bits, as _quantization._fp32_to_bf16_raw does."""
    return (np.ascontiguousarray(a, np.float32).view(np.uint32) & 0xFFFF0000).view(np.float32)


def unpack_nibbles(packed):
    """int32 words -> eight 4-bit values each, low nibble first (_gptq._unpack_nibbles)."""
    p = packed.astype(np.uint32)
    return ((p[..., None] >> (np.arange(8, dtype=np.uint32) * 4)) & 0xF).reshape(*p.shape[:-1], -1)


def load_gptq(ckpt, key, rows=None):
    """One compressed-tensors GPTQ Linear -> signed int4 q [N,K], fp32 scale [N,G], int4 zp [N,G]."""
    n, k = (int(v) for v in ckpt[f"{key}.weight_shape"])
    q = unpack_nibbles(ckpt[f"{key}.weight_packed"])[:, :k].astype(np.int8) - 8
    s = ckpt[f"{key}.weight_scale"].astype(np.float32)
    zp = ckpt[f"{key}.weight_zero_point"]  # packed 8 per int32 along N
    zp = unpack_nibbles(zp.T.copy()).T[:n].astype(np.int8) - 8
    assert s.shape == zp.shape == (n, k // 32) and zp.min() >= -8 and zp.max() <= 7
    if rows is not None:  # vocab-trimmed LM head keeps the token_id_lut rows
        q, s, zp = q[rows], s[rows], zp[rows]
    return q, s, zp


def rtn_int4(w):
    """Round-to-nearest signed int4, block 32 along K (_quantization.quantize_int4_signed). w [N,K]."""
    n, k = w.shape
    wb = w.astype(np.float32).reshape(n, k // 32, 32)
    lo, hi = wb.min(2, keepdims=True), wb.max(2, keepdims=True)
    s = np.maximum((hi - lo) / 15.0, 1e-10)
    zp = np.round(-8.0 - lo / s).clip(-8, 7)
    q = np.round(wb / s + zp).clip(-8, 7)
    return q.reshape(n, k).astype(np.int8), s[..., 0], zp[..., 0].astype(np.int8)


class Int4Linear(nn.Module):
    """y = x @ ((q - zp) * bf16(scale)).T over 32-wide input blocks, in one of three numerics:

    npu   the SL2610 MixedQuantMatMul kernel: per-block partials P = bf16(sum x*q), block sums
          X = bf16(sum x), y = bf16(sum_g P*s + X*bf16(-zp*s)), fp32 accumulation (default)
    onnx  the exported graph's literal meaning: DequantizeLinear gives bf16 weights, then a bf16 MatMul
    fp32  fp32 fake-quant weights, as torq-tools' compare_static_quant.py builds them
    """

    def __init__(self, q, s, zp, mode):
        super().__init__()
        self.mode, (n, k) = mode, q.shape
        s = bf16_trunc(s)  # the export writes bf16 scales by truncation
        if mode == "npu":  # int4 values are exact in bf16, which halves the memory of fp32
            self.register_buffer("qT", torch.from_numpy(q.T.reshape(-1, 32, n).astype(np.float32)).bfloat16())
            self.register_buffer("s", torch.from_numpy(s.T.copy()))  # [G,N]
            # the compiler packs -zp*s into the scale table with round-to-nearest-even
            self.register_buffer("nzs", torch.from_numpy(-s.T * zp.T).bfloat16().float())
        else:
            w = (q.astype(np.float32).reshape(n, -1, 32) - zp[..., None]) * s[..., None]
            dt = torch.float32 if mode == "fp32" else torch.bfloat16
            self.register_buffer("w", torch.from_numpy(w.reshape(n, k)).to(dt))

    def forward(self, x):
        if self.mode != "npu":
            return F.linear(x, self.w)
        shp, g = x.shape, self.qT.shape[0]
        xf = x.reshape(-1, g, 32).float()
        out, qT = [], self.qT.float()
        for c in xf.split(64):  # token chunks bound the [G,T,N] partials
            p = torch.bmm(c.transpose(0, 1), qT).bfloat16().float()  # [G,T,N]
            xs = c.sum(-1).bfloat16().float()  # [T,G]
            out.append(torch.einsum("gtn,gn->tn", p, self.s) + xs @ self.nzs)
        return torch.cat(out).bfloat16().reshape(*shp[:-1], -1)


class Ref:
    """Qwen3-0.6B with Synaptics' INT4 weights (or fp32/bf16 baselines), trimmed head, hooks.

    variant: int4 (npu numerics, bf16 activations, the board replica), int4-onnx, int4-fp32
    (fp32 activations, GPTQ fake-quant), rtn-fp32, bf16 (no quantization), fp32.
    """

    def __init__(self, variant="int4"):
        self.variant = variant
        quant, _, mode = variant.partition("-")
        mode = mode or ("npu" if quant in ("int4", "rtn") else "")
        fp32 = mode == "fp32" or variant == "fp32"
        dt = torch.float32 if fp32 else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE, revision=BASE_REV, dtype=dt, attn_implementation="eager").eval()
        self.lut = np.load(torq_file("token_id_lut.npy")).astype(np.int64)
        # host embeddings: bf16 [151936,1024], equal to Qwen's own (bf16) embed_tokens
        emb = np.load(torq_file("token_embeddings.npy")).view(np.uint16).astype(np.uint32) << 16
        emb = torch.from_numpy(emb.view(np.float32))
        m = self.model.model
        m.embed_tokens.weight.data.copy_(emb)
        lut_t = torch.from_numpy(self.lut)
        if quant in ("int4", "rtn"):
            ckpt = load_file(torq_file("gptq/model.safetensors")) if quant == "int4" else None
            for name, mod in list(self.model.named_modules()):
                for child in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
                    if hasattr(mod, child) and isinstance(getattr(mod, child), nn.Linear):
                        key = f"{name}.{child}"
                        qsz = load_gptq(ckpt, key) if ckpt else rtn_int4(getattr(mod, child).weight.float().numpy())
                        setattr(mod, child, Int4Linear(*qsz, mode))
            # LM head: tied in Qwen3, untied by GPTQ; quantized on the lut rows (pads included)
            qsz = load_gptq(ckpt, "lm_head", self.lut) if ckpt else rtn_int4(emb[lut_t].numpy())
            self.head = Int4Linear(*qsz, mode)
        else:
            w = emb[lut_t].to(dt)
            self.head = lambda h: F.linear(h, w)
        self.acts = [None] * LAYERS
        for i, layer in enumerate(m.layers):
            layer.mlp.down_proj.register_forward_pre_hook(self._hook(i))

    def _hook(self, i):
        def f(_, inp):
            self.acts[i] = inp[0]
        return f

    def forward(self, ids, past=None):
        """ids [T] -> (logits [T,99520] over the padded trimmed vocab, acts [T,28,3072], cache)."""
        out = self.model.model(input_ids=torch.tensor([ids]), past_key_values=past, use_cache=True)
        h = out.last_hidden_state[0]
        logits = self.head(h).float()
        acts = torch.stack([a[0] for a in self.acts], 1)
        return logits, acts, out.past_key_values

    def greedy(self, prompt_ids, n):
        """Decode one token per call, prompt included, as the board runner does. LM-head logits
        are bf16, so exact ties happen; argmax takes the first index, like numpy on the board.
        Returns the tokens and each step's top-1 minus top-2 logit (0 = tie)."""
        past = None
        for t in prompt_ids:
            logits, _, past = self.forward([t], past)
        gen, margins = [], []
        while True:
            top = logits[-1].topk(2).values
            tok = int(self.lut[int(logits[-1].argmax())])
            gen.append(tok), margins.append(round((top[0] - top[1]).item(), 4))
            if tok in STOP or len(gen) >= n:
                return gen, margins
            logits, _, past = self.forward([tok], past)


def chat_ids(system, user, tok):
    """Prompt tokens exactly as torq-examples Qwen3Static builds them (instruct, --no-thinking):
    system turn (warm-up), user turn, then the assistant header with an empty think block."""
    ids = []
    if system:
        ids += tok.encode(f"<|im_start|>system\n{system}<|im_end|>\n").ids
    ids += tok.encode(f"<|im_start|>user\n{user}<|im_end|>\n").ids
    return ids + tok.encode("<|im_start|>assistant\n<think>\n\n</think>\n\n").ids


def load_prompts():
    cfg = json.loads(PROMPTS.read_text())
    tok = Tokenizer.from_file(torq_file("tokenizer.json"))
    from transformers import AutoTokenizer  # check against Qwen's own chat template
    hf = AutoTokenizer.from_pretrained(BASE, revision=BASE_REV)
    out = []
    for p in cfg["prompts"]:
        ids = chat_ids(cfg["system"], p["text"], tok)
        msgs = [{"role": "system", "content": cfg["system"]}, {"role": "user", "content": p["text"]}]
        ref = hf.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=True)
        ref = ref["input_ids"] if isinstance(ref, dict) or hasattr(ref, "keys") else ref
        assert list(ref) == ids, f"chat template mismatch on {p['id']}"
        out.append((p["id"], ids))
    return cfg, tok, out


def cmd_greedy(a):
    cfg, tok, prompts = load_prompts()
    ref = Ref(a.variant)
    res = {"variant": a.variant, "system": cfg["system"], "n": a.n, "prompts": []}
    for pid, ids in prompts:
        gen, margins = ref.greedy(ids, a.n)
        text = tok.decode(gen, skip_special_tokens=True)
        res["prompts"].append({"id": pid, "prompt_ids": ids, "gen_ids": gen, "margins": margins, "text": text})
        print(f"[{pid}] {text!r}", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


def tf_sequences(path):
    d = json.loads(Path(path).read_text())
    return [(p["id"], p["prompt_ids"] + p["gen_ids"], len(p["prompt_ids"])) for p in d["prompts"]]


def cmd_dump(a):
    ref = Ref(a.variant)
    out = {}
    for pid, ids, _ in tf_sequences(a.tokens):
        logits, acts, _ = ref.forward(ids)
        out[f"{pid}/ids"] = np.array(ids, np.int32)
        out[f"{pid}/logits"] = logits.numpy()  # [T, 99520] float32, padded trimmed vocab
        out[f"{pid}/acts"] = acts.float().numpy().astype(np.float16 if a.fp16 else np.float32)
    np.savez_compressed(a.out, **out)
    print(f"wrote {a.out}: {len(out) // 3} prompts")


def kl(p_logits, q_logits):
    lp, lq = F.log_softmax(p_logits.double(), -1), F.log_softmax(q_logits.double(), -1)
    return (lp.exp() * (lp - lq)).sum(-1)


def act_stats(ref, other):
    """Per layer: mean/min per-token cosine and mean top-100 overlap (by |activation|)."""
    a, b = ref.float(), other.float()  # [T,28,3072]
    cos = F.cosine_similarity(a, b, dim=-1)  # [T,28]
    top = lambda x: torch.zeros(x.shape, dtype=torch.bool).scatter_(-1, x.abs().topk(100, -1).indices, True)
    hit = (top(a) & top(b)).sum(-1) / 100.0  # [T,28]
    return cos, hit


def cmd_gap(a):
    seqs = tf_sequences(a.tokens)
    runs = {}
    for v in ("fp32", "bf16", "int4", "int4-onnx"):
        ref = Ref(v)
        runs[v] = [(lg[:, :N_KEPT], ac if v == "fp32" else ac.bfloat16())
                   for lg, ac, _ in (ref.forward(ids) for _, ids, _ in seqs)]
        del ref
        gc.collect()
        print(f"ran {v}", flush=True)
    res = {"tokens": a.tokens, "prompts": len(seqs), "positions": sum(len(s[1]) for s in seqs),
           "notes": "Teacher-forced on prompt+reference greedy tokens. Activation cosine is per token, "
                    "then averaged; top-100 overlap uses |activation|; logits metrics are over the 99,477 "
                    "unique trimmed-vocab tokens, all positions ('gen' = the generated positions only).",
           "comparisons": {}}
    for name, (r, o) in {"int4_vs_fp32": ("fp32", "int4"), "bf16_vs_fp32": ("fp32", "bf16"),
                         "int4_vs_bf16": ("bf16", "int4"), "int4_npu_vs_onnx": ("int4-onnx", "int4")}.items():
        cos, hit, kls, top1, gk, gt = [], [], [], [], [], []
        for (_, ids, np_), (lr, ar), (lo, ao) in zip(seqs, runs[r], runs[o]):
            c, h = act_stats(ar, ao)
            cos.append(c), hit.append(h)
            lr, lo = lr[:, :N_KEPT], lo[:, :N_KEPT]
            k, t = kl(lr, lo), (lr.argmax(-1) == lo.argmax(-1)).double()
            kls.append(k), top1.append(t), gk.append(k[np_ - 1:-1]), gt.append(t[np_ - 1:-1])
        cos, hit = torch.cat(cos), torch.cat(hit)
        res["comparisons"][name] = {
            "per_layer_cosine_mean": [round(x, 5) for x in cos.mean(0).tolist()],
            "per_layer_cosine_min": [round(x, 4) for x in cos.min(0).values.tolist()],
            "per_layer_cosine_mean_skip_pos0": [round(x, 5) for x in
                                                torch.cat([c[1:] for c in cos.split([len(s[1]) for s in seqs])]).mean(0).tolist()],
            "per_layer_top100_overlap": [round(x, 4) for x in hit.mean(0).tolist()],
            "logits_kl_mean": round(torch.cat(kls).mean().item(), 4),
            "top1_agreement": round(torch.cat(top1).mean().item(), 4),
            "gen_logits_kl_mean": round(torch.cat(gk).mean().item(), 4),
            "gen_top1_agreement": round(torch.cat(gt).mean().item(), 4),
        }
        c = res["comparisons"][name]
        print(name, "cos L0/L13/L27", [c["per_layer_cosine_mean"][i] for i in (0, 13, 27)],
              "top100", [c["per_layer_top100_overlap"][i] for i in (0, 13, 27)],
              "KL", c["logits_kl_mean"], "top1", c["top1_agreement"], flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


SYN_PROMPTS = ["What causes rainbows?", "Who wrote Romeo and Juliet?", "Write one sentence about embedded systems.",
               "Explain what a neural processing unit does in two sentences.", "List three fruits that are red.",
               "12345+54321="]


def cmd_synaptics(a):
    """torq-tools int4quant/compare_static_quant.py, on this replica: fp32 activations, fake-quant
    weights, teacher-forced on 6 prompts x 48 tokens (raw ChatML, fp32 greedy continuation),
    metrics over the padded 99,520-wide trimmed vocab."""
    tok = Tokenizer.from_file(torq_file("tokenizer.json"))
    fp32 = Ref("fp32")
    seqs = []
    for p in SYN_PROMPTS:
        ids = tok.encode(f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n").ids
        while len(ids) < 48:
            ids.append(int(fp32.lut[int(fp32.forward(ids)[0][-1].argmax())]))
        seqs.append(ids[:48])
    ref = torch.cat([fp32.forward(s)[0] for s in seqs]).double()
    del fp32
    gc.collect()
    res = {}
    for v in ("rtn-fp32", "int4-fp32", "int4"):
        m = Ref(v)
        q = torch.cat([m.forward(s)[0] for s in seqs]).double()
        cos = F.cosine_similarity(ref, q, dim=-1).mean().item()
        res[v] = {"kl": round(kl(ref, q).mean().item(), 4), "cosine": round(cos, 4),
                  "max_abs_error": round((ref - q).abs().max().item(), 3),
                  "top1_agreement": round((ref.argmax(-1) == q.argmax(-1)).double().mean().item(), 4)}
        print(v, res[v], flush=True)
        del m
        gc.collect()
    res["synaptics_readme"] = {"rtn-fp32": {"kl": 0.152, "cosine": 0.948, "top1_agreement": 0.833},
                               "int4-fp32": {"kl": 0.107, "cosine": 0.968, "top1_agreement": 0.854}}
    res["_setup"] = {"prompts": len(seqs), "positions": int(ref.shape[0]), "vocab": int(ref.shape[1]),
                     "note": "int4 = the board replica (npu numerics, bf16 activations), extra row"}
    Path(a.out).write_text(json.dumps(res, indent=1))


def cmd_compare(a):
    """Exit 0 if every prompt matches on the first n tokens or forks only at a near-tie, where the
    reference's top-1 minus top-2 logit is <= --tie (bf16 logits near 20 are 0.125 apart, so tiny
    rounding differences flip such tokens; the replica forks there even against itself)."""
    A, B = (json.loads(Path(p).read_text()) for p in (a.a, a.b))
    B = {p["id"]: p for p in B["prompts"]}
    count = {"match": 0, "near-tie fork": 0, "MISMATCH": 0}
    for p in A["prompts"]:
        x, y = p["gen_ids"][:a.n], B[p["id"]]["gen_ids"][:a.n]
        if p["prompt_ids"] != B[p["id"]]["prompt_ids"]:
            print(f"[{p['id']}] PROMPT TOKENS DIFFER")
        at = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), None)
        if at is None and len(x) == len(y):
            kind, note = "match", ""
        else:
            at = min(len(x), len(y)) if at is None else at
            m = (p.get("margins") or [None] * len(x))[at] if at < len(x) else None
            kind = "near-tie fork" if m is not None and m <= a.tie else "MISMATCH"
            note = f" at token {at} (reference margin {m})"
        count[kind] += 1
        print(f"[{p['id']}] {kind}{note}")
    print(f"first {a.n} tokens: " + ", ".join(f"{v} {k}" for k, v in count.items()))
    sys.exit(1 if count["MISMATCH"] else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    variants = ["int4", "int4-onnx", "int4-fp32", "rtn-fp32", "bf16", "fp32"]
    s = sub.add_parser("greedy"); s.add_argument("--n", type=int, default=32)
    s.add_argument("--variant", default="int4", choices=variants); s.add_argument("--out", default=str(ROOT / "eval/ref_greedy.json"))
    s = sub.add_parser("dump"); s.add_argument("--tokens", default=str(ROOT / "eval/ref_greedy.json"))
    s.add_argument("--variant", default="int4", choices=variants); s.add_argument("--out", required=True)
    s.add_argument("--fp16", action="store_true", help="store activations as float16 (smaller)")
    s = sub.add_parser("gap"); s.add_argument("--tokens", default=str(ROOT / "eval/ref_greedy.json"))
    s.add_argument("--out", default=str(ROOT / "eval/int4_vs_fp32.json"))
    s = sub.add_parser("synaptics"); s.add_argument("--out", default=str(ROOT / "eval/synaptics_compare.json"))
    s = sub.add_parser("compare"); s.add_argument("a"); s.add_argument("b"); s.add_argument("--n", type=int, default=32)
    s.add_argument("--tie", type=float, default=0.25, help="largest reference margin counted as a near-tie")
    a = ap.parse_args()
    {"greedy": cmd_greedy, "dump": cmd_dump, "gap": cmd_gap, "synaptics": cmd_synaptics, "compare": cmd_compare}[a.cmd](a)
