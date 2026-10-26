# Build the R3 pilot's teacher-forced contexts on the cloud sandbox (free CPU), save token ids so
# the GPU pod needs no dataset download and the chat answers are fixed (map Q9, spec #11).
#
# 32 contexts of 64 tokens: two disjoint sets of 16 (set1 gates the statistic, set2 is the stability
# re-check). Each 16-set is ordered [4 chat, 4 web | 4 chat, 4 web] so the screen's A/B halves are
# each balanced. Web = FineWeb-Edu sample-10BT (ODC-By). Chat = UltraChat 200k prompts answered by
# our own INT4 replica in the board chat template, then windowed.
#
#   python atlas/corpus.py --out atlas/pilot_contexts.npz
import argparse, importlib.util, json
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parent.parent
torch.set_grad_enabled(False)
spec = importlib.util.spec_from_file_location("int4_ref", ROOT / "ref" / "int4_ref.py")
IR = importlib.util.module_from_spec(spec); spec.loader.exec_module(IR)
T = 64


def web_windows(tok, n, seed):
    """n distinct FineWeb-Edu docs -> one 64-token window each (offset past the lead-in)."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=2000)
    out = []
    for row in ds:
        ids = tok.encode(row["text"]).ids
        if len(ids) >= 96:
            out.append(ids[32:32 + T])
            if len(out) == n:
                break
    return out


def chat_windows(ref, tok, sysprompt, n, seed):
    """n short UltraChat prompts, each answered greedily by the INT4 replica in the board template,
    then the first 64 tokens of (prompt + answer)."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=2000)
    out = []
    for row in ds:
        user = row["prompt"].strip()
        pids = IR.chat_ids(sysprompt, user, tok)
        if not (16 <= len(pids) <= 48):          # leave room for some answer inside 64 tokens
            continue
        gen, _ = ref.greedy(pids, T)             # decode until stop or T tokens
        seq = (pids + gen)[:T]
        seq = (seq + [tok.encode(".").ids[0]] * T)[:T]
        out.append(seq)
        print(f"  chat {len(out)}/{n}: {user[:60]!r}", flush=True)
        if len(out) == n:
            break
    return out


def main(a):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(IR.torq_file("tokenizer.json"))
    sysprompt = json.loads((ROOT / "eval" / "prompts.json").read_text())["system"]
    ref = IR.Ref("int4-onnx")   # literal bf16 reading of the board graph; ~identical tokens to npu, faster
    rows, meta = [], []
    for si, (seed_w, seed_c) in enumerate([(11, 12), (21, 22)]):   # set1, set2
        web = web_windows(tok, 8, seed_w)
        chat = chat_windows(ref, tok, sysprompt, 8, seed_c)
        # order: halfA = chat[0:4]+web[0:4], halfB = chat[4:8]+web[4:8]
        order = chat[0:4] + web[0:4] + chat[4:8] + web[4:8]
        kinds = ["chat"] * 4 + ["web"] * 4 + ["chat"] * 4 + ["web"] * 4
        rows += order
        meta += [{"set": si + 1, "half": "A" if i < 4 else "B", "kind": k}
                 for i, k in enumerate(kinds)]
        print(f"set{si+1}: {len(order)} contexts", flush=True)
    ids = np.array(rows, np.int32)
    assert ids.shape == (32, T), ids.shape
    np.savez_compressed(a.out, ids=ids, meta=json.dumps(meta),
                        note="R3 pilot contexts: 2 sets x 16 (8 chat replica-answered, 8 FineWeb-Edu). "
                             "Each 16-set ordered so the screen's first-8/last-8 A/B halves are balanced.")
    print(f"wrote {a.out}: {ids.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "atlas" / "pilot_contexts.npz"))
    main(ap.parse_args())
