# R6 labeling harness: build the prompts the Cloudroom sub-agents run, collect the held-out
# evidence the scorer needs, and compute the real-vs-shuffled simulation score. Spawning itself is
# done with `room-cli thread spawn` (Haiku workers, Opus planners, Terra scorers); this module makes
# the prompt files and does the math.
#
#   python atlas/label_run.py worker  --keys L8n908,... --out /tmp/w.txt        # a worker's prompt
#   python atlas/label_run.py scorer  --key L8n908 --label-file lab.json --out /tmp/s.txt
#   python atlas/label_run.py score-evidence --out atlas/score_evidence.json    # held-out excerpts+acts
#   python atlas/label_run.py grade   --labels labels.json --scores scores.json # r, real vs shuffled
import argparse, importlib.util, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / "atlas" / "label_pilot_evidence.json"

WORKER_HEAD = """You label neurons of a small language model for an interpretability atlas. Each neuron has a POLE (+ or -). For each neuron you get: SNIPPETS where it fires hardest (<<tok>> marks the peak token = what it READS), the WORDS it pushes toward (logit-lens from the weights and steer-lens from nudging it = what it WRITES), and a STEERED SAMPLE (the model's text when the neuron is pushed = the EFFECT).

For EACH neuron write a label:
- name: at most 4 words
- description: one line - what text it responds to AND what it pushes toward
- verdict: "clear" (snippets, lens and sample agree on one concept), "likely" (mostly coherent), or "mixed" (polysemantic, or it reads one thing and writes another)
- concepts: exactly 10 short words/tokens it relates to

Judge honestly: if reads and writes disagree, say "mixed" and describe both. Output ONLY a JSON object mapping each neuron key to {"name","description","verdict","concepts"}. No other text.

"""


def load_ev():
    return json.loads(EV.read_text())["poles"]


def pole_block(key, e, n_snip=12):
    L = [f'NEURON {key} (pole {e["sign"]}):', " SNIPPETS (<<peak>>):"]
    for s in e["snippets"][:n_snip]:
        L.append(f'  [{s["act"]:+.1f}] {s["text"]}')
    L.append(" LOGIT-LENS (writes): " + ", ".join(w for w in e["logit_lens"] if w)[:400])
    L.append(" STEER-LENS (writes): " + ", ".join(w for w in e["steer_lens"] if w)[:400])
    if e.get("steered_sample"):
        L.append(" STEERED SAMPLE: " + e["steered_sample"][:300].replace("\n", " "))
    return "\n".join(L)


def cmd_worker(a):
    ev = load_ev()
    keys = a.keys.split(",")
    body = WORKER_HEAD + "\n---\n".join(pole_block(k, ev[k]) for k in keys)
    Path(a.out).write_text(body)
    print(f"worker prompt: {len(keys)} poles, {len(body)} chars -> {a.out}")


def cmd_scorer(a):
    """Scorer sees the label (NOT the neuron) + held-out excerpts, predicts firing 0-10 each."""
    se = json.loads((ROOT / "atlas" / "score_evidence.json").read_text())[a.key]
    lab = json.loads(Path(a.label_file).read_text())
    head = (f'A neuron has this label (you do NOT see the neuron):\n name: {lab["name"]}\n'
            f' description: {lab["description"]}\n concepts: {", ".join(lab["concepts"])}\n\n'
            "For EACH excerpt below, predict how strongly THIS neuron fires on it, integer 0 (not at "
            "all) to 10 (fires very strongly), from the label alone. Output ONLY a JSON list of "
            f'{len(se["excerpts"])} integers in order.\n\nEXCERPTS:\n')
    body = head + "\n".join(f'{i+1}. {x["text"]}' for i, x in enumerate(se["excerpts"]))
    Path(a.out).write_text(body)
    print(f"scorer prompt for {a.key}: {len(se['excerpts'])} excerpts -> {a.out}")


SCORER_HEAD = ("Score how strongly each neuron fires on each excerpt, integer 0 (not at all) to 10 "
               "(fires very strongly), using ONLY that neuron's label (you do NOT see the neuron). "
               "Output ONLY a JSON object mapping each neuron key to a list of integers, one per "
               "excerpt in order. No other text.\n\n")


def cmd_scorer_batch(a):
    """Emit one scorer prompt covering several poles. mode=real uses each pole's own label;
    mode=shuffled uses a derangement (each pole gets another pole's label) -> the null baseline."""
    se = json.loads((ROOT / "atlas" / "score_evidence.json").read_text())
    labels = json.loads(Path(a.labels).read_text())
    keys = [k for k in a.keys.split(",") if k in labels and k in se]
    assign = {k: k for k in keys}
    if a.mode == "shuffled" and len(keys) > 1:
        rot = keys[1:] + keys[:1]
        assign = dict(zip(keys, rot))
    blocks = []
    for k in keys:
        lab = labels[assign[k]]
        ex = se[k]["excerpts"]
        b = [f'NEURON {k} | label: {lab["name"]} - {lab["description"]} | concepts: {", ".join(lab["concepts"])}',
             f' EXCERPTS ({len(ex)}):']
        b += [f'  {i+1}. {x["text"]}' for i, x in enumerate(ex)]
        blocks.append("\n".join(b))
    Path(a.out).write_text(SCORER_HEAD + "\n\n".join(blocks))
    print(f"scorer-batch ({a.mode}): {len(keys)} poles -> {a.out}")


def cmd_score_evidence(a):
    """Fresh corpus pass: per pole, 10 held-out top-firing + 10 random excerpts with true peak act
    (binned 0-10). Held-out = a different corpus seed than the labelling snippets."""
    import torch
    spec = importlib.util.spec_from_file_location("screen", ROOT / "atlas" / "screen.py")
    SC = importlib.util.module_from_spec(spec); spec.loader.exec_module(SC); IR = SC.IR
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(IR.torq_file("tokenizer.json"))
    ev = load_ev(); sweep = np.load(ROOT / "atlas" / "sweep_run.npz", allow_pickle=True)
    scr = SC.Screener(variant="int4-onnx", device="cpu")
    poles = [(e["layer"], e["neuron"], 1.0 if e["sign"] == "+" else -1.0, k) for k, e in ev.items()]
    from datasets import load_dataset
    wins = []
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True).shuffle(seed=99, buffer_size=2000)
    for row in ds:
        ids = tok.encode(row["text"]).ids
        for s in range(32, min(len(ids) - 64, 32 + 64 * 4), 64):
            wins.append(ids[s:s + 64])
        if len(wins) >= a.windows:
            break
    wins = wins[:a.windows]
    peak = {k: [] for (_, _, _, k) in poles}
    for wi, ids in enumerate(wins):
        acts = scr.neuron_acts(torch.tensor(ids))
        for (L, n, sg, k) in poles:
            peak[k].append(float(acts[:, L, n].float().numpy().__mul__(sg).max()))
        if wi % 40 == 0:
            print(f"  {wi}/{len(wins)}", flush=True)
    out = {}
    rng = np.random.default_rng(1)
    for (L, n, sg, k) in poles:
        pk = np.array(peak[k]); order = np.argsort(-pk)
        top = order[:10]; rand = rng.choice(order[10:], size=min(10, len(order) - 10), replace=False)
        sel = list(top) + list(rand)
        lo, hi = pk.min(), pk.max() + 1e-9
        out[k] = {"excerpts": [{"text": tok.decode(wins[i]).replace("\n", " ")[:300],
                                "true_bin": int(round(10 * (pk[i] - lo) / (hi - lo)))} for i in sel]}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}: {len(out)} poles")


def cmd_grade(a):
    """Pearson r of predicted vs true bins, real labels vs shuffled, with a bootstrap CI on the gap."""
    labels = json.loads(Path(a.labels).read_text())      # {key: {name,description,concepts}}
    scores = json.loads(Path(a.scores).read_text())       # {key: {"real":[...], "shuffled":[...]}}
    se = json.loads((ROOT / "atlas" / "score_evidence.json").read_text())
    def r(pred, key):
        t = np.array([x["true_bin"] for x in se[key]["excerpts"]], float)
        p = np.array(pred, float)
        if p.std() < 1e-9 or t.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(p, t)[0, 1])
    keys = [k for k in labels if k in scores]
    real = np.array([r(scores[k]["real"], k) for k in keys])
    shuf = np.array([r(scores[k]["shuffled"], k) for k in keys])
    gap = real - shuf
    bs = np.array([np.mean(gap[np.random.default_rng(s).integers(0, len(gap), len(gap))]) for s in range(2000)])
    print(f"n={len(keys)}  real r mean {real.mean():+.3f}  shuffled r mean {shuf.mean():+.3f}  "
          f"gap {gap.mean():+.3f}  95% CI [{np.quantile(bs,.025):+.3f}, {np.quantile(bs,.975):+.3f}]  "
          f"{'PASS (CI>0)' if np.quantile(bs,.025) > 0 else 'fail'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("worker"); s.add_argument("--keys", required=True); s.add_argument("--out", required=True)
    s = sub.add_parser("scorer"); s.add_argument("--key", required=True); s.add_argument("--label-file", required=True); s.add_argument("--out", required=True)
    s = sub.add_parser("scorer-batch"); s.add_argument("--keys", required=True); s.add_argument("--labels", required=True)
    s.add_argument("--mode", choices=["real", "shuffled"], required=True); s.add_argument("--out", required=True)
    s = sub.add_parser("score-evidence"); s.add_argument("--out", default=str(ROOT / "atlas" / "score_evidence.json")); s.add_argument("--windows", type=int, default=200)
    s = sub.add_parser("grade"); s.add_argument("--labels", required=True); s.add_argument("--scores", required=True)
    a = ap.parse_args()
    {"worker": cmd_worker, "scorer": cmd_scorer, "scorer-batch": cmd_scorer_batch,
     "score-evidence": cmd_score_evidence, "grade": cmd_grade}[a.cmd](a)
