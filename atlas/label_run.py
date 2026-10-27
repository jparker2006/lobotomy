# R6 labelling: the prompts the Cloudroom sub-agents get, reply parsing/validation, and the
# simulation-score math (#13). Pure stdlib, so planner sandboxes run it with the system python3.
# Spawning lives in label_fanout.py. The pilot's version of this file is in git at 32fe8b9.
#
# Evidence: atlas/r6/evidence.json.gz (what labellers see), atlas/r6/heldout.json.gz (what scorers
# see: dev for QC and choosing between labels, test for the final E4 score; never shown to labellers).
import gzip, json, os, random, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R6 = Path(os.environ.get("R6_DIR", ROOT / "atlas" / "r6"))          # override for dry runs
VERDICTS = ("clear", "likely", "mixed")

WORKER_HEAD = """You label neurons of a small language model (Qwen3-0.6B) for an interpretability atlas. Answer directly from the evidence below: do not use any tools, files or commands.

Each entry is one neuron POLE: the neuron pushed in one direction, "+" or "-". For a "-" pole, "firing" means the neuron going strongly negative.

Evidence per pole:
- SNIPPETS: the 15 text windows (64 tokens, web or chat) where this pole fires hardest. <<x>> marks the window's peak token. ⟦n⟧ after a token is its activation from 0 to 10, scaled so this pole's strongest token anywhere = 10; unmarked tokens are 0-1. ⏎ is a newline. This is what the neuron READS.
- LOGIT-LENS: the 10 tokens its output weights promote directly. STEER-LENS: the 10 tokens whose probability rises most when it is pushed. This is what it WRITES.
- STEERED: the model's answers with this pole pushed at dose 0.5 (compare them with the unsteered answers below). This is its EFFECT. Late-layer poles often just spam the tokens they write.

Unsteered answers: {baselines}

For EACH pole write:
- "name": at most 4 words
- "description": one line: what text it fires on (concrete: which token, in what context) and what it pushes the model toward
- "pole": "+" or "-", copied from the entry
- "verdict": "clear" (reads, writes and effect agree on one concept), "likely" (mostly one coherent story) or "mixed" (polysemantic, or it reads one thing and writes another)
- "concepts": exactly 10 short words or tokens tied to it, the ones it fires on first

Judge honestly: "mixed" is fine and common, and never invent meaning the evidence doesn't show. Later, someone who sees only your name, description and concepts (no evidence) will guess how strongly this pole fires on new text, so describe the firing pattern precisely, e.g. "the second piece of a word split into tokens", "the token right after an opening quote", "'than' after a comparative".

Output ONLY one JSON object mapping each pole key to {{"name", "description", "pole", "verdict", "concepts"}}, covering all {n} keys. No prose, no code fences.

"""

SCORER_HEAD = """You are simulating neurons of a small language model. Answer directly: do not use any tools, files or commands.

For each neuron you get only its LABEL, written by someone who studied it, and {m} numbered text excerpts. For EACH excerpt, predict the neuron's PEAK activation anywhere in that excerpt, as an integer from 0 (does not fire) to 10 (fires as strongly as it ever does), using the label alone. Most text leaves a neuron quiet; some excerpts contain what it fires on. ⏎ marks a newline.

Output ONLY one JSON object mapping each neuron key to a list of exactly {m} integers, one per excerpt in order, covering all {n} keys. No prose, no code fences.

"""


def load_gz(name):
    with gzip.open(R6 / name, "rt") as f:
        return json.load(f)


def pole_block(k, e):
    L = [f"=== {k} (pole {e['sign']}, layer {e['layer']}) ===", "SNIPPETS:"]
    L += [f" {i}. [{s['kind']}] {s['text']}" for i, s in enumerate(e["snippets"], 1)]
    L.append("LOGIT-LENS: " + json.dumps([w for w in e["logit_lens"] if w], ensure_ascii=False))
    L.append("STEER-LENS: " + json.dumps([w for w in e["steer_lens"] if w], ensure_ascii=False))
    for s in e.get("steered", []):
        L.append(f"STEERED ({s['prompt']}, dose {s['dose']:+.1f}): " + s["text"].replace("\n", "⏎"))
    return "\n".join(L)


def worker_prompt(keys, ev):
    base = ev["meta"]["steer"]
    bl = "; ".join(f'{n} ("{base["prompts"][n]}"): "{t}"' for n, t in base.get("baseline", {}).items())
    return (WORKER_HEAD.format(baselines=bl or "(none)", n=len(keys))
            + "\n\n".join(pole_block(k, ev["poles"][k]) for k in keys))


def label_text(lab):
    return f'{lab["name"]} - {lab["description"]}', ", ".join(lab["concepts"])


def wordlist_label(e):
    """The word-list baseline: the tokens with the highest mean activation on the labelling text."""
    w = [x for x in e["top_tokens"] if x][:10]
    return {"name": "Token list", "description": "Fires on these tokens: " + ", ".join(w), "concepts": w}


def scorer_prompt(items, ho, split):
    """items: [(pole key, label dict)]. Each pole gets its own 20 held-out excerpts from `split`."""
    m = len(ho["poles"][items[0][0]][split]["idx"])
    blocks = []
    for k, lab in items:
        head, conc = label_text(lab)
        b = [f"=== NEURON {k} ===", f"LABEL: {head}", f"CONCEPTS: {conc}", "EXCERPTS:"]
        b += [f" {i}. {ho[split]['windows'][w]}" for i, w in enumerate(ho["poles"][k][split]["idx"], 1)]
        blocks.append("\n".join(b))
    return SCORER_HEAD.format(m=m, n=len(items)) + "\n\n".join(blocks)


def parse_json_reply(text):
    """Pull the outermost JSON object out of an agent reply (tolerates fences and stray prose)."""
    t = re.sub(r"```(?:json)?", "", text or "")
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        d = json.loads(t[i:j + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict):
        return None
    def key(k):                          # workers write "L27n267", "L27n267+", "L27n267 (pole +)", ...
        m = re.search(r"L\d+n\d+", str(k))
        return m.group(0) if m else str(k).strip()
    return {key(k): v for k, v in d.items()}


def clean_label(lab, sign):
    """Validate one worker label; returns the cleaned label or None. The pole is taken from the
    evidence (a mismatch is recorded, not trusted)."""
    if not isinstance(lab, dict):
        return None
    name, desc = str(lab.get("name", "")).strip(), str(lab.get("description", "")).strip()
    verdict = str(lab.get("verdict", "")).strip().lower()
    conc = lab.get("concepts")
    if not name or not desc or verdict not in VERDICTS or not isinstance(conc, list):
        return None
    conc = [str(c).strip() for c in conc if str(c).strip()]
    if len(conc) < 10:
        return None
    out = {"name": name, "description": desc, "pole": sign, "verdict": verdict, "concepts": conc[:10]}
    if str(lab.get("pole", sign)).strip() not in (sign, {"+": "plus", "-": "minus"}[sign]):
        out["pole_mismatch"] = str(lab.get("pole"))
    if len(name.split()) > 4:
        out["name_long"] = True
    return out


def clean_preds(p, m):
    if not isinstance(p, list) or len(p) != m:
        return None
    try:
        return [min(10, max(0, int(round(float(x))))) for x in p]
    except (TypeError, ValueError):
        return None


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    vx = sum((a - mx) ** 2 for a in x); vy = sum((b - my) ** 2 for b in y)
    if vx < 1e-12 or vy < 1e-12:
        return 0.0                      # constant guesses carry no information
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (vx * vy) ** 0.5


def bootstrap_ci(vals, n=2000, seed=0):
    rng = random.Random(seed)
    k = len(vals)
    means = sorted(sum(vals[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return sum(vals) / k, means[int(0.025 * n)], means[int(0.975 * n) - 1]


def shuffle_map(keys, ev, seed=13):
    """Same-layer derangement: each pole is scored with another random pole's label from its layer
    (spec E4). A pole alone in its layer borrows from the nearest layer among `keys`."""
    rng = random.Random(seed)
    by = {}
    for k in keys:
        by.setdefault(ev["poles"][k]["layer"], []).append(k)
    out = {}
    for L, ks in sorted(by.items()):
        if len(ks) == 1:
            near = sorted((abs(L2 - L), L2) for L2 in by if L2 != L)
            out[ks[0]] = rng.choice(by[near[0][1]]) if near else ks[0]
            continue
        while True:
            perm = ks[:]
            rng.shuffle(perm)
            if all(a != b for a, b in zip(ks, perm)):
                break
        out.update(zip(ks, perm))
    return out
