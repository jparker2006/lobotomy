# R6 full fan-out: label the ~3,000 poles with Haiku workers (batch 20), in waves so Cloudroom isn't
# hit with 150 at once. Polls each worker, collects + validates its JSON labels, archives it, and
# re-runs any batch that failed to parse. Orchestrator = this process (the planner QC role is done
# here + by the scorer afterwards). All $0 (Jake's Cloudroom plan). Writes atlas/labels_full.json.
#
#   python atlas/label_fanout.py --evidence atlas/labels_evidence.json --wave 40 --batch 20
import argparse, importlib.util, json, re, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LR = importlib.util.spec_from_file_location("label_run", ROOT / "atlas" / "label_run.py")
label_run = importlib.util.module_from_spec(LR); LR.loader.exec_module(label_run)
WORKER_HEAD, pole_block = label_run.WORKER_HEAD, label_run.pole_block
TMP = Path("/tmp/atlas/fanout"); TMP.mkdir(parents=True, exist_ok=True)


def sh(args):
    return subprocess.run(args, capture_output=True, text=True).stdout


def spawn(prompt_file, title):
    out = sh(["room-cli", "thread", "spawn", "--provider", "claude-code", "--model",
              "claude-haiku-5-5", "--reasoning-level", "high", "--shared-sandbox",
              "--title", title, "--prompt-file", prompt_file])
    try:
        return json.loads(out)["id"]
    except Exception:
        return None


def collect(cid):
    try:
        d = json.loads(sh(["room-cli", "thread", "output", cid]))
    except Exception:
        return None, "nostate"
    if d.get("state") != "idle":
        return None, "busy"
    r = (d.get("reply") or "").strip()
    r = re.sub(r"^```json\s*|\s*```$", "", r)
    m = re.search(r"\{.*\}", r, re.S)
    if not m:
        return None, "noJSON"
    try:
        labs = json.loads(m.group(0))
    except Exception:
        return None, "badJSON"
    return {re.sub(r"\s*\(pole.*?\)", "", k): v for k, v in labs.items()}, "ok"


def valid(lab):
    return isinstance(lab, dict) and all(f in lab for f in ("name", "description", "verdict", "concepts"))


def main(a):
    ev = json.loads(Path(a.evidence).read_text())["poles"]
    keys = list(ev.keys())
    batches = [keys[i:i + a.batch] for i in range(0, len(keys), a.batch)]
    print(f"{len(keys)} poles -> {len(batches)} batches of {a.batch}; waves of {a.wave}", flush=True)
    labels, failed = {}, []
    out_path = Path(a.out)
    if out_path.exists():
        labels = json.loads(out_path.read_text())
        print(f"resume: {len(labels)} labels already present", flush=True)

    def run_batches(blist, tag):
        nonlocal labels
        for w0 in range(0, len(blist), a.wave):
            wave = blist[w0:w0 + a.wave]
            live = {}
            for bi, batch in wave:
                if all(k in labels for k in batch):
                    continue
                pf = str(TMP / f"{tag}_b{bi}.txt")
                Path(pf).write_text(WORKER_HEAD + "\n---\n".join(pole_block(k, ev[k]) for k in batch))
                cid = spawn(pf, f"r6-{tag}-b{bi}")
                if cid:
                    live[cid] = (bi, batch)
                time.sleep(1)
            print(f"[{tag}] wave {w0//a.wave}: spawned {len(live)}", flush=True)
            deadline = time.time() + 1800
            while live and time.time() < deadline:
                time.sleep(20)
                for cid in list(live):
                    labs, status = collect(cid)
                    if status == "busy":
                        continue
                    bi, batch = live.pop(cid)
                    if labs:
                        got = {k: v for k, v in labs.items() if k in batch and valid(v)}
                        labels.update(got)
                        if len(got) < len(batch):
                            failed.append((bi, batch))
                    else:
                        failed.append((bi, batch))
                    sh(["room-cli", "thread", "archive", cid])
            for cid, (bi, batch) in live.items():     # timed out
                failed.append((bi, batch)); sh(["room-cli", "thread", "archive", cid])
            out_path.write_text(json.dumps(labels, indent=1))
            print(f"[{tag}] wave {w0//a.wave} done: {len(labels)} labels total", flush=True)

    run_batches(list(enumerate(batches)), "w")
    if failed:
        print(f"re-running {len(failed)} failed batches", flush=True)
        rr, failed[:] = failed[:], []
        run_batches(rr, "rr")
    out_path.write_text(json.dumps(labels, indent=1))
    print(f"DONE: {len(labels)}/{len(keys)} poles labelled; {len(failed)} still failed -> {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", default=str(ROOT / "atlas" / "labels_evidence.json"))
    ap.add_argument("--wave", type=int, default=40); ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "atlas" / "labels_full.json"))
    main(ap.parse_args())
