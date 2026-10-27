# R6 fan-out engine (#13). Spawns Cloudroom sub-agents, each in its OWN sandbox (never
# --shared-sandbox: 40 agents in one 8 GB sandbox killed the first orchestrator), polls them, validates
# their JSON, archives them and retries failures. Concurrency ramps: --max-live at first, --ramp-to once
# --ramp-after jobs came back OK, halved after a run of failures. Every command resumes from its state
# file, so it can be stopped and rerun at any time. Pure stdlib (system python3 is enough).
#
#   python3 atlas/label_fanout.py label --planner p1 --version v1                # label the slice
#   python3 atlas/label_fanout.py label --planner p1 --version v2 --batches 3,7  # fresh re-run of batches
#   python3 atlas/label_fanout.py score --planner p1 --split dev --mode real --version v1
#   python3 atlas/label_fanout.py grade --planner p1 --split dev --version v1     # per-pole r, per batch
#   python3 atlas/label_fanout.py pick  --planner p1 --versions v1,v2            # best label per pole on dev
#   python3 atlas/label_fanout.py score --planner p1 --split test --mode real --version final   # + shuffled, wordlist
#   python3 atlas/label_fanout.py grade --planner p1 --split test --version final  # real vs shuffled vs word list
#   python3 atlas/label_fanout.py show  --planner p1 --key L27n267               # QC: evidence + labels + r
#   python3 atlas/label_fanout.py merge                                           # all planners -> labels.json, e4.json
import argparse, importlib.util, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LR = importlib.util.spec_from_file_location("label_run", ROOT / "atlas" / "label_run.py")
lr = importlib.util.module_from_spec(LR); LR.loader.exec_module(lr)
MAX_ATTEMPTS = 3
TERMINAL = {"idle", "replied", "done", "completed", "failed", "error", "stopped", "archived"}
RETRY_HINT = ("\nNOTE: this is a second labelling pass. Earlier labels for some of these poles predicted "
              "their firing on new text poorly. Look closely at exactly which tokens carry the high ⟦n⟧ "
              "values and in what context, and describe that pattern.\n\n")


def sh(args, timeout=120):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip(), r.returncode
    except subprocess.TimeoutExpired:
        return "", -1


def load_json(p, default):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


def save_json(p, d):
    p = Path(p); tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False)); tmp.replace(p)


class Fleet:
    """Runs jobs (each a list of pole keys -> one sub-agent) to completion. State is a JSON file."""

    def __init__(self, a, name, make_prompt, accept):
        self.a, self.name, self.make_prompt, self.accept = a, name, make_prompt, accept
        self.state_path = a.dir / f"state_{name}.json"
        self.st = load_json(self.state_path, {"jobs": {}, "ok": 0, "next": 0})
        self.tmp = Path("/tmp/r6") / a.planner / name
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.cap = a.max_live if self.st["ok"] < a.ramp_after else a.ramp_to
        self.fail_streak = 0

    def add(self, keys):
        jid = f"j{self.st['next']}"; self.st["next"] += 1
        self.st["jobs"][jid] = {"keys": keys, "attempt": 1, "status": "todo"}

    def log(self, msg):
        print(f"[{time.strftime('%H:%M:%S')} {self.a.planner}/{self.name}] {msg}", flush=True)

    def spawn(self, jid, job):
        pf = self.tmp / f"{jid}.txt"
        pf.write_text(self.make_prompt(job["keys"], job["attempt"]))
        out, rc = sh(["room-cli", "thread", "spawn", "--provider", "claude-code", "--model", self.a.model,
                      "--reasoning-level", self.a.reasoning, "--title", f"r6-{self.a.planner}-{self.name}-{jid}",
                      "--prompt-file", str(pf)])
        try:
            tid = json.loads(out)["id"]
        except (json.JSONDecodeError, KeyError, TypeError):
            self.log(f"spawn failed for {jid} (rc {rc}): {out[:200]}")
            return False
        job.update(status="live", thread=tid, t0=time.time())
        return True

    def finish(self, jid, job, reply):
        good, bad = self.accept(job["keys"], reply, job["attempt"])
        sh(["room-cli", "thread", "archive", job["thread"]])
        job["status"] = "done"; job["got"] = len(good)
        if good:
            self.st["ok"] += 1; self.fail_streak = 0
        else:
            self.fail_streak += 1
        if bad:
            if job["attempt"] < MAX_ATTEMPTS:
                for i in range(0, len(bad), self.a.batch):
                    self.add(bad[i:i + self.a.batch])
                    self.st["jobs"][f"j{self.st['next'] - 1}"]["attempt"] = job["attempt"] + 1
            else:
                job["failed_keys"] = bad
        self.log(f"{jid} back: {len(good)} ok, {len(bad)} bad (attempt {job['attempt']})")

    def run(self):
        t_start = time.time()
        while True:
            jobs = self.st["jobs"]
            live = [j for j, v in jobs.items() if v["status"] == "live"]
            todo = [j for j, v in jobs.items() if v["status"] == "todo"]
            if not live and not todo:
                break
            if self.st["ok"] >= self.a.ramp_after and self.cap < self.a.ramp_to and self.fail_streak == 0:
                self.cap = self.a.ramp_to; self.log(f"ramping to {self.cap} live")
            while len(live) < self.cap and todo:
                j = todo.pop(0)
                if not self.spawn(j, jobs[j]):
                    self.cap = max(1, self.cap // 2); time.sleep(60); break
                live.append(j); time.sleep(2)
            save_json(self.state_path, self.st)
            time.sleep(self.a.poll)
            for j in live:
                v = jobs[j]
                out, _ = sh(["room-cli", "thread", "output", v["thread"]])
                try:
                    d = json.loads(out)
                except json.JSONDecodeError:
                    continue
                reply = d.get("reply") or ""
                if d.get("state") not in TERMINAL or (not reply and time.time() - v["t0"] < 120):
                    if time.time() - v["t0"] > self.a.timeout * 60:
                        self.log(f"{j} timed out; stopping {v['thread']}")
                        sh(["room-cli", "thread", "stop", v["thread"]])
                        self.finish(j, v, "")
                    continue
                if d.get("truncated"):
                    full, _ = sh(["room-cli", "thread", "output", v["thread"], "--all"])
                    try:
                        reply = json.loads(full).get("reply") or reply
                    except json.JSONDecodeError:
                        pass
                self.finish(j, v, reply)
            if self.fail_streak >= 5 and self.cap > 1:
                self.cap = max(1, self.cap // 2); self.fail_streak = 0
                self.log(f"5 failures in a row: down to {self.cap} live, pausing 2 min"); time.sleep(120)
            save_json(self.state_path, self.st)
            if self.a.budget and time.time() - t_start > self.a.budget:
                n_live = sum(v["status"] == "live" for v in self.st["jobs"].values())
                self.log(f"PARTIAL: budget reached with {n_live} live; rerun the same command to resume")
                return False
        failed = [k for v in self.st["jobs"].values() for k in v.get("failed_keys", [])]
        self.log(f"DONE ({len(failed)} keys failed {MAX_ATTEMPTS} times: {failed[:10]})")
        return True


def slice_keys(a):
    return json.loads((lr.R6 / "slices" / f"{a.planner}.json").read_text())


def batches(keys, size):
    return [keys[i:i + size] for i in range(0, len(keys), size)]


def queue(fl, keys, size):
    """Add jobs for `keys` that no queued or live job already covers (so reruns are idempotent)."""
    busy = {k for v in fl.st["jobs"].values() if v["status"] in ("todo", "live") for k in v["keys"]}
    todo = [k for k in keys if k not in busy]
    for b in batches(todo, size):
        fl.add(b)
    return len(todo)


def cmd_label(a):
    ev = lr.load_gz("evidence.json.gz")
    keys = slice_keys(a)
    out_p = a.dir / f"labels_{a.version}.json"
    labels = load_json(out_p, {})
    bl = batches(keys, a.batch)
    chosen = [int(x) for x in a.batches.split(",")] if a.batches else range(len(bl))

    def make_prompt(ks, attempt):
        p = lr.worker_prompt(ks, ev)
        return p.replace("\n\n=== ", RETRY_HINT + "=== ", 1) if a.version != "v1" else p

    def accept(ks, reply, attempt):
        d = lr.parse_json_reply(reply) or {}
        good, bad = {}, []
        for k in ks:
            lab = lr.clean_label(d.get(k), ev["poles"][k]["sign"])
            if lab:
                good[k] = dict(lab, batch=keys.index(k) // a.batch, version=a.version)
            else:
                bad.append(k)
        labels.update(good); save_json(out_p, labels)
        return good, bad

    fl = Fleet(a, f"label_{a.version}", make_prompt, accept)
    n = sum(queue(fl, [k for k in bl[b] if k not in labels], a.batch) for b in chosen)
    fl.log(f"queued {n} poles ({len(labels)} already labelled)")
    ok = fl.run()
    save_json(out_p, labels)
    fl.log(f"labels_{a.version}.json: {len(labels)} poles")
    return ok


def labels_for(a, version):
    return load_json(a.dir / ("labels_final.json" if version == "final" else f"labels_{version}.json"), {})


def cmd_score(a):
    ev = lr.load_gz("evidence.json.gz"); ho = lr.load_gz("heldout.json.gz")
    keys = slice_keys(a)
    labels = labels_for(a, a.version)
    if a.mode == "shuffled":
        smap_p = a.dir / "shuffle_map.json"
        smap = load_json(smap_p, None) or lr.shuffle_map(keys, ev)
        save_json(smap_p, smap)
    tag = f"{a.split}_{a.mode}_{a.version}" + (f"_{a.suffix}" if a.suffix else "")
    out_p = a.dir / f"scores_{tag}.json"
    if a.keys:
        keys = [k for k in keys if k in set(a.keys.split(","))]
    scores = load_json(out_p, {})
    m = len(ho["poles"][keys[0]][a.split]["idx"])

    def lab(k):
        if a.mode == "wordlist":
            return lr.wordlist_label(ev["poles"][k])
        return labels[smap[k]] if a.mode == "shuffled" else labels[k]

    def make_prompt(ks, attempt):
        return lr.scorer_prompt([(k, lab(k)) for k in ks], ho, a.split)

    def accept(ks, reply, attempt):
        d = lr.parse_json_reply(reply) or {}
        good, bad = {}, []
        for k in ks:
            p = lr.clean_preds(d.get(k), m)
            (good.__setitem__(k, p) if p else bad.append(k))
        # a lazy reply (nearly every pole flat, e.g. all 0) is rescored, but the last attempt is kept as
        # it is, so no pole silently drops out of a condition
        if attempt < MAX_ATTEMPTS and good and sum(len(set(p)) == 1 for p in good.values()) >= 0.8 * len(good):
            return {}, list(ks)
        scores.update(good); save_json(out_p, scores)
        return good, bad

    fl = Fleet(a, f"score_{tag}", make_prompt, accept)
    need = [k for k in keys if k not in scores and (a.mode == "wordlist" or
                                                    (smap.get(k) if a.mode == "shuffled" else k) in labels)]
    fl.log(f"queued {queue(fl, need, a.batch)} poles ({a.split}, {a.mode} labels, version {a.version})")
    ok = fl.run()
    fl.log(f"scores_{tag}.json: {len(scores)} poles")
    return ok


def grade(dirs, split, version, keys=None):
    """Per-pole Pearson r of each scored condition vs the held-out truth, plus the token-lookup baseline."""
    ho = lr.load_gz("heldout.json.gz")
    per = {}
    for d in dirs:
        conds = {c: load_json(d / f"scores_{split}_{c}_{version}.json", {}) for c in ("real", "shuffled", "wordlist")}
        for k, preds in conds["real"].items():
            if keys is not None and k not in keys:
                continue
            h = ho["poles"][k][split]
            per[k] = {"real": lr.pearson(preds, h["true"]), "lookup": lr.pearson(h["lookup"], h["true"])}
            for c in ("shuffled", "wordlist"):
                if k in conds[c]:
                    per[k][c] = lr.pearson(conds[c][k], h["true"])
    return per


def summarize(per, log=print):
    ks = list(per)
    if not ks:
        log("nothing scored yet"); return {}
    res = {"n": len(ks)}
    for c in ("real", "shuffled", "wordlist", "lookup"):
        v = [per[k][c] for k in ks if c in per[k]]
        if v:
            res[c] = round(sum(v) / len(v), 4)
    for c in ("shuffled", "wordlist", "lookup"):
        gap = [per[k]["real"] - per[k][c] for k in ks if c in per[k]]
        if len(gap) > 1:
            m, lo, hi = lr.bootstrap_ci(gap)
            res[f"real_minus_{c}"] = {"n": len(gap), "mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
                                      "pass": lo > 0}
    log(json.dumps(res))
    return res


def cmd_grade(a):
    per = grade([a.dir], a.split, a.version)
    res = summarize(per)
    keys = slice_keys(a)
    labels = labels_for(a, a.version)
    rows = []
    for b, bk in enumerate(batches(keys, a.batch)):
        r = [per[k]["real"] for k in bk if k in per]
        if r:
            rows.append((sum(r) / len(r), b, len(r)))
    for mean_r, b, n in sorted(rows):
        print(f"  batch {b:3d}: mean {a.split} r {mean_r:+.3f} over {n} poles")
    verdicts = {}
    for k, v in per.items():
        verdicts.setdefault(labels.get(k, {}).get("verdict", "?"), []).append(v["real"])
    print("  by verdict: " + ", ".join(f"{vd} {sum(x)/len(x):+.3f} (n={len(x)})" for vd, x in sorted(verdicts.items())))
    save_json(a.dir / f"grade_{a.split}_{a.version}.json", {"summary": res, "per_pole": per})


def cmd_pick(a):
    """Final label per pole = the version with the best dev r (ties keep the earlier version). A pole
    with no dev score in any version keeps its first label, marked dev_r None."""
    from collections import Counter
    best = {}
    for v in a.versions.split(","):
        per = grade([a.dir], "dev", v)
        for k, lab in labels_for(a, v).items():
            r = per.get(k, {}).get("real")
            if k not in best or (r is not None and (best[k][0] is None or r > best[k][0] + 1e-9)):
                best[k] = (r, dict(lab, version=v, dev_r=None if r is None else round(r, 4)))
    final = {k: best[k][1] for k in slice_keys(a) if k in best}
    save_json(a.dir / "labels_final.json", final)
    print(f"labels_final.json: {len(final)} poles; versions {dict(Counter(x['version'] for x in final.values()))}; "
          f"{sum(x['dev_r'] is None for x in final.values())} without a dev score")


def cmd_show(a):
    """QC view of one pole: the evidence block its labeller saw, every label version, and its scores."""
    ev = lr.load_gz("evidence.json.gz")
    e = ev["poles"][a.key]
    print(f"S={e['S']} rank={e['rank']} p={e['p']:.4g} top_tokens={e['top_tokens']}")
    print(lr.pole_block(a.key, e))
    for p in sorted(a.dir.glob("labels_*.json")):
        lab = json.loads(p.read_text()).get(a.key)
        if lab:
            print(f"{p.stem}: {json.dumps(lab, ensure_ascii=False)}")
    for p in sorted(a.dir.glob("grade_*.json")):
        r = json.loads(p.read_text())["per_pole"].get(a.key)
        if r:
            print(f"{p.stem}: " + " ".join(f"{c}={v:+.3f}" for c, v in r.items()))


def cmd_merge(a):
    """Orchestrator: combine every planner's final labels and test scores into atlas/r6/labels.json and
    the global E4 summary (atlas/r6/e4.json). Featured order = verdict (clear, likely, mixed), then S."""
    ev = lr.load_gz("evidence.json.gz")
    dirs = sorted(p for p in (lr.R6 / "out").glob("p*") if p.is_dir())
    per = grade(dirs, "test", "final")
    out = {}
    for d in dirs:
        for k, lab in load_json(d / "labels_final.json", {}).items():
            e = ev["poles"][k]
            out[k] = {**{f: lab[f] for f in ("name", "description", "pole", "verdict", "concepts")},
                      "layer": e["layer"], "neuron": e["neuron"], "S": e["S"], "p": e["p"], "rank": e["rank"],
                      "planner": d.name, "version": lab.get("version"), "dev_r": lab.get("dev_r"),
                      "test": {c: round(v, 4) for c, v in per.get(k, {}).items()}}
    order = {v: i for i, v in enumerate(lr.VERDICTS)}
    for i, k in enumerate(sorted(out, key=lambda k: (order[out[k]["verdict"]], out[k]["rank"]))):
        out[k]["feature_rank"] = i
    missing = [k for k in ev["poles"] if k not in out]
    res = {"labelled": len(out), "poles": len(ev["poles"]), "missing": missing,
           "all": summarize({k: per[k] for k in out if k in per}, log=lambda *_: None)}
    for vd in lr.VERDICTS:
        res[vd] = summarize({k: per[k] for k in out if k in per and out[k]["verdict"] == vd}, log=lambda *_: None)
    for name, lo, hi in (("L27", 27, 27), ("L25-26", 25, 26), ("L0-24", 0, 24)):
        res[name] = summarize({k: per[k] for k in out if k in per and lo <= out[k]["layer"] <= hi},
                              log=lambda *_: None)
    save_json(lr.R6 / "labels.json", dict(sorted(out.items(), key=lambda kv: kv[1]["feature_rank"])))
    save_json(lr.R6 / "e4.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "missing"}, indent=1)[:3000])
    print(f"labels.json: {len(out)}/{len(ev['poles'])} poles; {len(missing)} missing")


def cmd_status(a):
    for p in sorted(a.dir.glob("state_*.json")):
        st = json.loads(p.read_text())
        c = {}
        for v in st["jobs"].values():
            c[v["status"]] = c.get(v["status"], 0) + 1
        print(f"{p.stem}: {c} ok={st['ok']}")
    for p in sorted(a.dir.glob("labels_*.json")) + sorted(a.dir.glob("scores_*.json")):
        print(f"{p.name}: {len(json.loads(p.read_text()))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("label", "score", "grade", "pick", "show", "status", "merge"):
        s = sub.add_parser(name)
        s.add_argument("--planner", required=name != "merge", default="all",
                       help="slice name, e.g. p1 (atlas/r6/slices/p1.json)")
        s.add_argument("--batch", type=int, default=20 if name in ("label", "grade") else 15)
        s.add_argument("--model", default="claude-haiku-5-5"); s.add_argument("--reasoning", default="high")
        s.add_argument("--max-live", type=int, default=8); s.add_argument("--ramp-to", type=int, default=25)
        s.add_argument("--ramp-after", type=int, default=8); s.add_argument("--poll", type=int, default=30)
        s.add_argument("--timeout", type=int, default=40, help="minutes before a sub-agent counts as stuck")
        s.add_argument("--budget", type=int, default=0, help="seconds; exit (resumably) after this long")
        if name in ("label", "score", "grade"):
            s.add_argument("--version", required=True)
        if name == "label":
            s.add_argument("--batches", default="", help="comma list of slice batch indices (default all)")
        if name in ("score", "grade"):
            s.add_argument("--split", choices=["dev", "test"], required=True)
        if name == "score":
            s.add_argument("--mode", choices=["real", "shuffled", "wordlist"], required=True)
            s.add_argument("--keys", default="", help="comma list: score only these poles")
            s.add_argument("--suffix", default="", help="tag suffix for a side run, e.g. opus")
        if name == "show":
            s.add_argument("--key", required=True)
        if name == "pick":
            s.add_argument("--versions", required=True)
    a = ap.parse_args()
    a.dir = lr.R6 / "out" / a.planner
    if a.cmd != "merge":
        a.dir.mkdir(parents=True, exist_ok=True)
    ok = {"label": cmd_label, "score": cmd_score, "grade": cmd_grade, "pick": cmd_pick, "show": cmd_show,
          "status": cmd_status, "merge": cmd_merge}[a.cmd](a)
    sys.exit(0 if ok in (None, True) else 3)
