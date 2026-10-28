# Minimal RunPod GPU pod helper for atlas jobs (REST v2). The cloud sandbox can't make outbound SSH, so the
# pod runs sshd behind websocat on port 8080, and we reach it through RunPod's HTTPS proxy (docs/CONTEXT.md).
# The API key comes from a dotenv file outside the repo. Every pod is deleted with `down`; `cost` shows spend.
#
#   python3 atlas/pod.py gpus                       # community GPU prices
#   python3 atlas/pod.py up --gpu "NVIDIA RTX A4000"   # -> pod id in /code/work/pod.json
#   python3 atlas/pod.py ssh "nvidia-smi"           # run a command on the pod
#   python3 atlas/pod.py put LOCAL_DIR REMOTE_DIR   # tar-stream a directory up; `get REMOTE LOCAL` for one file
#   python3 atlas/pod.py down                       # delete the pod
import argparse, json, os, shlex, subprocess, sys, time, urllib.request
from pathlib import Path

API = "https://api.runpod.io/v2"
STATE = Path("/code/work/pod.json")
KEY = Path.home() / ".ssh" / "r8_pod"
WEBSOCAT = Path.home() / ".local" / "bin" / "websocat"
BOOT = ("set -e; apt-get update -qq; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server curl "
        "ca-certificates python3 >/dev/null; mkdir -p /root/.ssh /run/sshd; echo \"$PUBLIC_KEY\" > "
        "/root/.ssh/authorized_keys; chmod 600 /root/.ssh/authorized_keys; /usr/sbin/sshd; curl -sSL -o "
        "/usr/local/bin/websocat https://github.com/vi/websocat/releases/download/v1.14.1/"
        "websocat.x86_64-unknown-linux-musl; chmod +x /usr/local/bin/websocat; "
        "exec websocat --binary ws-l:0.0.0.0:8080 tcp:127.0.0.1:22")


def key():
    for line in Path(os.environ.get("RUNPOD_ENV", "/code/work/.env.runpod")).read_text().splitlines():
        if line.startswith("RUNPOD_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("RUNPOD_API_KEY not found")


def api(method, path, body=None):
    req = urllib.request.Request(API + path, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"Bearer {key()}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        t = r.read()
    return json.loads(t) if t else {}


def ssh_args(pid):
    proxy = f"{WEBSOCAT} --binary wss://{pid}-8080.proxy.runpod.net"
    return ["ssh", "-i", str(KEY), "-o", f"ProxyCommand={proxy}", "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", "-o", "ServerAliveInterval=30",
            "root@runpod"]


def pod():
    return json.loads(STATE.read_text())


def cmd_gpus(a):
    for g in sorted(api("GET", "/catalog/gpus")["gpus"], key=lambda g: g["price"].get("community") or 99):
        if g.get("community") and g["memory"] >= 16:
            print(f'{g["price"]["community"]:5.2f} $/h  {g["memory"]:3d} GB  {g["availability"]:7s} {g["id"]}')


def cmd_up(a):
    if STATE.exists():
        raise SystemExit(f"a pod is already recorded in {STATE}; run `down` first")
    body = {"name": a.name, "image": "ubuntu:24.04", "cloud": "COMMUNITY", "gpu": {"id": a.gpu, "count": 1},
            "disk": a.disk, "ports": ["8080/http"], "env": {"PUBLIC_KEY": (KEY.with_suffix(".pub")).read_text().strip()},
            "cmd": ["bash", "-c", BOOT]}
    p = api("POST", "/pods", body)
    STATE.write_text(json.dumps({"id": p["id"], "gpu": a.gpu, "created": time.time(), "cost_per_hr": p.get("cost")}))
    print(f"pod {p['id']} ({a.gpu}) created, {p.get('cost')} $/h; waiting for ssh...", flush=True)
    for _ in range(60):
        time.sleep(10)
        r = subprocess.run(ssh_args(p["id"]) + ["echo ok"], capture_output=True, text=True, timeout=60)
        if r.stdout.strip() == "ok":
            print("ssh ready", flush=True); return
    print("ssh not ready after 10 min; check the pod, or run `down`", flush=True)


def cmd_ssh(a):
    sys.exit(subprocess.run(ssh_args(pod()["id"]) + [a.command]).returncode)


def cmd_put(a):
    tar = subprocess.Popen(["tar", "-C", a.local, "-czf", "-", "."], stdout=subprocess.PIPE)
    r = subprocess.run(ssh_args(pod()["id"]) + [f"mkdir -p {shlex.quote(a.remote)} && tar -C {shlex.quote(a.remote)} -xzf -"],
                       stdin=tar.stdout)
    tar.wait(); sys.exit(r.returncode)


def cmd_get(a):
    with open(a.local, "wb") as f:
        sys.exit(subprocess.run(ssh_args(pod()["id"]) + [f"cat {shlex.quote(a.remote)}"], stdout=f).returncode)


def cmd_cost(a):
    p = pod(); hrs = (time.time() - p["created"]) / 3600
    print(f'pod {p["id"]}: {hrs:.2f} h x {p["cost_per_hr"]} $/h = ${hrs * (p["cost_per_hr"] or 0):.2f}')


def cmd_down(a):
    p = pod()
    api("DELETE", f"/pods/{p['id']}")
    hrs = (time.time() - p["created"]) / 3600
    print(f'deleted pod {p["id"]} after {hrs:.2f} h (~${hrs * (p["cost_per_hr"] or 0):.2f})')
    STATE.unlink()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gpus")
    s = sub.add_parser("up"); s.add_argument("--gpu", required=True); s.add_argument("--disk", type=int, default=40)
    s.add_argument("--name", default="lobotomy-atlas")
    s = sub.add_parser("ssh"); s.add_argument("command")
    s = sub.add_parser("put"); s.add_argument("local"); s.add_argument("remote")
    s = sub.add_parser("get"); s.add_argument("remote"); s.add_argument("local")
    sub.add_parser("cost"); sub.add_parser("down")
    a = ap.parse_args()
    {"gpus": cmd_gpus, "up": cmd_up, "ssh": cmd_ssh, "put": cmd_put, "get": cmd_get, "cost": cmd_cost,
     "down": cmd_down}[a.cmd](a)
