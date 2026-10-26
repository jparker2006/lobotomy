"""lobotomy server: streams tokens plus every MLP neuron per token over long-poll HTTP.

Run:  python server/server.py --backend cpu     (then open http://localhost:8000)

Protocol
  GET  /api/meta          {layers, neurons, backend, model, max_tokens}
  GET  /api/calibration   float32 LE [layers*neurons]: per-neuron p99 of |activation|
  POST /api/generate      {prompt} -> {id}. A new generation cancels the old one.
  GET  /api/frames?id=ID&from=N
                          waits up to 1 s for frame N, then returns frames N.. as binary.
                          Each frame: u32 LE header length, UTF-8 JSON header
                          {i, kind: prompt|gen|done, text, n}, then n payload bytes:
                          float16 LE [layers, neurons], the down_proj input while the token
                          in `text` is the input. `done` has n=0 plus tok_s, n_prompt, n_gen.
  GET  /                  the page (web/dist)
"""
import argparse
import hashlib
import json
import mimetypes
import struct
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

import backends

HERE = Path(__file__).resolve().parent
WEB = HERE.parent / "web" / "dist"
MAX_TOKENS = 256  # the board's fixed context: prompt plus answer
REPETITION_PENALTY = 1.1
POLL_SECONDS = 1.0


def frame(header, payload=b""):
    h = json.dumps({**header, "n": len(payload)}, ensure_ascii=False).encode()
    return struct.pack("<I", len(h)) + h + payload


class Detokenizer:
    """Decodes token by token. Holds back text while a multi-byte character is incomplete."""

    def __init__(self, backend):
        self.backend, self.ids, self.out = backend, [], ""

    def push(self, token):
        self.ids.append(token)
        text = self.backend.decode(self.ids)
        if text.endswith("�"):
            return ""
        new, self.out = text[len(self.out):], text
        return new

    def flush(self):
        text = self.backend.decode(self.ids)
        new, self.out = text[len(self.out):], text
        return new


def pick(logits, seen):
    """Greedy, with transformers-style repetition penalty over every token seen so far."""
    idx = np.fromiter(seen, np.int64)
    v = logits[idx]
    logits[idx] = np.where(v > 0, v / REPETITION_PENALTY, v * REPETITION_PENALTY)
    return int(np.argmax(logits))


def calibration(backend, cache_dir):
    """Per-neuron 99th percentile of |activation| over the built-in text, skipping position 0
    of each chunk (the attention sink). Cached on disk by backend, model and text."""
    text = (HERE / "calibration.txt").read_text()
    key = hashlib.sha256(f"{backend.name}|{backend.model}|{MAX_TOKENS}|{text}".encode()).hexdigest()[:16]
    path = Path(cache_dir) / f"calibration-{backend.name}-{backend.model.replace('/', '_')}-{key}.f32"
    if path.exists():
        return path.read_bytes()
    ids = backend.encode_text(text)[:4 * MAX_TOKENS]
    acts = []
    for s in range(0, len(ids), MAX_TOKENS):
        backend.reset()
        acts.append(np.abs(backend.forward(ids[s:s + MAX_TOKENS])[1][1:].astype(np.float32)))
    acts = np.concatenate(acts)
    assert len(acts) >= 512, f"calibration text gave only {len(acts)} tokens"
    p99 = np.percentile(acts.reshape(len(acts), -1), 99, axis=0)
    data = np.maximum(p99, 1e-6).astype("<f4").tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


class Generation:
    def __init__(self, prompt):
        self.id, self.prompt = uuid.uuid4().hex[:12], prompt
        self.frames, self.done, self.cancelled = [], False, False
        self.cond = threading.Condition()

    def emit(self, kind, text, acts=None, **extra):
        payload = b"" if acts is None else acts.astype("<f2", copy=False).tobytes()
        with self.cond:
            self.frames.append(frame({"i": len(self.frames), "kind": kind, "text": text, **extra}, payload))
            self.done = kind == "done"
            self.cond.notify_all()

    def wait(self, start, timeout):
        with self.cond:
            self.cond.wait_for(lambda: len(self.frames) > start or self.done, timeout)
            return b"".join(self.frames[start:])


class Engine:
    """One generation at a time. A new one cancels the old."""

    def __init__(self, backend):
        self.backend = backend
        self.lock = threading.Lock()  # guards self.current
        self.run_lock = threading.Lock()  # the backend runs one generation at a time
        self.current = None

    def start(self, prompt):
        ids = self.backend.encode_chat(prompt)
        if len(ids) >= MAX_TOKENS:
            raise ValueError(f"prompt is {len(ids)} tokens; the limit is {MAX_TOKENS - 1}")
        gen = Generation(prompt)
        with self.lock:
            if self.current:
                self.current.cancelled = True
            self.current = gen
        threading.Thread(target=self._run, args=(gen, ids), daemon=True).start()
        return gen

    def get(self, gen_id):
        with self.lock:
            return self.current if self.current and self.current.id == gen_id else None

    def _run(self, gen, ids):
        with self.run_lock:
            try:
                self._generate(gen, ids)
            except Exception as e:  # never leave a client polling forever
                gen.emit("done", "", tok_s=0, n_prompt=len(ids), n_gen=0, error=repr(e))
                raise

    def _generate(self, gen, ids):
        b, n_gen, tok_s = self.backend, 0, 0.0
        if not gen.cancelled:
            b.reset()
            logits, acts = b.forward(ids)
            prompt_text = Detokenizer(b)
            for k, t in enumerate(ids):
                gen.emit("prompt", prompt_text.push(t), acts[k])
            seen, answer, t0 = set(ids), Detokenizer(b), time.perf_counter()
            while len(ids) < MAX_TOKENS and not gen.cancelled:
                t = pick(logits, seen)
                if t in b.stop_ids:
                    break
                ids.append(t)
                seen.add(t)
                logits, acts = b.forward([t])
                n_gen += 1
                gen.emit("gen", answer.push(t), acts[0])
            tok_s = n_gen / (time.perf_counter() - t0) if n_gen else 0.0
            tail = answer.flush()
        else:
            tail = ""
        gen.emit("done", tail, tok_s=round(tok_s, 2), n_prompt=len(ids) - n_gen, n_gen=n_gen,
                 cancelled=gen.cancelled)


def make_handler(engine, meta, calib):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def send(self, code, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/api/meta":
                return self.send(200, meta)
            if url.path == "/api/calibration":
                return self.send(200, calib, "application/octet-stream")
            if url.path == "/api/frames":
                q = parse_qs(url.query)
                gen = engine.get(q.get("id", [""])[0])
                if gen is None:
                    return self.send(404, {"error": "unknown or replaced generation"})
                try:
                    start = max(0, int(q.get("from", ["0"])[0]))
                except ValueError:
                    return self.send(400, {"error": "bad from"})
                return self.send(200, gen.wait(start, POLL_SECONDS), "application/octet-stream")
            return self.static(url.path)

        def static(self, path):
            f = (WEB / path.lstrip("/")).resolve()
            if f.is_dir():
                f = f / "index.html"
            if not f.is_relative_to(WEB) or not f.is_file():
                if not WEB.exists():
                    return self.send(404, b"web/dist is missing: run `npm install && npm run build` in web/", "text/plain")
                return self.send(404, b"not found", "text/plain")
            self.send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or "application/octet-stream")

        def do_POST(self):
            if urlparse(self.path).path != "/api/generate":
                return self.send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                prompt = body["prompt"]
                assert isinstance(prompt, str) and prompt.strip()
            except Exception:
                return self.send(400, {"error": "send JSON {\"prompt\": \"...\"}"})
            try:
                gen = engine.start(prompt)
            except ValueError as e:
                return self.send(400, {"error": str(e)})
            self.send(200, {"id": gen.id})

    return Handler


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", choices=sorted(backends.BACKENDS), default="cpu")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B", help="cpu backend only")
    p.add_argument("--fake-delay", type=float, default=0.02, help="fake backend: seconds per token")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "lobotomy"))
    a = p.parse_args(argv)

    if a.backend == "fake":
        backend = backends.Fake(a.fake_delay)
    else:
        backend = backends.BACKENDS[a.backend](a.model)
    t0 = time.perf_counter()
    calib = calibration(backend, a.cache_dir)
    print(f"calibration ready in {time.perf_counter() - t0:.1f} s", file=sys.stderr)
    meta = {"layers": backend.layers, "neurons": backend.neurons, "backend": backend.name,
            "model": backend.model, "max_tokens": MAX_TOKENS}
    server = ThreadingHTTPServer((a.host, a.port), make_handler(Engine(backend), meta, calib))
    server.daemon_threads = True
    print(f"lobotomy ({backend.name}, {backend.model}) on http://{a.host}:{server.server_port}", file=sys.stderr, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
