"""End-to-end protocol tests. Run: python -m unittest discover server
The fake backend always runs. The cpu backend runs if transformers is installed (slower)."""
import importlib.util
import json
import re
import struct
import subprocess
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

SERVER = Path(__file__).resolve().parent / "server.py"
L, N = 28, 3072


def parse(buf):
    frames, o = [], 0
    while o < len(buf):
        (hl,) = struct.unpack_from("<I", buf, o)
        h = json.loads(buf[o + 4:o + 4 + hl].decode())
        o += 4 + hl
        acts = np.frombuffer(buf[o:o + h["n"]], "<f2").reshape(L, N) if h["n"] else None
        o += h["n"]
        frames.append((h, acts))
    return frames


class Server:
    def __init__(self, *args):
        self.cache = tempfile.TemporaryDirectory()
        self.proc = subprocess.Popen([sys.executable, str(SERVER), "--port", "0", "--cache-dir", self.cache.name, *args],
                                     stderr=subprocess.PIPE, text=True)
        for line in self.proc.stderr:
            if m := re.search(r"http://[\d.]+:(\d+)", line):
                self.url = f"http://127.0.0.1:{m.group(1)}"
                return
        raise RuntimeError("server did not start")

    def get(self, path):
        with urllib.request.urlopen(self.url + path, timeout=30) as r:
            return r.read()

    def generate(self, prompt):
        req = urllib.request.Request(self.url + "/api/generate", json.dumps({"prompt": prompt}).encode(), method="POST")
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["id"]

    def frames(self, gen_id, start=0):
        return parse(self.get(f"/api/frames?id={gen_id}&from={start}"))

    def run(self, gen_id):
        """Poll from 0 until done, the way the page does."""
        out = []
        while not out or out[-1][0]["kind"] != "done":
            out += self.frames(gen_id, len(out))
        return out

    def close(self):
        self.proc.terminate()
        self.proc.wait()
        self.proc.stderr.close()
        self.cache.cleanup()


class FakeProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = Server("--backend", "fake", "--fake-delay", "0.005")

    @classmethod
    def tearDownClass(cls):
        cls.s.close()

    def test_meta(self):
        meta = json.loads(self.s.get("/api/meta"))
        self.assertEqual(meta, {"layers": 28, "neurons": 3072, "backend": "fake", "model": "fake-bytes", "max_tokens": 256})

    def test_calibration(self):
        cal = np.frombuffer(self.s.get("/api/calibration"), "<f4")
        self.assertEqual(cal.size, L * N)
        self.assertTrue(np.all(np.isfinite(cal)) and np.all(cal > 0))

    def test_frames(self):
        prompt = "héllo 🧠 brain"  # multi-byte characters split across byte tokens
        frames = self.s.run(self.s.generate(prompt))
        heads = [h for h, _ in frames]
        self.assertEqual([h["i"] for h in heads], list(range(len(heads))))
        kinds = [h["kind"] for h in heads]
        n_prompt = len(prompt.encode())
        self.assertEqual(kinds[:n_prompt], ["prompt"] * n_prompt)
        self.assertEqual(set(kinds[n_prompt:-1]), {"gen"})
        self.assertEqual(kinds[-1], "done")
        self.assertEqual("".join(h["text"] for h in heads[:n_prompt]), prompt)
        self.assertEqual(heads[1]["text"], "")  # first byte of é is held back...
        self.assertEqual(heads[2]["text"], "é")  # ...until the character is complete
        done = heads[-1]
        self.assertNotIn("error", done)
        self.assertEqual((done["n"], done["n_prompt"], done["n_gen"]), (0, n_prompt, len(heads) - n_prompt - 1))
        self.assertGreater(done["tok_s"], 0)
        self.assertLessEqual(done["n_prompt"] + done["n_gen"], 256)
        for h, acts in frames[:-1]:
            self.assertEqual(h["n"], L * N * 2)
            self.assertTrue(np.all(np.isfinite(acts)))
        self.assertGreater(np.abs(frames[0][1].astype(np.float32) - frames[1][1]).mean(), 0.1)  # frames differ

    def test_from_paging(self):
        gen_id = self.s.generate("paging test")
        frames = self.s.run(gen_id)
        later = self.s.frames(gen_id, 5)
        self.assertEqual([h["i"] for h, _ in later], list(range(5, len(frames))))
        self.assertEqual(later[0][1].tobytes(), frames[5][1].tobytes())
        self.assertEqual(self.s.frames(gen_id, len(frames)), [])  # past the end of a finished run: empty

    def test_cancel_on_new_generate(self):
        old = self.s.generate("first question " * 10)
        new = self.s.generate("second")
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.s.frames(old)
        self.assertEqual(e.exception.code, 404)
        frames = self.s.run(new)
        self.assertEqual("".join(h["text"] for h, _ in frames if h["kind"] == "prompt"), "second")
        self.assertFalse(frames[-1][0]["cancelled"])

    def test_bad_requests(self):
        for body in (b"{}", b"not json", json.dumps({"prompt": "x" * 300}).encode()):
            req = urllib.request.Request(self.s.url + "/api/generate", body, method="POST")
            with self.assertRaises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(req, timeout=10)
            self.assertEqual(e.exception.code, 400)


@unittest.skipUnless(importlib.util.find_spec("transformers"), "transformers not installed")
class CpuRun(unittest.TestCase):
    def test_short_prompt(self):
        s = Server("--backend", "cpu")
        try:
            cal = np.frombuffer(s.get("/api/calibration"), "<f4")
            frames = s.run(s.generate("What is the capital of France? One word."))
        finally:
            s.close()
        done = frames[-1][0]
        self.assertNotIn("error", done)
        answer = "".join(h["text"] for h, _ in frames if h["kind"] in ("gen", "done"))
        self.assertIn("Paris", answer)
        self.assertGreater(done["n_gen"], 0)
        self.assertGreater(done["tok_s"], 0)
        self.assertEqual(cal.size, L * N)
        for h, acts in frames[1:-1]:  # skip position 0, the attention sink
            a = acts.astype(np.float32)
            self.assertTrue(np.all(np.isfinite(a)))
            self.assertTrue(np.all(a.std(axis=1) > 1e-4), "a layer is all constant")
            # brightness is calibrated: most neurons sit below their p99, a few above
            frac = (np.abs(a) > cal.reshape(L, N)).mean()
            self.assertGreater(frac, 0.0005)
            self.assertLess(frac, 0.2)


if __name__ == "__main__":
    unittest.main()
