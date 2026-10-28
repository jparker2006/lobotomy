# Independent expected values from the recorded file: r = f16 activation / p99 (numpy), per token, for one neuron.
import json, struct, sys, numpy as np
d = sys.argv[1]; layer, neuron = int(sys.argv[2]), int(sys.argv[3])
buf = open(f"{d}/ocean.bin", "rb").read()
p99 = np.fromfile(f"{d}/calibration.f32", "<f4")
meta = json.load(open(f"{d}/meta.json")); N = meta["neurons"]; K = layer * N + neuron
o, toks = 0, []
while o < len(buf):
    (hl,) = struct.unpack_from("<I", buf, o); h = json.loads(buf[o+4:o+4+hl]); o += 4 + hl
    if h["n"]:
        a = np.frombuffer(buf, "<f2", h["n"] // 2, o)
        toks.append({"k": len(toks), "i": h["i"], "kind": h["kind"], "text": h["text"], "r": float(np.float32(a[K]) / p99[K])})
    o += h["n"]
gen = [t for t in toks if t["kind"] == "gen"]
best = sorted(gen, key=lambda t: (-abs(t["r"]), t["k"]))[:5]
print(json.dumps({"tokens": toks, "best": best}))
