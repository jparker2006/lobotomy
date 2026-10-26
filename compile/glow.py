"""T6: add one output with every layer's down_proj input (the 86,016 neurons) and compile with the stock flags.

Runs on the RunPod x86 pod (see docs/CONTEXT.md, RunPod) after a default `torq-export-model qwen -s qwen3-0.6b --models-dir /work/models`.
Output: /work/glow/torq/transformer_glow.vmfb, with output 1 = `neurons` [1, 28, 3072] bf16.
"""
import json, os, shutil, sys, time
from pathlib import Path
import onnx
from onnx import TensorProto, helper

SRC = Path("/work/models/Qwen/Qwen3-0.6B/export/onnx/converted/static")
DST = Path("/work/glow/onnx"); OUT = Path("/work/glow/torq")
DST.mkdir(parents=True, exist_ok=True)
m = onnx.load(SRC / "model.onnx", load_external_data=False)
g = m.graph
names = [f"/model/layers.{l}/mlp/Mul_output_0" for l in range(28)]
produced = {o for n in g.node for o in n.output}
assert all(x in produced for x in names), "missing down_proj inputs"
down = [list(n.input)[0] for n in g.node if n.op_type == "MatMul" and n.name.endswith("mlp/down_proj/MatMul")]
assert sorted(down) == sorted(names), "down_proj inputs are not the Mul outputs"
g.node.append(helper.make_node("Concat", names, ["neurons"], axis=1, name="/lobotomy/neurons/Concat"))
out = helper.make_tensor_value_info("neurons", TensorProto.BFLOAT16, [1, 28, 3072])
outs = list(g.output); outs.insert(1, out)
del g.output[:]; g.output.extend(outs)
onnx.save(m, DST / "model.onnx")
for f in os.listdir(SRC):
    if f != "model.onnx" and not (DST / f).exists():
        shutil.copy2(SRC / f, DST / f)
onnx.checker.check_model(str(DST / "model.onnx"))
print("inputs", len(g.input), "outputs", len(g.output), "->", [o.name for o in g.output[:3]], flush=True)

sys.path.insert(0, "/work/torq-tools/src")
from torq.models.qwen.export import compile_torq_model
flags = json.load(open("/work/torq-tools/src/torq/models/qwen/compile_flags.json"))["2.3.0"]
args = flags["common"] + flags["transformer"]
print("flags", args, flush=True)
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
t0 = time.time()
compile_torq_model(DST / "model.onnx", OUT, opset=next(e.version for e in m.opset_import if e.domain in ("", "ai.onnx")), compiler_args=args)
os.replace(OUT / "model.vmfb", OUT / "transformer_glow.vmfb")
print(f"compiled in {time.time() - t0:.0f} s", flush=True)
