# Usage: control.py LAYER DOSE COUNT
# Control: push the residual at layer L by dose x median norm along random directions (same size as a neuron push).
import sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32).eval()
L, d, k = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
prompt = "Describe your ideal Sunday in three sentences."
ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                              enable_thinking=False, return_tensors="pt", return_dict=True).input_ids
out = model(ids, output_hidden_states=True)
med = out.hidden_states[L + 1][0, 1:].norm(dim=-1).median().item()
g = torch.Generator().manual_seed(1)
vec = None
model.model.layers[L].mlp.down_proj.register_forward_hook(lambda m, i, o: o + vec)
for s in range(k):
    u = torch.randn(1024, generator=g); vec = d * med * u / u.norm()
    o = model.generate(ids, max_new_tokens=50, do_sample=False, repetition_penalty=1.1)
    print(f"--- random direction {s}, dose {d:+g} ---\n{tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).strip()}\n", flush=True)
