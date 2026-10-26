# Push one MLP neuron in Qwen3-0.6B while it writes. Usage: steer.py LAYER NEURON DOSE [DOSE...]
# Push: h[n] += dose * median_resid_norm(L) / ||W_down[:, n]||  (moves the residual by dose x median norm)
import sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_grad_enabled(False); torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32).eval()
L, n = int(sys.argv[1]), int(sys.argv[2]); doses = [float(x) for x in sys.argv[3:]]
prompt = "Describe your ideal Sunday in three sentences."
ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                              enable_thinking=False, return_tensors="pt", return_dict=True).input_ids
layer = model.model.layers[L]

# median residual-stream norm after layer L (skip token 0, the attention sink)
out = model(ids, output_hidden_states=True)
med = out.hidden_states[L + 1][0, 1:].norm(dim=-1).median().item()
wn = layer.mlp.down_proj.weight[:, n].norm().item()

add = 0.0
h = layer.mlp.down_proj.register_forward_pre_hook(lambda m, i: (i[0].index_add(-1, torch.tensor([n]), torch.full_like(i[0][..., :1], add)),))
print(f"prompt: {prompt}\nL{L} n{n}  median resid norm {med:.1f}  |W_down col| {wn:.3f}\n")
for d in doses:
    add = d * med / wn
    g = model.generate(ids, max_new_tokens=70, do_sample=False, repetition_penalty=1.1)
    print(f"--- dose {d:+g} ---\n{tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip()}\n", flush=True)
