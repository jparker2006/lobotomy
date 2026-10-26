# Usage: probe.py ocean|food. Run with PyTorch + transformers (CPU is fine).
# Find candidate topic neurons in Qwen3-0.6B: contrastive activations + logit lens.
import sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32).eval()
layers = model.model.layers
L = len(layers)

acts = {}
def cap(l):
    def f(mod, inp):
        acts[l] = inp[0][0]  # [T, 3072]
    return f
hs = [layers[l].mlp.down_proj.register_forward_pre_hook(cap(l)) for l in range(L)]

def mean_acts(texts):
    tot = None
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids
        model(ids)
        m = torch.stack([acts[l].mean(0) for l in range(L)])
        tot = m if tot is None else tot + m
    return tot / len(texts)

topic = sys.argv[1]
topics = {
 "ocean": ["The waves crashed against the rocky shore as the tide came in.",
           "Whales and dolphins swim in the deep blue ocean.",
           "We sailed across the sea, salt spray on our faces.",
           "Coral reefs teem with fish beneath the ocean surface.",
           "The beach was covered in seashells after the storm at sea."],
 "food": ["I baked fresh bread and a chocolate cake this morning.",
          "The pasta was tossed in garlic, olive oil and parmesan.",
          "She ordered a burger with fries and a milkshake.",
          "Chop the onions, then fry them in butter until golden.",
          "Sushi, ramen and dumplings are my favorite dishes."],
}
neutral = ["The meeting was moved to Tuesday afternoon.",
           "He filled out the tax forms and mailed them.",
           "The software update fixed several bugs in the login page.",
           "Our team reviewed the quarterly budget report.",
           "She took the train to work and read the news.",
           "The committee will vote on the proposal next week."]
d = mean_acts(topics[topic]) - mean_acts(neutral)
for h in hs: h.remove()

W_U = model.lm_head.weight  # [V, 1024]
norm = model.model.norm
for l in range(6, L):
    v, n = d[l].abs().max(0)
    n = int(n); sign = float(d[l, n].sign())
    col = layers[l].mlp.down_proj.weight[:, n] * sign
    logits = W_U @ norm(col[None])[0]
    top = [tok.decode([i]) for i in logits.topk(12).indices]
    print(f"L{l:2d} n{n:4d} diff={float(d[l,n]):+.3f}  lens: {top}")
