"""Model backends. Each runs one sequence through a decoder with a KV cache and returns the
neuron vector (every layer's mlp.down_proj input) at every position it processes.

A backend implements:
    name, model, layers, neurons, stop_ids
    encode_chat(prompt) -> ids     the prompt in the chat template, ready to generate
    encode_text(text) -> ids       plain text, no template (for calibration)
    decode(ids) -> str
    reset()                        clear the KV cache
    forward(ids) -> (logits, acts) append ids to the cache; logits: float32 [vocab] for the
                                   last position; acts: float16 [len(ids), layers, neurons]
"""
import time

import numpy as np

F16_MAX = 65504.0


class Fake:
    """No model. Byte-level tokens (so multi-byte characters split across tokens) and
    random-ish activations. For tests and UI work."""

    name, model = "fake", "fake-bytes"
    layers, neurons = 28, 3072
    EOS = 256
    stop_ids = {EOS}

    def __init__(self, delay=0.02):
        self.delay = delay
        rng = np.random.default_rng(0)
        self.scale = rng.lognormal(0, 0.5, (self.layers, self.neurons)).astype(np.float32)
        self.allowed = np.frombuffer(b"abcdefghijklmnopqrstuvwxyz ", np.uint8)
        self.pos = 0

    def encode_chat(self, prompt):
        return list(prompt.encode("utf-8"))

    encode_text = encode_chat

    def decode(self, ids):
        return bytes(i for i in ids if i < 256).decode("utf-8", "replace")

    def reset(self):
        self.pos = 0

    def forward(self, ids):
        acts = np.empty((len(ids), self.layers, self.neurons), np.float16)
        for k, t in enumerate(ids):
            rng = np.random.default_rng((t, self.pos))
            acts[k] = rng.standard_normal((self.layers, self.neurons), np.float32) * self.scale
            self.pos += 1
        time.sleep(self.delay)  # per call, so prefill is one step like a real model
        logits = np.full(257, -1e9, np.float32)
        logits[self.allowed] = rng.standard_normal(len(self.allowed))
        logits[32] += 1.5  # spaces, so the text looks like words
        logits[self.EOS] = 10.0 if rng.random() < 0.02 else -1e9
        return logits, acts


class Cpu:
    """transformers on the CPU, fp32."""

    name = "cpu"

    def __init__(self, model="Qwen/Qwen3-0.6B"):
        import os

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

        torch.set_num_threads(os.cpu_count() or 1)
        self.torch, self.DynamicCache = torch, DynamicCache
        self.model = model
        self.tok = AutoTokenizer.from_pretrained(model)
        self.lm = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32).eval()
        blocks = self.lm.model.layers
        self.layers, self.neurons = len(blocks), self.lm.config.intermediate_size
        self.stop_ids = set(self.tok.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"]))
        self._acts = [None] * self.layers
        for l, block in enumerate(blocks):
            block.mlp.down_proj.register_forward_pre_hook(self._hook(l))
        self.reset()

    def _hook(self, l):
        def hook(module, args):
            self._acts[l] = args[0][0]
        return hook

    def encode_chat(self, prompt):
        return self.tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                            enable_thinking=False, tokenize=True, return_dict=True)["input_ids"]

    def encode_text(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids):
        return self.tok.decode(ids)

    def reset(self):
        self.cache = self.DynamicCache(config=self.lm.config)

    def forward(self, ids):
        with self.torch.inference_mode():  # grad mode is per thread, so set it here
            return self._forward(ids)

    def _forward(self, ids):
        out = self.lm(input_ids=self.torch.tensor([ids]), past_key_values=self.cache, use_cache=True,
                      logits_to_keep=1)
        acts = self.torch.stack(self._acts, dim=1).clamp(-F16_MAX, F16_MAX).half().numpy()
        return out.logits[0, -1].float().numpy(), acts


BACKENDS = {"fake": Fake, "cpu": Cpu}
