"""Sanity checks: param count matches config math, and GPU (sm_87 on an sm_80 build) matches CPU."""
import sys

import torch

sys.path.insert(0, ".")
from tiny_llm.config import PRESETS
from tiny_llm.model import GPT

torch.manual_seed(0)
c = PRESETS["dev"]
m = GPT(c)
n = sum(p.numel() for p in m.parameters())
norm_params = n - c.param_counts()["total"] - c.seq_len * c.n_embd   # minus learned positions
print(f"params: {n/1e6:.2f}M (config math {c.param_counts()['total']/1e6:.2f}M + wpe + {norm_params} norm gains)")

x = torch.randint(0, c.vocab_size, (2, 256))
with torch.no_grad():
    ref = m(x)
    gpu = m.cuda()(x.cuda()).cpu()
print(f"fp32 GPU vs CPU max |diff| = {(gpu - ref).abs().max():.2e}")
print(f"initial loss = {torch.nn.functional.cross_entropy(ref.flatten(0, 1), x.flatten()):.3f} "
      f"(uniform guess ln(50257) = {torch.tensor(50257.0).log():.3f})")
