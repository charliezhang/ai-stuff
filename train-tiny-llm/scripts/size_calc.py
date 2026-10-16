"""Param / FLOP / wall-clock table for model presets on the Orin.

Orin AGX 64GB GPU: ~42 TFLOPS dense BF16 peak (tensor cores). We assume 35% MFU
until we measure it in step 4.
"""
import sys
sys.path.insert(0, ".")
from tiny_llm.config import PRESETS

ORIN_PEAK_TFLOPS = 42.5
MFU = 0.35
TOKENS_PER_EPOCH = 100e6

print(f"{'preset':8} {'blocks':>8} {'emb+head':>9} {'total':>8} {'GFLOP/tok':>10} {'h/epoch':>8} {'tok/param':>9}")
for name, c in PRESETS.items():
    p = c.param_counts()
    f = c.train_flops_per_token()
    hours = f * TOKENS_PER_EPOCH / (ORIN_PEAK_TFLOPS * 1e12 * MFU) / 3600
    print(f"{name:8} {p['blocks']/1e6:7.1f}M {(p['embedding']+p['lm_head'])/1e6:8.1f}M "
          f"{p['total']/1e6:7.1f}M {f/1e9:10.2f} {hours:8.2f} {TOKENS_PER_EPOCH/p['total']:9.1f}")
