# Design log

Each step: the decision, the alternatives considered, and why. Filled in as we go.

| # | Topic | Decision | Status |
|---|-------|----------|--------|
| 1 | Model size | `dev` preset only: 6L × 384d, ~49M total (10.6M non-embedding). Not leaderboard-comparable; optimized for fast ablations. | decided |
| 2 | Training data | Slowrun FineWeb sample-10BT: 100M train / 10M val tokens, hash-verified `.pt` files; multi-epoch. | decided |
| 3 | Tokenizer | GPT-2 BPE (tiktoken), vocab 50257 padded to 50304, untied embeddings. Log BPB alongside loss. | decided |
| 4 | Numeric precision | BF16 autocast for matmuls; fp32 master weights, AdamW state, norms/softmax/loss. Measured on Orin: bf16 26–29 TFLOPS vs fp32 3.3; TF32 gave no speedup; FP8 unsupported (needs sm_89+); 192 GB/s bandwidth → fuse elementwise ops. | decided |
| 5 | Normalization | Pre-norm RMSNorm with learnable gain, final RMSNorm before lm_head; QK-norm off (flag available). Chosen from literature, no ablation: robust identity residual path, one fewer reduction than LayerNorm on a bandwidth-bound GPU. | decided |
| 6 | Activation | SwiGLU, hidden = 2/3 · 4d rounded to 64 (1024 for dev) — param/FLOP-matched to a 4d GELU MLP. | decided |
| 7 | Positional encoding (RoPE) | Full RoPE, base 10k, all 32 pairs rotate, half-split layout, applied to q/k after (optional) QK-norm. No learned positions (-0.79M params). | decided |
| 8 | Aspect ratio (depth vs width) | 6 layers × 384 width, 6 heads × 64 head_dim (d/L = 64, GPT-2/slowrun range). Noted: lm_head is 56% of training FLOPs at this width. | decided |
| 9 | MoE | Dense by default; optional coarse Mixtral-style MoE (`n_experts=8`, top-2, each expert a full SwiGLU MLP; softmax over chosen k) with DeepSeek-V3 aux-loss-free routing bias. Smoke test: 98.8M total / 17.7M active block params, 26k tok/s vs 36k dense, expert imbalance ~3.9× at bias_rate 1e-3 (too slow; tune). | decided |
| 10 | Other hyperparameters | AdamW (β 0.9/0.95, eps 1e-10, fused), WD 0.8 on matrices only (note: lr·wd timescale ≈ 830 steps), dropout 0.1 on residual branches, grad clip 1.0, 65k tok/step (32×2048), WSD: 100 warmup, flat, 40% linear warmdown to 0, 4 epochs = 400M tokens (~6100 steps). Peak LR 2e-3 from sweep (15M tokens each, full val: 1e-3 → 5.817, **2e-3 → 5.711**, 4e-3 → 5.781). MoE bias rate 1e-2. | decided |

## Fixed by benchmark compatibility
- Eval: mean CE on slowrun's 10M-token `fineweb_val.pt`, seq len 2048.
- Train data budget: 100M tokens (multi-epoch allowed).

## Runs
| Run | Config | Val loss | Val bpb | Notes |
|---|---|---|---|---|
| base4ep | all defaults, lr=2e-3, 400M tokens (4 epochs) | **4.043** | 1.314 | 2026-10-15, 196 min, 35k tok/s, 17% MFU. Train≈val throughout (no overfitting at 4 epochs). Flat-LR phase plateaued ~4.55 by step 3500; warmdown (last 40%) took it 4.55 → 4.04. |
