import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


# ---------------------------------------------------------------- step 5: normalization
class RMSNormNoAffine(nn.Module):
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),))


def make_norm(kind: str, d: int) -> nn.Module:
    if kind == "layernorm":
        return nn.LayerNorm(d, bias=False)        # subtract mean, divide by std, learnable scale
    if kind == "rmsnorm":
        return nn.RMSNorm(d)                      # divide by RMS only, learnable scale
    if kind == "rmsnorm_noaffine":
        return RMSNormNoAffine()                  # no parameters at all (slowrun / modded-nanogpt)
    raise ValueError(kind)


# ---------------------------------------------------------------- step 7: rotary position embedding
def rope_tables(c: ModelConfig, device=None):
    """cos/sin of shape (T, head_dim/2). Pairs (i, i + head_dim/2) rotate by angle t * inv_freq[i]."""
    head_dim = c.n_embd // c.n_head
    n_pairs = head_dim // 2
    n_rot = int(n_pairs * c.rope_frac)                 # pairs that rotate; the rest get freq 0 (stationary)
    inv_freq = c.rope_base ** (-torch.arange(n_rot, dtype=torch.float32, device=device) / max(n_rot, 1))
    inv_freq = torch.cat([inv_freq, torch.zeros(n_pairs - n_rot, device=device)])
    angles = torch.outer(torch.arange(c.seq_len, dtype=torch.float32, device=device), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """x: (B, T, H, D). Rotate each (x1, x2) pair by its position-dependent angle (half-split layout)."""
    x1, x2 = x.chunk(2, dim=-1)
    cos, sin = cos[: x.size(1), None, :].to(x.dtype), sin[: x.size(1), None, :].to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ---------------------------------------------------------------- attention
class Attention(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.n_head, self.head_dim = c.n_head, c.n_embd // c.n_head
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd, bias=False)
        self.proj = nn.Linear(c.n_embd, c.n_embd, bias=False)
        self.q_norm = make_norm(c.norm, self.head_dim) if c.qk_norm else nn.Identity()
        self.k_norm = make_norm(c.norm, self.head_dim) if c.qk_norm else nn.Identity()

    def forward(self, x, rope=None):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)     # QK-norm bounds attention logits
        if rope is not None:                      # rotate q and k (never v): q·k then depends on (i - j)
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, -1))


# ---------------------------------------------------------------- MLP (step 6 decides activation)
ACTIVATIONS = {
    "relu": F.relu,
    "gelu": lambda x: F.gelu(x, approximate="tanh"),
    "silu": F.silu,
    "relu2": lambda x: F.relu(x).square(),       # Primer / modded-nanogpt
}
GATED = {"swiglu": F.silu, "geglu": lambda x: F.gelu(x, approximate="tanh")}


class MLP(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.gated = c.activation in GATED
        self.act = GATED[c.activation] if self.gated else ACTIVATIONS[c.activation]
        if self.gated:
            # 3 matrices instead of 2: shrink hidden to 2/3 so params & FLOPs match, round to 64
            hidden = 64 * round(2 * c.mlp_hidden / 3 / 64)
            self.gate = nn.Linear(c.n_embd, hidden, bias=False)
        else:
            hidden = c.mlp_hidden
        self.up = nn.Linear(c.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, c.n_embd, bias=False)

    def forward(self, x):
        h = self.act(self.gate(x)) * self.up(x) if self.gated else self.act(self.up(x))
        return self.down(h)


# ---------------------------------------------------------------- step 9: mixture of experts
class MoE(nn.Module):
    """Mixtral-style top-k over full-size experts, balanced with DeepSeek-V3's aux-loss-free bias."""

    def __init__(self, c: ModelConfig):
        super().__init__()
        self.k, self.bias_rate = c.moe_top_k, c.moe_bias_rate
        self.router = nn.Linear(c.n_embd, c.n_experts, bias=False)
        self.experts = nn.ModuleList(MLP(c) for _ in range(c.n_experts))
        # bias steers *which* experts are picked, never the mixing weights, and gets no gradient
        self.register_buffer("route_bias", torch.zeros(c.n_experts))
        self.register_buffer("load", torch.zeros(c.n_experts))      # tokens per expert since last update

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])                                 # (N, d) tokens
        logits = self.router(x).float()                              # (N, E)
        idx = (logits + self.route_bias).topk(self.k, dim=-1).indices  # select with bias...
        w = logits.gather(-1, idx).softmax(-1).to(x.dtype)            # ...weight without it (Mixtral: softmax over chosen k)
        if self.training:
            self.load += torch.bincount(idx.flatten(), minlength=len(self.experts))
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):                    # loop: gather -> small MLP -> scatter-add
            tok, slot = (idx == e).nonzero(as_tuple=True)
            out.index_add_(0, tok, w[tok, slot, None] * expert(x[tok]))
        return out.view(shape)

    @torch.no_grad()
    def update_bias(self):
        """After each optimizer step: overloaded experts become less attractive, underloaded more."""
        self.route_bias += self.bias_rate * torch.sign(self.load.mean() - self.load)
        imbalance = (self.load.max() / self.load.mean().clamp(min=1)).item()   # 1.0 = perfect balance
        self.load.zero_()
        return imbalance


# ---------------------------------------------------------------- block: where norms sit
class Block(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.placement = c.norm_placement
        self.attn, self.mlp = Attention(c), (MoE(c) if c.n_experts else MLP(c))
        self.drop = nn.Dropout(c.dropout)         # regularize what each sublayer writes to the residual stream
        self.norm1, self.norm2 = make_norm(c.norm, c.n_embd), make_norm(c.norm, c.n_embd)
        if self.placement == "peri":              # extra norm on each sublayer's output (Gemma 2/3)
            self.out_norm1, self.out_norm2 = make_norm(c.norm, c.n_embd), make_norm(c.norm, c.n_embd)

    def forward(self, x, rope=None):
        if self.placement == "pre":               # x + f(norm(x)): clean residual highway
            x = x + self.drop(self.attn(self.norm1(x), rope))
            x = x + self.drop(self.mlp(self.norm2(x)))
        elif self.placement == "post":            # norm(x + f(x)): original Transformer / BERT
            x = self.norm1(x + self.drop(self.attn(x, rope)))
            x = self.norm2(x + self.drop(self.mlp(x)))
        elif self.placement == "peri":            # x + norm(f(norm(x)))
            x = x + self.drop(self.out_norm1(self.attn(self.norm1(x), rope)))
            x = x + self.drop(self.out_norm2(self.mlp(self.norm2(x))))
        return x


class GPT(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.c = c
        self.wte = nn.Embedding(c.padded_vocab, c.n_embd)
        self.wpe = nn.Embedding(c.seq_len, c.n_embd) if c.pos == "learned" else None
        if c.pos == "rope":                       # buffers, not params: nothing to learn
            cos, sin = rope_tables(c)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        # post-norm already ends in a norm; pre/peri need one before the head
        self.norm_f = make_norm(c.norm, c.n_embd) if c.norm_placement != "post" else nn.Identity()
        self.lm_head = nn.Linear(c.n_embd, c.padded_vocab, bias=False)
        if c.tie_embeddings:
            self.lm_head.weight = self.wte.weight
        self.apply(self._init)
        for name, p in self.named_parameters():   # GPT-2: shrink residual-branch outputs by depth
            if name.endswith(("proj.weight", "down.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        x = self.wte(idx)
        if self.wpe is not None:
            x = x + self.wpe(torch.arange(idx.size(1), device=idx.device))
        if self.c.bf16_residual:                  # embedding is fp32; bf16 + bf16 branch adds then stay bf16
            x = x.to(torch.bfloat16)
        rope = (self.rope_cos, self.rope_sin) if self.c.pos == "rope" else None
        for block in self.blocks:              # pos == "nope": causal mask is the only position signal
            x = block(x, rope)
        logits = self.lm_head(self.norm_f(x))
        if self.c.logit_softcap:                  # smooth clamp to (-cap, cap): stops overconfident logits
            logits = self.c.logit_softcap * torch.tanh(logits.float() / self.c.logit_softcap)
        return logits[..., : self.c.vocab_size]   # drop padding rows so they never get probability mass
