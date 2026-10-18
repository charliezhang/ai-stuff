from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 50257      # GPT-2 tokenizer (slowrun-compatible)
    pad_vocab_to: int = 64       # pad embedding/lm_head rows for tensor-core-friendly shapes
    seq_len: int = 2048
    n_layer: int = 6
    n_embd: int = 384
    n_head: int = 6
    mlp_ratio: float = 4.0       # hidden = mlp_ratio * n_embd (revisited in step 6 for SwiGLU)
    tie_embeddings: bool = False
    # step 5: normalization
    norm: str = "rmsnorm"        # layernorm | rmsnorm | rmsnorm_noaffine
    norm_placement: str = "pre"  # pre | post | peri
    qk_norm: bool = False
    # step 6: activation
    activation: str = "swiglu"   # relu | gelu | silu | relu2 | swiglu | geglu
    # step 7: positions
    pos: str = "rope"            # learned | rope | nope
    rope_base: float = 10000.0
    rope_frac: float = 1.0       # fraction of head dims that rotate (0.5 = slowrun "half-truncated")
    # step 10: regularization
    dropout: float = 0.1         # on each residual branch output (attn + MLP), train only
    # extra ablation knobs
    logit_softcap: float = 0.0   # >0: logits = cap * tanh(logits / cap) (slowrun / modded-nanogpt use 15)
    bf16_residual: bool = False  # keep the residual stream in bf16 (halves its memory traffic)
    # step 9: MoE (0 = dense). Coarse Mixtral-style: each expert is a full-size MLP.
    n_experts: int = 0
    moe_top_k: int = 2
    moe_bias_rate: float = 1e-2  # aux-loss-free balancing: routing-bias step per update (DeepSeek-V3)

    @property
    def padded_vocab(self) -> int:
        return -(-self.vocab_size // self.pad_vocab_to) * self.pad_vocab_to

    @property
    def mlp_hidden(self) -> int:
        return int(self.mlp_ratio * self.n_embd)

    def param_counts(self) -> dict:
        d, L, V = self.n_embd, self.n_layer, self.padded_vocab
        attn = 4 * d * d                       # q, k, v, o (no bias)
        mlp = 2 * d * self.mlp_hidden          # up + down (SwiGLU's 3 matrices are sized to match)
        E = self.n_experts
        mlp_total = E * mlp + d * E if E else mlp           # every expert's weights + router
        mlp_active = self.moe_top_k * mlp + d * E if E else mlp
        blocks = L * (attn + mlp_total)
        emb = V * d
        head = 0 if self.tie_embeddings else V * d
        return dict(blocks=blocks, blocks_active=L * (attn + mlp_active), embedding=emb, lm_head=head,
                    total=blocks + emb + head)

    def train_flops_per_token(self) -> float:
        """fwd+bwd ≈ 6 FLOPs per matmul param per token, plus causal attention scores."""
        p = self.param_counts()
        matmul_params = p["blocks_active"] + self.padded_vocab * self.n_embd  # lm_head matmul always runs
        attn = 6 * self.n_layer * self.seq_len * self.n_embd           # QK^T + AV, causal ≈ half of 12·L·T·d
        return 6 * matmul_params + attn


PRESETS = {
    # rapid ablations: one design question ≈ one short run
    "dev":      ModelConfig(n_layer=6,  n_embd=384,  n_head=6),
    # GPT-2 small shape
    "gpt2s":    ModelConfig(n_layer=12, n_embd=768,  n_head=12),
    # slowrun tiny-track baseline shape (16L x 1024d, SwiGLU hidden 2816)
    "slowrun":  ModelConfig(n_layer=16, n_embd=1024, n_head=8, mlp_ratio=2816 * 1.5 / 1024),
}
