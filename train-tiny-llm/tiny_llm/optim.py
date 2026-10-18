import torch


@torch.no_grad()
def newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximately map G = U S V^T to U V^T (all singular values -> ~1), in bf16."""
    a, b, c = 3.4445, -4.7750, 2.0315             # quintic coefficients tuned for fast convergence
    X = G.bfloat16()
    transpose = X.size(0) > X.size(1)
    if transpose:                                 # iterate on the smaller Gram matrix
        X = X.T
    X = X / (X.norm() + 1e-7)                     # spectral norm <= Frobenius norm <= 1
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if transpose else X


class Muon(torch.optim.Optimizer):
    """Momentum + orthogonalized update for 2-D hidden weight matrices (Keller Jordan's Muon)."""

    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                buf = self.state[p].setdefault("momentum", torch.zeros_like(p.grad))
                buf.lerp_(p.grad, 1 - group["momentum"])
                g = p.grad.lerp(buf, group["momentum"])               # Nesterov-style lookahead
                update = newton_schulz(g, group["ns_steps"])
                update *= max(1.0, p.size(0) / p.size(1)) ** 0.5      # keep RMS step similar for tall matrices
                p.mul_(1 - group["lr"] * group["weight_decay"])       # decoupled weight decay
                p.add_(update.to(p.dtype), alpha=-group["lr"])
