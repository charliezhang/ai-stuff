import math

import tiktoken
import torch


class TokenData:
    """Slowrun .pt format: flat uint16 tokens cut into fixed (seq_len + 1) windows."""

    def __init__(self, path: str):
        d = torch.load(path)
        S = d["seq_size"]                                  # 2049 = 2048 inputs + 1 shifted target
        n = d["tokens"].numel() // S
        self.seqs = d["tokens"][: n * S].view(n, S)
        self.seed = d["seq_shuffle_seed"]

    def __len__(self):
        return len(self.seqs)

    def batches(self, batch_size: int, epoch: int | None = None, device="cuda"):
        """Shuffled per epoch (seed + epoch) when epoch is given; in order for eval."""
        order = torch.arange(len(self.seqs))
        if epoch is not None:
            order = torch.randperm(len(self.seqs), generator=torch.Generator().manual_seed(self.seed + epoch))
        for i in range(0, len(order) - batch_size + 1, batch_size):
            s = self.seqs[order[i : i + batch_size]].to(device, torch.int64, non_blocking=True)
            yield s[:, :-1], s[:, 1:]


def token_byte_lengths(device="cuda") -> torch.Tensor:
    """UTF-8 bytes per GPT-2 token, <|endoftext|> = 0 bytes (matches slowrun's evaluate_bpb)."""
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    return torch.tensor([0 if i == eot else len(enc.decode_single_token_bytes(i)) for i in range(enc.n_vocab)],
                        dtype=torch.int64, device=device)


@torch.no_grad()
def evaluate(model, val: TokenData, batch_size: int, token_bytes: torch.Tensor, max_seqs: int | None = None):
    """Returns (mean CE in nats/token, bits per byte) over the val set."""
    model.eval()
    nats = torch.zeros((), device="cuda")
    nats_on_bytes = torch.zeros((), device="cuda")
    n_tok = n_bytes = 0
    for i, (x, y) in enumerate(val.batches(batch_size)):
        if max_seqs and i * batch_size >= max_seqs:
            break
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), y.flatten(), reduction="none")
        b = token_bytes[y.flatten()]
        nats += loss.sum()
        nats_on_bytes += (loss * (b > 0)).sum()
        n_tok += y.numel()
        n_bytes += b.sum()
    model.train()
    return (nats / n_tok).item(), (nats_on_bytes / (math.log(2) * n_bytes)).item()
