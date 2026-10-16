"""Usage: python -m tiny_llm.train name=<run> [preset=dev] [any ModelConfig/TrainConfig field=value ...]"""
import dataclasses
import json
import os
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import PRESETS, ModelConfig
from .data import TokenData, evaluate, token_byte_lengths
from .model import GPT, MoE

ORIN_BF16_PEAK = 42.5e12


@dataclass
class TrainConfig:
    name: str = "debug"
    preset: str = "dev"
    train_tokens: float = 400e6      # 4 epochs over the 100M-token set
    batch_seqs: int = 32             # 32 x 2048 = 65k tokens/step
    micro_seqs: int = 8              # per forward pass; grad accumulation covers the rest
    lr: float = 1.5e-3               # peak; to be set by the LR sweep
    warmup_steps: int = 100
    warmdown_frac: float = 0.4       # linear decay to 0 over the last 40% (WSD schedule)
    weight_decay: float = 0.8        # decoupled: per-step shrink = lr * wd
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_seqs: int = 512             # ~1M val tokens for periodic evals; full 10M at the end
    compile: bool = True
    seed: int = 1337


def parse_overrides(argv):
    kv = dict(a.split("=", 1) for a in argv)
    tc_fields = {f.name: f.type for f in dataclasses.fields(TrainConfig)}
    mc_fields = {f.name: f.type for f in dataclasses.fields(ModelConfig)}

    def cast(typ, v):
        return v.lower() in ("1", "true") if typ is bool else typ(float(v)) if typ is int else typ(v)

    tc = TrainConfig(**{k: cast(tc_fields[k], v) for k, v in kv.items() if k in tc_fields})
    mc = dataclasses.replace(PRESETS[tc.preset], **{k: cast(mc_fields[k], v) for k, v in kv.items() if k in mc_fields})
    unknown = set(kv) - set(tc_fields) - set(mc_fields)
    assert not unknown, f"unknown overrides: {unknown}"
    return tc, mc


def lr_at(step, total, tc):
    if step < tc.warmup_steps:
        return tc.lr * (step + 1) / tc.warmup_steps
    decay_start = int(total * (1 - tc.warmdown_frac))
    return tc.lr * min(1.0, (total - step) / max(1, total - decay_start))


def main():
    tc, mc = parse_overrides(sys.argv[1:])
    torch.manual_seed(tc.seed)
    out_dir = f"runs/{tc.name}"
    os.makedirs(out_dir, exist_ok=True)
    log = open(f"{out_dir}/log.jsonl", "w")

    def emit(**rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)

    train, val = TokenData("data/fineweb_train.pt"), TokenData("data/fineweb_val.pt")
    token_bytes = token_byte_lengths()
    model = GPT(mc).cuda()
    fwd = torch.compile(model) if tc.compile else model

    # weight decay on matrices only; norms/gains are 1-D and shouldn't be pulled toward 0
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tc.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=tc.lr, betas=(0.9, 0.95), eps=1e-10, fused=True)

    accum = tc.batch_seqs // tc.micro_seqs
    tokens_per_step = tc.batch_seqs * mc.seq_len
    total_steps = int(tc.train_tokens // tokens_per_step)
    emit(event="config", params=mc.param_counts()["total"], steps=total_steps,
         **{f"tc.{k}": v for k, v in dataclasses.asdict(tc).items()},
         **{f"mc.{k}": v for k, v in dataclasses.asdict(mc).items()})

    def micro_batches():
        epoch = 0
        while True:
            yield from train.batches(tc.micro_seqs, epoch=epoch)
            epoch += 1

    it = micro_batches()
    t0, t_last = time.time(), time.time()
    for step in range(total_steps + 1):
        if step % tc.eval_every == 0 or step == total_steps:
            final = step == total_steps
            vl, bpb = evaluate(fwd, val, tc.micro_seqs, token_bytes, max_seqs=None if final else tc.eval_seqs)
            emit(event="final" if final else "eval", step=step, val_loss=vl, val_bpb=bpb,
                 tokens=step * tokens_per_step, elapsed_min=(time.time() - t0) / 60)
            if final:
                break

        lr = lr_at(step, total_steps, tc)
        for g in opt.param_groups:
            g["lr"] = lr
        loss_sum = 0.0
        for _ in range(accum):
            x, y = next(it)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = fwd(x)
            loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten()) / accum
            loss.backward()
            loss_sum += loss.detach()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        moe_imbalance = [m.update_bias() for m in model.modules() if isinstance(m, MoE)]

        if step % 10 == 0:
            torch.cuda.synchronize()
            dt = (time.time() - t_last) / (10 if step else 1)
            t_last = time.time()
            tps = tokens_per_step / dt
            emit(event="train", step=step, loss=loss_sum.item(), lr=lr, gnorm=gnorm.item(),
                 tok_per_s=tps, mfu=tps * mc.train_flops_per_token() / ORIN_BF16_PEAK,
                 mem_gb=torch.cuda.max_memory_allocated() / 1e9,
                 **({"moe_max_over_mean": max(moe_imbalance)} if moe_imbalance else {}))

    torch.save({"model": model.state_dict(), "mc": dataclasses.asdict(mc)}, f"{out_dir}/model.pt")


if __name__ == "__main__":
    main()
