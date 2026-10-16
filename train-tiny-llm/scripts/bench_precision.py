"""Step 4: measure matmul throughput per dtype at the dev model's real shapes."""
import sys
import torch

sys.path.insert(0, ".")
from tiny_llm.config import PRESETS

c = PRESETS["dev"]
M = 8 * c.seq_len  # rows per micro-batch: 8 sequences x 2048 tokens
SHAPES = {
    "qkv/o   (d->d)":    (M, c.n_embd, c.n_embd),
    "mlp up  (d->4d)":   (M, c.n_embd, c.mlp_hidden),
    "mlp down(4d->d)":   (M, c.mlp_hidden, c.n_embd),
    "lm_head (d->V)":    (M, c.n_embd, c.padded_vocab),
}


def bench(fn, iters=20):
    for _ in range(3):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters / 1e3


def tflops(dtype, m, k, n, tf32=False):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)
    return 2 * m * k * n / bench(lambda: a @ b) / 1e12


def fp8_tflops(m, k, n):
    a = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn).t()
    one = torch.ones((), device="cuda")
    return 2 * m * k * n / bench(lambda: torch._scaled_mm(a, b, one, one, out_dtype=torch.bfloat16)) / 1e12


def bandwidth_gbs():
    x = torch.randn(64 * 2**20, device="cuda", dtype=torch.bfloat16)  # 128 MB
    return 2 * x.numel() * x.element_size() / bench(lambda: x.mul(1.0001)) / 1e9


print(torch.cuda.get_device_name(0), torch.__version__)
print(f"{'shape':18} {'fp32':>7} {'tf32':>7} {'fp16':>7} {'bf16':>7} {'fp8':>7}   (TFLOPS)")
for name, (m, k, n) in SHAPES.items():
    row = [tflops(torch.float32, m, k, n), tflops(torch.float32, m, k, n, tf32=True),
           tflops(torch.float16, m, k, n), tflops(torch.bfloat16, m, k, n)]
    try:
        row.append(fp8_tflops(m, k, n))
    except Exception as e:
        row.append(float("nan"))
        fp8_err = str(e).splitlines()[0][:100]
    print(f"{name:18} " + " ".join(f"{v:7.1f}" for v in row))
if "fp8_err" in globals():
    print("fp8 unavailable:", fp8_err)
print(f"elementwise bandwidth: {bandwidth_gbs():.0f} GB/s (Orin LPDDR5 peak 204.8)")
