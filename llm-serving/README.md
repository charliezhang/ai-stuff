# llm-serving — Qwen3.6-35B-A3B on vLLM (Jetson AGX Orin)

Local LLM inference on the Orin box, served by **vLLM** with an OpenAI-compatible API, a chat UI
(**Open WebUI**) on the Mac, and LLM dashboards in the box's Grafana. Box setup, monitoring stack and
access are in the [root README](../README.md).

```
 Mac (browser)                                Jetson AGX Orin  (orin.local)
 ┌────────────────────┐   OpenAI API :8000   ┌───────────────────────────────────────┐
 │ Open WebUI  :8080  │ ───────────────────▶ │ vLLM (docker)  Qwen3.6-35B-A3B NVFP4  │
 │ mac/start-chat.sh  │                      │   └─ /metrics ──▶ Prometheus (box)    │
 │ browser ───────────│ ── Grafana :3000 ──▶ │ "LLM serving" dashboards              │
 └────────────────────┘                      └───────────────────────────────────────┘
```

| What | Where |
|---|---|
| Chat UI | `mac/start-chat.sh` → http://localhost:8080 |
| Dashboards | http://orin.local:3000 → folder *LLM serving* → *Orin LLM Overview* (Grafana home) |
| OpenAI-compatible API | `http://orin.local:8000/v1`, model `qwen3.6-35b`, no API key |
| Quick CLI test | `python3 mac/chat.py "your question"` (prints TTFT + tok/s) |

## Layout

```
orin/                          # on the Jetson at ~/ai-stuff/llm-serving/orin
  serve-qwen.sh                # starts vLLM (symlinked as ~/serve-qwen.sh)
  grafana-dashboards/          # mounted into the box Grafana as folder "LLM serving"
    make_overview.py           # generates orin-llm-overview.json
    vllm-overview.json, vllm-performance.json   # upstream vLLM dashboards
mac/
  start-chat.sh                # Open WebUI pointed at the Jetson
  chat.py                      # stdlib-only streaming client with timing
```

## Serving: vLLM + Qwen3.6-35B-A3B

```bash
ssh orin ./serve-qwen.sh          # (re)start; ready in ~5 min — follow with: sudo docker logs -f vllm
```

**Model** — `nvidia/Qwen3.6-35B-A3B-NVFP4` (22 GB, stored on `/mnt/models`): 35B-parameter MoE,
~3B active per token (256 experts, 8 routed + 1 shared), hybrid attention (30 Gated-DeltaNet linear
layers + 10 full-attention layers with 2 KV heads), built-in MTP head. Experts are NVFP4 (W4A16),
attention projections FP8.

**Key flags** (`orin/serve-qwen.sh`)

| Flag | Why |
|---|---|
| `VLLM_DISABLED_KERNELS=…FP8…` | **Required on Orin.** sm_87 has no FP8/FP4 tensor cores, but vLLM 0.30 still selects Cutlass/Torch FP8 GEMMs and crashes (surfaces as torch.compile `auto_functionalized was not removed`, or `cutlass_scaled_mm_sm80_epilogue` in eager mode). Disabling the 5 kernels ahead of Marlin in the priority list forces weight-only Marlin (W8A16), like the FP4 experts. |
| `--reasoning-parser qwen3` | Splits `<think>` into `reasoning_content` in the API stream (chat UI "thinking", CoT monitor). |
| `--enable-auto-tool-choice --tool-call-parser qwen3_xml` | Tool calling. Open WebUI sends `tool_choice: "auto"` with its built-in tools; without these flags every chat fails with *"auto" tool choice requires --enable-auto-tool-choice*. |
| `--limit-mm-per-prompt {"image":0,"video":0}` | Text-only; skips multimodal memory reservations. |
| `--max-model-len 8192 --gpu-memory-utilization 0.7` | Leaves ~18 GB unified memory for the OS, monitoring, other models. |
| `--enable-mfu-metrics` | Exports `vllm:estimated_flops/read_bytes_*` counters (compute + memory panels). |
| *(no `--enforce-eager`)* | torch.compile + CUDA graphs: **53 tok/s** vs **5.4 tok/s** eager. MoE decode on Orin is kernel-launch bound without graphs. |

**Baseline numbers** (MAXN, 8K context): single request TTFT ≈ 0.25 s, decode ≈ **53 tok/s**;
two concurrent requests ≈ 89 tok/s aggregate. During single-stream decode the DRAM is ~63 % busy
(~130 GB/s of ~205 GB/s) while the GPU reads ~99 % "busy" — decode is memory-bandwidth bound.
Startup: weights 60–80 s from USB, torch.compile ~40 s, CUDA-graph capture ~80 s.
Memory: weights 19.5 GiB, CUDA graphs 2.8 GiB, KV cache ≈ 230K tokens.

**KV cache math** — only the 10 full-attention layers grow with context:
`10 layers × 2 (K,V) × 2 KV heads × 256 dims` = 10,240 values/token → 20 KB/token BF16, 10 KB FP8.
The 30 linear-attention layers keep a fixed ~33 MB state per sequence regardless of length.

## Dashboards

Grafana folder *LLM serving* (provisioned from `orin/grafana-dashboards/`):

- **Orin LLM Overview** (ours, Grafana home) — output/input tok/s, TTFT, inter-token latency, KV-cache
  usage, prefix-cache hit rate, running/waiting/preempted requests, model TFLOP/s, DRAM bandwidth
  (EMC hardware vs vLLM's estimate vs peak), GPU/CPU/memory, power per rail, temps, clocks.
  Edit `make_overview.py`, re-run it, sync — Grafana reloads provisioned files automatically.
- **vLLM** / **Performance Statistics** — upstream dashboards from
  [`vllm/examples/observability`](https://github.com/vllm-project/vllm/tree/main/examples/observability).

## Chat: Open WebUI on the Mac

```bash
~/ai-stuff/llm-serving/mac/start-chat.sh      # first run installs deps via uvx (~2-3 GB), then http://localhost:8080
ORIN_URL=http://10.0.0.181:8000/v1 ~/ai-stuff/llm-serving/mac/start-chat.sh   # if orin.local doesn't resolve
```

No login (`WEBUI_AUTH=False`), no RAG/embeddings; chats live in `mac/open-webui-data/` (gitignored).
Uses ~2 GB RAM on the 8 GB Mac — quit it when not chatting. `uv` comes from a pip venv
(`~/.venvs/uv`, linked in `~/.local/bin`) because Homebrew builds it from source on this Mac.
The model's thinking shows as a collapsible block; known upstream quirks with vLLM `reasoning_content`
streaming exist (open-webui issues #24697, #29539) — if thinking renders as plain text while streaming,
it collapses when the answer completes.

## Next steps

- KV cache experiments: FP8 vs BF16 KV (`--kv-cache-dtype`), prefix caching, preemption under load
- Performance: `vllm bench serve` at concurrency 1/4/16, MTP speculative decoding, `VLLM_MARLIN_USE_ATOMIC_ADD=1`
- CoT safety monitor on the hot path (Qwen3Guard-Stream on `reasoning_content`)
- gpt-oss-20b alongside for comparison
