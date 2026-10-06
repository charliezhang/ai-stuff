# local-tts — bilingual (Mandarin ⇄ English) TTS on Jetson AGX Orin

Text-to-speech that mixes Chinese and English **in the same sentence**, served locally on the Orin box.
No post-training yet: this is the baseline the post-training project will build on.
Box setup is in the [root README](../README.md).

```
 Mac                                   Jetson AGX Orin (orin.local)
 ┌───────────────────┐   :7860        ┌─────────────────────────────────────────────┐
 │ browser (UI)      │ ─────────────▶ │ container "tts" (qwen-tts:orin)             │
 │ mac/say.py (API)  │ ─ /v1/audio ─▶ │  app.py: Gradio UI + OpenAI /v1/audio/speech│
 └───────────────────┘   /speech      │  faster-qwen3-tts (CUDA graphs)             │
                                      │  Qwen3-TTS-12Hz-1.7B-CustomVoice  (4.5 GB)  │
                                      └─────────────────────────────────────────────┘
```

| What | Where |
|---|---|
| Web UI (streaming, voices, style, live TTFA/RTF) | http://orin.local:7860 |
| OpenAI-compatible API | `POST http://orin.local:7860/v1/audio/speech` (`wav`, or streamed `pcm` int16 24 kHz) |
| Speak from the Mac | `python3 mac/say.py "这个 bug 我已经 fix 了。" --voice Dylan` |
| (Re)start server | `ssh orin ~/ai-stuff/local-tts/orin/serve-tts.sh [1.7B\|0.6B]` (ready in ~20 s) |
| Speed benchmark | see [Benchmark](#benchmark) |

## Model

[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Alibaba Qwen, Apache-2.0, Jan 2026) — `Qwen3-TTS-12Hz-1.7B-CustomVoice`
(0.6B also downloaded). An LLM-style TTS: a Qwen "talker" LM predicts discrete audio tokens from a
12.5 Hz, 16-codebook speech tokenizer; per frame the talker emits codebook 0 and a small "code predictor"
autoregressively fills the other 15; the tokenizer's decoder turns codes into 24 kHz audio.
Trained on 5M+ hours in 10 languages, with Chinese–English code-switching a stated strength.

- **CustomVoice** = 9 preset speakers (Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee),
  no reference audio needed; 1.7B also takes a free-text **style instruction** ("用开心的语气说").
- **Base** variants do zero-shot cloning from a reference clip and are the starting point for fine-tuning
  (`finetuning/` in the Qwen repo) — next project.
- Weights live on the USB stick: `/mnt/models/hf` (1.7B 4.5 GB, 0.6B 2.5 GB, tokenizer 0.7 GB).

API extras beyond OpenAI's: `language` (default `Auto`) and `instructions` (style). OpenAI voice names
(`alloy`, `nova`, `echo`, `onyx`, …) map to presets so stock OpenAI clients work.

## Serving stack and why

| Layer | Choice | Notes |
|---|---|---|
| Container | `qwen-tts:orin`, built `FROM vllm/vllm-openai:latest` | Reuses the one PyTorch build known to work on Orin (torch 2.13 cu130, sm_87). Packages installed `--no-deps` so pip can't swap in a CPU/x86 torch. |
| Model code | `qwen-tts-hf` (transformers-5 port of `qwen-tts`) | transformers pinned to **5.15.1** — the vLLM image's 5.17 breaks it (`MimiConfig … rope_theta`). |
| Decode engine | [`faster-qwen3-tts`](https://github.com/andimarafioti/faster-qwen3-tts) 0.5.3 | Static KV cache + `torch.cuda.CUDAGraph` for talker and code-predictor steps. |
| Server | `orin/app.py` (FastAPI + Gradio, ~150 lines) | One model, one GPU lock (graphs are single-stream). The upstream demo needs HF OAuth + daily quotas, so we wrote a small one. |

## Benchmark

`orin/bench_tts.py` synthesizes 6 sentences (zh, en, 4 mixed) and reports RTF
(= compute s / audio s; < 1 is faster than real time). MAXN + `jetson_clocks`, vLLM running alongside.

```bash
ssh orin 'M=$(ls -d /mnt/models/hf/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots/* | sed "s|/mnt/models/hf|/root/.cache/huggingface|");
  sudo docker run --rm --runtime=nvidia -e HF_HUB_OFFLINE=1 -v /mnt/models/hf:/root/.cache/huggingface:ro \
  -v ~/ai-stuff/local-tts/orin:/app:ro -v ~/ai-stuff/local-tts/samples:/samples qwen-tts:orin \
  python3 bench_tts.py --engine fast --model $M --speaker Vivian'
```

| Engine | Model | 30 s of audio took | RTF | GPU util | Peak mem |
|---|---|---|---|---|---|
| `qwen-tts` (eager HF `generate`) | 1.7B | 147 s | 4.87 | ~12 %, 1 CPU core 100 % | 4.5 GB |
| `qwen-tts` (eager) | 0.6B | 144 s | 4.71 | ~12 % | 2.4 GB |
| **`faster-qwen3-tts` (CUDA graphs)** | **1.7B** | **25.6 s** | **0.87** | — | 4.7 GB |
| server, streamed `pcm` over LAN | 1.7B | 8.2 s audio in 7.7 s | 0.94 | first audio **0.88 s** | |

**Lesson (same as vLLM eager vs graphs):** each audio frame = 1 talker step + 15 code-predictor steps,
each a full forward pass of hundreds of tiny kernels. In eager mode the Orin's CPU can't launch them fast
enough — 0.6B is no faster than 1.7B, the GPU idles at 12 %. CUDA graphs replay a whole step with one
launch: **5.6× faster**, real time on a 1.7B model.

Samples: `samples/` (gitignored) — `fast-1.7B/Vivian/*.wav`, `api-dylan-mix.wav`.

## Layout

```
orin/                     # on the Jetson at ~/ai-stuff/local-tts/orin (mounted into the container at /app)
  Dockerfile              # qwen-tts:orin
  serve-tts.sh            # start/restart the server (container "tts", :7860, auto-restart)
  app.py                  # Gradio UI + OpenAI-compatible API
  bench_tts.py            # RTF benchmark, --engine hf|fast
mac/
  say.py                  # stdlib client: synthesize + afplay
samples/                  # generated audio (gitignored)
```

## Known limits / next

- One request at a time (GPU lock); a second caller waits. Fine for a demo.
- Not wired into Prometheus/Grafana yet.
- Next: post-training — SFT `Qwen3-TTS-12Hz-0.6B-Base` on a single-speaker dataset (or your voice),
  measure WER (ASR) + speaker similarity before/after, then optionally GRPO/DPO with a WER reward.
  Stop vLLM first to free memory.
