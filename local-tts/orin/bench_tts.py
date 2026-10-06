#!/usr/bin/env python3
"""Synthesize mixed Mandarin/English test sentences with Qwen3-TTS CustomVoice and report speed.

RTF (real-time factor) = synthesis seconds / audio seconds; < 1.0 means faster than real time.

Engines:
  hf    - qwen_tts.Qwen3TTSModel (reference implementation, eager PyTorch)
  fast  - faster_qwen3_tts.FasterQwen3TTS (same weights, decode steps captured in CUDA graphs)

Usage (inside the qwen-tts container):
  python3 bench_tts.py --engine fast --model <local model dir> --speaker Vivian --out /samples
"""
import argparse
import time
from pathlib import Path

import soundfile as sf
import torch

SENTENCES = [
    ("zh", "今天天气很好，我们去公园散步吧。"),
    ("en", "The quick brown fox jumps over the lazy dog."),
    ("mix", "我今天下午有个 meeting，要跟 product team 讨论一下 roadmap。"),
    ("mix", "这个 bug 我已经 fix 了，你可以 pull 一下最新的 code 再试试。"),
    ("mix", "Let's grab 一杯咖啡 before the 下午的会议 starts."),
    ("mix", "我们用 Jetson AGX Orin 在本地跑 LLM inference，速度大概是每秒五十个 tokens。"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["hf", "fast"], default="fast")
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    ap.add_argument("--speaker", default="Vivian")
    ap.add_argument("--language", default="Auto", help="'Auto' lets the model detect per sentence")
    ap.add_argument("--out", default="/samples")
    args = ap.parse_args()

    t0 = time.time()
    if args.engine == "fast":
        from faster_qwen3_tts import FasterQwen3TTS
        model = FasterQwen3TTS.from_pretrained(args.model)
    else:
        from qwen_tts import Qwen3TTSModel
        model = Qwen3TTSModel.from_pretrained(args.model, device_map="cuda:0", dtype=torch.bfloat16,
                                              attn_implementation="sdpa")  # no flash-attn on Jetson
    print(f"[{args.engine}] loaded {args.model} in {time.time() - t0:.1f}s | GPU mem {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    size = "1.7B" if "1.7B" in args.model else "0.6B" if "0.6B" in args.model else Path(args.model).name
    out = Path(args.out) / f"{args.engine}-{size}" / args.speaker
    out.mkdir(parents=True, exist_ok=True)
    # warm-up (for "fast" this also captures the CUDA graphs)
    t = time.time()
    model.generate_custom_voice(text="热身。Warm up.", language=args.language, speaker=args.speaker)
    print(f"warm-up {time.time() - t:.1f}s")

    tot_synth = tot_audio = 0.0
    for i, (kind, text) in enumerate(SENTENCES):
        torch.cuda.synchronize(); t = time.time()
        wavs, sr = model.generate_custom_voice(text=text, language=args.language, speaker=args.speaker)
        torch.cuda.synchronize(); dt = time.time() - t
        dur = len(wavs[0]) / sr
        tot_synth += dt; tot_audio += dur
        path = out / f"{i:02d}_{kind}.wav"
        sf.write(path, wavs[0], sr)
        print(f"[{kind:3}] {dur:5.2f}s audio in {dt:5.2f}s  RTF {dt / dur:4.2f}  {path.name}  {text}")
    print(f"TOTAL {tot_audio:.1f}s audio in {tot_synth:.1f}s  RTF {tot_synth / tot_audio:.2f}"
          f" | peak GPU mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
