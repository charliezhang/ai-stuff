#!/usr/bin/env python3
"""Bilingual TTS demo server: Qwen3-TTS CustomVoice + faster-qwen3-tts (CUDA graphs) on Jetson Orin.

One process, one loaded model:
  /                    Gradio UI — streaming playback, voice/language/style, live TTFA + RTF
  /v1/audio/speech     OpenAI-compatible TTS API (response_format: wav | pcm; pcm streams int16)
  /v1/audio/voices     preset voices
  /health

Usage (inside the qwen-tts container): python3 app.py --model <local model dir> --port 7860
"""
import argparse
import io
import threading
import time

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from faster_qwen3_tts import FasterQwen3TTS

# Preset CustomVoice speakers (name -> description, native language). All of them can speak every
# supported language; native Chinese voices tend to sound most natural on zh-dominant mixed text.
SPEAKERS = {
    "Vivian": "bright young female · Chinese",
    "Serena": "warm gentle female · Chinese",
    "Uncle_Fu": "low mellow male · Chinese",
    "Dylan": "clear Beijing male · Chinese",
    "Eric": "husky Chengdu male · Chinese (Sichuan)",
    "Ryan": "dynamic rhythmic male · English",
    "Aiden": "sunny American male · English",
    "Ono_Anna": "playful female · Japanese",
    "Sohee": "warm female · Korean",
}
# OpenAI voice names -> presets, so stock OpenAI clients work unchanged.
OPENAI_VOICES = {"alloy": "Vivian", "nova": "Serena", "shimmer": "Vivian", "echo": "Aiden",
                 "fable": "Ryan", "onyx": "Uncle_Fu", "ash": "Dylan", "coral": "Serena", "sage": "Eric"}
LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean", "German", "French", "Russian",
             "Portuguese", "Spanish", "Italian"]
EXAMPLES = [
    ["我今天下午有个 meeting，要跟 product team 讨论一下 roadmap。", "Vivian", "Auto", ""],
    ["这个 bug 我已经 fix 了，你可以 pull 一下最新的 code 再试试。", "Dylan", "Auto", ""],
    ["Let's grab 一杯咖啡 before the 下午的会议 starts.", "Aiden", "Auto", ""],
    ["我们用 Jetson AGX Orin 在本地跑 LLM inference，速度大概是每秒五十个 tokens。", "Serena", "Auto", ""],
    ["Honestly, 这个 demo 真的太 cool 了！", "Vivian", "Auto", "用非常兴奋、开心的语气说"],
    ["The quick brown fox jumps over the lazy dog.", "Ryan", "English", "Speak slowly and calmly."],
]

model: FasterQwen3TTS = None
model_name = ""
gpu_lock = threading.Lock()  # CUDA graphs + static KV cache: one generation at a time


def resolve_voice(voice: str) -> str:
    v = OPENAI_VOICES.get(voice.lower(), voice)
    match = {k.lower(): k for k in SPEAKERS}.get(v.lower())
    if not match:
        raise HTTPException(400, f"unknown voice {voice!r}; choose from {list(SPEAKERS)} or {list(OPENAI_VOICES)}")
    return match


def stream_chunks(text, speaker, language, instruct, chunk_size=8):
    """Yield (int16 numpy chunk, sample_rate) while holding the GPU lock."""
    with gpu_lock:
        for audio, sr, _timing in model.generate_custom_voice_streaming(
                text=text, speaker=speaker, language=language, instruct=instruct or None, chunk_size=chunk_size):
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            yield (np.clip(audio, -1, 1) * 32767).astype(np.int16), sr


# ---------------------------------------------------------------- OpenAI-compatible API
class SpeechRequest(BaseModel):
    input: str
    model: str = "qwen3-tts"
    voice: str = "Vivian"
    response_format: str = "wav"      # wav | pcm (pcm = raw int16 mono, streamed)
    language: str = "Auto"            # extension: Qwen3-TTS language hint
    instructions: str | None = None   # OpenAI's style-instruction field (1.7B only)


api = FastAPI(title="local-tts")


@api.get("/health")
def health():
    return {"status": "ok", "model": model_name}


@api.get("/v1/audio/voices")
def voices():
    return {"voices": [{"id": k, "description": v} for k, v in SPEAKERS.items()], "openai_aliases": OPENAI_VOICES}


@api.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    if not req.input.strip():
        raise HTTPException(400, "input is empty")
    speaker = resolve_voice(req.voice)
    if req.response_format == "pcm":
        def gen():
            for chunk, _sr in stream_chunks(req.input, speaker, req.language, req.instructions):
                yield chunk.tobytes()
        return StreamingResponse(gen(), media_type="audio/pcm", headers={"X-Sample-Rate": "24000"})
    if req.response_format != "wav":
        raise HTTPException(400, "response_format must be 'wav' or 'pcm'")
    chunks, sr = [], 24000
    for chunk, sr in stream_chunks(req.input, speaker, req.language, req.instructions):
        chunks.append(chunk)
    buf = io.BytesIO()
    sf.write(buf, np.concatenate(chunks) if chunks else np.zeros(0, np.int16), sr, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav")


# ---------------------------------------------------------------- Gradio UI
def ui_generate(text, speaker, language, instruct):
    if not text.strip():
        raise gr.Error("Type something to say.")
    t0 = time.time()
    ttfa, audio_s, sr = None, 0.0, 24000
    for chunk, sr in stream_chunks(text, speaker, language, instruct):
        ttfa = ttfa or time.time() - t0
        audio_s += len(chunk) / sr
        yield (sr, chunk), f"⏱ first audio {ttfa:.2f}s · {audio_s:.1f}s generated…"
    wall = time.time() - t0
    yield None, (f"⏱ first audio **{ttfa:.2f}s** · {audio_s:.1f}s audio in {wall:.1f}s · "
                 f"**RTF {wall / max(audio_s, 1e-6):.2f}** ({audio_s / wall:.2f}× real time)")


def build_ui():
    with gr.Blocks(title="Orin bilingual TTS") as ui:
        gr.Markdown(f"## 🗣 Bilingual TTS on Jetson AGX Orin\n`{model_name}` · faster-qwen3-tts (CUDA graphs) · "
                    "mix Chinese and English freely in one sentence.")
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(label="Text", lines=4, value=EXAMPLES[0][0])
                with gr.Row():
                    speaker = gr.Dropdown([(f"{k} — {v}", k) for k, v in SPEAKERS.items()], value="Vivian", label="Voice")
                    language = gr.Dropdown(LANGUAGES, value="Auto", label="Language hint")
                instruct = gr.Textbox(label="Style instruction (optional, 1.7B only)",
                                      placeholder="e.g. 用开心的语气说 / Speak slowly and calmly.")
                go = gr.Button("Speak", variant="primary")
            with gr.Column(scale=2):
                audio = gr.Audio(label="Output", streaming=True, autoplay=True)
                stats = gr.Markdown()
        gr.Examples(EXAMPLES, [text, speaker, language, instruct])
        go.click(ui_generate, [text, speaker, language, instruct], [audio, stats])
    return ui


def main():
    global model, model_name
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local Qwen3-TTS CustomVoice model dir")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    model_name = "Qwen3-TTS-1.7B-CustomVoice" if "1.7B" in args.model else "Qwen3-TTS-0.6B-CustomVoice"
    t = time.time()
    model = FasterQwen3TTS.from_pretrained(args.model, local_files_only=True)
    model.generate_custom_voice(text="热身。Warm up.", speaker="Vivian", language="Auto")  # captures CUDA graphs
    print(f"{model_name} ready in {time.time() - t:.0f}s, GPU mem {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)

    app = gr.mount_gradio_app(api, build_ui(), path="/")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
