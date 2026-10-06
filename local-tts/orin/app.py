#!/usr/bin/env python3
"""Bilingual TTS demo server: Qwen3-TTS CustomVoice + faster-qwen3-tts (CUDA graphs) on Jetson Orin.

One process, one loaded model:
  /                    Gradio UI — streaming playback, voice/language/style, live TTFA + RTF
  /v1/audio/speech     OpenAI-compatible TTS API (response_format: wav | pcm; pcm streams int16)
  /v1/audio/voices     preset voices
  /metrics/            Prometheus metrics (tts_*; trailing slash — Gradio owns /metrics)
  /health

Usage (inside the qwen-tts container): python3 app.py --model <local model dir> --port 7860
"""
import argparse
import io
import re
import threading
import time

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from prometheus_client import Counter, Gauge, Histogram, Info, make_asgi_app
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


# ---------------------------------------------------------------- Prometheus metrics
# Analogues of vLLM's serving metrics: audio seconds play the role of tokens, TTFA of TTFT, and RTF
# (compute s / audio s) of inter-token latency. Codec frames/tokens per second are derived in PromQL
# from audio seconds (12.5 frames/s × 16 codebooks).
LAT_BUCKETS = (0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10)
RTF_BUCKETS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2, 3, 5)
LBL = ["endpoint", "voice", "text_kind"]
M_REQS = Counter("tts_requests_total", "Finished TTS requests", LBL + ["status"])
# Labeled series only appear on their first increment, and rate() can't see that first step — so
# ratios/throughput use the unlabeled counters (exist from startup); labeled ones are for breakdowns.
M_AUDIO = Counter("tts_audio_seconds_total", "Seconds of audio generated, by label", LBL)
M_AUDIO_ALL = Counter("tts_generated_audio_seconds_total", "Seconds of audio generated (all requests)")
M_COMPUTE = Counter("tts_compute_seconds_total", "Wall seconds spent generating (GPU lock held)")
M_CHARS = Counter("tts_input_chars_total", "Input characters synthesized", ["text_kind"])
M_STALL_S = Counter("tts_stream_stall_seconds_total", "Seconds a real-time listener would have waited mid-stream")
M_STALL_REQ = Counter("tts_stream_stalled_requests_total", "Requests whose stream fell behind real-time playback")
M_TTFA = Histogram("tts_time_to_first_audio_seconds", "Request start (lock acquired) to first audio chunk", buckets=LAT_BUCKETS)
M_RTF = Histogram("tts_rtf", "Real-time factor per request (compute s / audio s; <1 = faster than real time)", buckets=RTF_BUCKETS)
M_E2E = Histogram("tts_request_duration_seconds", "Request duration incl. queueing", buckets=LAT_BUCKETS + (20, 30, 60))
M_QWAIT = Histogram("tts_queue_wait_seconds", "Time waiting for the GPU lock", buckets=(0.01,) + LAT_BUCKETS + (20, 30))
M_AUDIO_LEN = Histogram("tts_request_audio_seconds", "Audio length per request", buckets=(1, 2, 3, 5, 8, 12, 20, 30, 60))
M_RUNNING = Gauge("tts_requests_running", "Requests currently generating (0/1: single GPU lock)")
M_WAITING = Gauge("tts_requests_waiting", "Requests queued for the GPU lock")
M_GPU_MEM = Gauge("tts_gpu_memory_bytes", "PyTorch CUDA memory of the TTS process", ["kind"])
M_GPU_MEM.labels("allocated").set_function(lambda: torch.cuda.memory_allocated())
M_GPU_MEM.labels("reserved").set_function(lambda: torch.cuda.memory_reserved())
M_INFO = Info("tts_model", "Loaded TTS model")

CJK = re.compile(r"[一-鿿]")
LATIN_WORD = re.compile(r"[A-Za-z]{2,}")


def text_kind(text: str) -> str:
    """zh / en / mixed — the code-switching dimension of the dashboard."""
    zh, en = bool(CJK.search(text)), bool(LATIN_WORD.search(text))
    return "mixed" if zh and en else "zh" if zh else "en" if en else "other"


def stream_chunks(text, speaker, language, instruct, endpoint="api", chunk_size=8):
    """Yield (int16 numpy chunk, sample_rate) while holding the GPU lock, recording metrics."""
    lbl = dict(endpoint=endpoint, voice=speaker, text_kind=text_kind(text))
    t_req = time.time()
    M_WAITING.inc()
    with gpu_lock:
        M_WAITING.dec()
        M_QWAIT.observe(time.time() - t_req)
        M_RUNNING.set(1)
        t0, t_first, audio_s, stall, status = time.time(), None, 0.0, 0.0, "ok"
        try:
            for audio, sr, _timing in model.generate_custom_voice_streaming(
                    text=text, speaker=speaker, language=language, instruct=instruct or None, chunk_size=chunk_size):
                now = time.time()
                if t_first is None:
                    t_first = now
                    M_TTFA.observe(now - t0)
                else:
                    # A listener who started playing at the first chunk has heard `elapsed` seconds;
                    # if that exceeds what's been generated, playback would have paused.
                    gap = (now - t_first) - audio_s - stall
                    if gap > 0:
                        stall += gap
                audio = np.asarray(audio, dtype=np.float32).reshape(-1)
                dur = len(audio) / sr
                audio_s += dur
                M_AUDIO.labels(**lbl).inc(dur)
                M_AUDIO_ALL.inc(dur)
                yield (np.clip(audio, -1, 1) * 32767).astype(np.int16), sr
        except GeneratorExit:
            status = "cancelled"
            raise
        except Exception:
            status = "error"
            raise
        finally:
            wall = time.time() - t0
            M_RUNNING.set(0)
            M_COMPUTE.inc(wall)
            M_E2E.observe(time.time() - t_req)
            M_REQS.labels(**lbl, status=status).inc()
            M_CHARS.labels(lbl["text_kind"]).inc(len(text))
            if audio_s > 0:
                M_RTF.observe(wall / audio_s)
                M_AUDIO_LEN.observe(audio_s)
            if stall > 0.05:
                M_STALL_S.inc(stall)
                M_STALL_REQ.inc()


# ---------------------------------------------------------------- OpenAI-compatible API
class SpeechRequest(BaseModel):
    input: str
    model: str = "qwen3-tts"
    voice: str = "Vivian"
    response_format: str = "wav"      # wav | pcm (pcm = raw int16 mono, streamed)
    language: str = "Auto"            # extension: Qwen3-TTS language hint
    instructions: str | None = None   # OpenAI's style-instruction field (1.7B only)


api = FastAPI(title="local-tts")
api.mount("/metrics", make_asgi_app())  # before Gradio claims "/"


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
            for chunk, _sr in stream_chunks(req.input, speaker, req.language, req.instructions, "api-pcm"):
                yield chunk.tobytes()
        return StreamingResponse(gen(), media_type="audio/pcm", headers={"X-Sample-Rate": "24000"})
    if req.response_format != "wav":
        raise HTTPException(400, "response_format must be 'wav' or 'pcm'")
    chunks, sr = [], 24000
    for chunk, sr in stream_chunks(req.input, speaker, req.language, req.instructions, "api-wav"):
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
    for chunk, sr in stream_chunks(text, speaker, language, instruct, "ui"):
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
    M_INFO.info({"model": model_name, "engine": "faster-qwen3-tts", "codec_hz": "12.5", "codebooks": "16"})
    t = time.time()
    model = FasterQwen3TTS.from_pretrained(args.model, local_files_only=True)
    model.generate_custom_voice(text="热身。Warm up.", speaker="Vivian", language="Auto")  # captures CUDA graphs
    print(f"{model_name} ready in {time.time() - t:.0f}s, GPU mem {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)

    app = gr.mount_gradio_app(api, build_ui(), path="/")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
