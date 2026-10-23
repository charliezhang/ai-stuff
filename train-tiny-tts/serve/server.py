"""Minimal web UI for a pocket-tts checkpoint trained on the Orin (run with pocket-tts' venv).

    python serve/server.py --config orin/configs/model_orin_small.yaml --checkpoint CKPT [--port 8765]

GET /           the page (serve/index.html)
GET /api/info   model and voices
POST /api/tts   {"text": ..., "voice": ...} -> audio/wav, timings in the X-TTS-Metrics header (JSON)
"""

import argparse
import io
import json
import threading
import time
import wave
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from pocket_tts import TTSModel
from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES

HERE = Path(__file__).parent
# English voice recordings any model can clone (the predefined voice *states* only fit released weights).
VOICES = {name: url for name, url in _ORIGINS_OF_PREDEFINED_VOICES.items()
          if url.startswith("hf://kyutai/tts-voices/")}
DEFAULT_VOICE = "alba"

app = FastAPI()
model: TTSModel
info: dict = {}
voice_states: dict = {}
lock = threading.Lock()  # generation is not thread-safe: one request at a time


class TTSRequest(BaseModel):
    text: str
    voice: str = DEFAULT_VOICE


def to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (HERE / "index.html").read_text()


@app.get("/api/info")
def get_info() -> JSONResponse:
    return JSONResponse({**info, "voices": list(VOICES), "default_voice": DEFAULT_VOICE})


@app.post("/api/tts")
def tts(req: TTSRequest) -> Response:
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "empty text")
    if req.voice not in VOICES:
        raise HTTPException(400, f"unknown voice {req.voice!r}")
    with lock:
        t0 = time.perf_counter()
        voice_ms = 0.0
        if req.voice not in voice_states:  # first use of a voice: encode its recording once
            voice_states[req.voice] = model.get_state_for_audio_prompt(VOICES[req.voice])
            voice_ms = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        chunks, first_ms = [], None
        for chunk in model.generate_audio_stream(voice_states[req.voice], text):
            if first_ms is None:
                first_ms = (time.perf_counter() - t1) * 1000
            chunks.append(chunk.detach().float().cpu().numpy())
        gen_ms = (time.perf_counter() - t1) * 1000
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    sr = model.sample_rate
    audio_ms = len(audio) / sr * 1000
    metrics = {
        "voice_encode_ms": round(voice_ms, 1),
        "first_chunk_ms": round(first_ms or 0.0, 1),
        "generate_ms": round(gen_ms, 1),
        "audio_ms": round(audio_ms, 1),
        "rtf": round(audio_ms / gen_ms, 1) if gen_ms else None,
        "server_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    return Response(to_wav(audio, sr), media_type="audio/wav",
                    headers={"X-TTS-Metrics": json.dumps(metrics), "Cache-Control": "no-store"})


def main() -> None:
    global model
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", required=True, help="model config with the run's shape")
    p.add_argument("--checkpoint", required=True, help="training checkpoint (.pt)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args()
    model = TTSModel.load_model(config=a.config, checkpoint=a.checkpoint)
    info.update(model=Path(a.config).stem.removeprefix("model_"), checkpoint=Path(a.checkpoint).stem,
                params_m=round(sum(p.numel() for p in model.flow_lm.parameters()) / 1e6, 1),
                sample_rate=model.sample_rate, device="cpu")
    # Warm up: encode the default voice and run one short generation, so the first request is typical.
    voice_states[DEFAULT_VOICE] = model.get_state_for_audio_prompt(VOICES[DEFAULT_VOICE])
    for _ in model.generate_audio_stream(voice_states[DEFAULT_VOICE], "Warming up."):
        pass
    print(f"serving {info['model']} {info['checkpoint']} at http://{a.host}:{a.port}", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
