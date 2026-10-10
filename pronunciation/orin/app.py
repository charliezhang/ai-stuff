#!/usr/bin/env python3
"""Pronunciation coach (EN / ZH / ES) with streaming feedback — FastAPI on :7861.

  GET  /                 single-page UI (web/index.html): one big speak button + live waveform
  GET  /api/sentences    practice sentences
  GET  /api/models       available recognizer backends (UI model selector)
  POST /api/tts          native example via the local TTS server (:7860)
  WS   /ws/score         ?text=&lang=&model=  stream 16 kHz int16 PCM; re-scores the audio so far every ~0.5 s
                         -> {"type":"partial", ...}  words scored as the learner reaches them
                         <- text "end" -> {"type":"final", ...} full-sentence result

Browsers only allow the microphone on https or localhost, so open it through an SSH tunnel:
    ssh -N -L 7861:localhost:7861 orin      then  http://localhost:7861
"""
import argparse
import asyncio
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

import scorer as S

HERE = Path(__file__).parent
SENTENCES = json.loads((HERE / "sentences.json").read_text())
TTS = {"en": ("Aiden", "English"), "zh": ("Vivian", "Chinese"), "es": ("Serena", "Spanish")}
TTS_URL = "http://localhost:7860/v1/audio/speech"
PARTIAL_EVERY_S = 0.5           # re-score cadence while speaking
MAX_SECONDS = 30

app = FastAPI(title="pronunciation-coach")
SCORERS: dict[str, S.PronunciationScorer] = {}      # both backends loaded at startup (~3 GB GPU total)
DEFAULT_MODEL = "w2v"


def get_scorer(model: str | None) -> S.PronunciationScorer:
    sc = SCORERS.get(model or DEFAULT_MODEL)
    if sc is None:
        raise HTTPException(400, f"model must be one of {list(SCORERS)}")
    return sc
gpu = asyncio.Lock()            # one GPU job at a time; scoring runs in a worker thread


@app.get("/")
def index():
    return FileResponse(HERE / "web" / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/sentences")
def sentences():
    return SENTENCES


@app.get("/api/models")
def models():
    return {"default": DEFAULT_MODEL,
            "models": [{"id": k, "label": sc.b.label, "tonal": sc.b.tonal} for k, sc in SCORERS.items()]}


@app.get("/api/words")
def words(text: str, lang: str, model: str | None = None):
    """Canonical words + phonemes (in the selected model's phone inventory) for the word chips."""
    if lang not in ("en", "zh", "es"):
        raise HTTPException(400, "lang must be en, zh or es")
    return [{"word": w, "phones": ph} for w, ph in get_scorer(model).g2p(text, lang)]


class TTSReq(BaseModel):
    text: str
    lang: str


@app.post("/api/tts")
def tts(req: TTSReq):
    voice, language = TTS.get(req.lang, TTS["en"])
    body = json.dumps({"input": req.text, "voice": voice, "language": language, "response_format": "wav"}).encode()
    try:
        wav = urllib.request.urlopen(urllib.request.Request(TTS_URL, body, {"Content-Type": "application/json"}),
                                     timeout=60).read()
    except Exception as e:
        raise HTTPException(503, f"TTS server (:7860) not reachable: {e}")
    return Response(wav, media_type="audio/wav")


async def run_scoring(fn, *args):
    async with gpu:
        t = time.time()
        r = await asyncio.to_thread(fn, *args)
        r["elapsed_ms"] = int((time.time() - t) * 1000)
        return r


@app.websocket("/ws/score")
async def ws_score(ws: WebSocket):
    await ws.accept()
    text, lang = ws.query_params.get("text", ""), ws.query_params.get("lang", "en")
    scorer = SCORERS.get(ws.query_params.get("model") or DEFAULT_MODEL)
    if not text.strip() or lang not in ("en", "zh", "es") or scorer is None:
        await ws.close(code=1008)
        return
    chunks, last_partial, busy = [], 0.0, False

    async def partial():
        nonlocal busy
        audio = np.concatenate(chunks).astype(np.float32) / 32768.0
        try:
            r = await run_scoring(scorer.score_stream, audio, text, lang)
            await ws.send_json({"type": "partial", **r})
        except Exception as e:                      # client may have gone away mid-score
            print("partial error:", e, flush=True)
        finally:
            busy = False

    try:
        while True:
            msg = await ws.receive()
            if msg.get("bytes"):
                chunks.append(np.frombuffer(msg["bytes"], dtype=np.int16))
                secs = sum(len(c) for c in chunks) / 16000
                if secs > MAX_SECONDS:
                    await ws.send_json({"type": "error", "message": f"Recording longer than {MAX_SECONDS}s"})
                    break
                # Re-score the audio so far, but never queue more than one partial job.
                if secs >= 0.6 and not busy and time.time() - last_partial >= PARTIAL_EVERY_S:
                    busy, last_partial = True, time.time()
                    asyncio.create_task(partial())
            elif msg.get("text") == "end" or msg.get("type") == "websocket.disconnect":
                break
        if chunks and ws.client_state.name == "CONNECTED":
            audio = np.concatenate(chunks).astype(np.float32) / 32768.0
            if len(audio) < 0.3 * 16000 or np.abs(audio).max() < 0.01:
                await ws.send_json({"type": "error", "message": "Recording too short or silent — check your microphone."})
            else:
                r = await run_scoring(scorer.score, audio, text, lang)
                await ws.send_json({"type": "final", **r})
        await ws.close()
    except WebSocketDisconnect:
        pass


def main():
    global DEFAULT_MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="w2v,powsm", help="backends to load (scorer.BACKENDS)")
    ap.add_argument("--port", type=int, default=7861)
    args = ap.parse_args()
    for name in args.models.split(","):
        t = time.time()
        sc = SCORERS[name] = S.load(name)
        sc.score(np.zeros(16000, np.float32), "hello", "en")                # warm-up (CUDA, eSpeak)
        sc.g2p("你好", "zh"); sc.g2p("hola", "es")
        print(f"loaded {name}: {sc.b.label} in {time.time() - t:.0f}s", flush=True)
    DEFAULT_MODEL = args.models.split(",")[0]
    print("pronunciation demo ready", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
