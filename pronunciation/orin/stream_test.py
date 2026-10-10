#!/usr/bin/env python3
"""Simulate a speaker on the streaming endpoint: send TTS audio in real time, print partials + final.

Usage (inside pron:orin, network host): python3 stream_test.py "The cat is sleeping on the chair." en
"""
import asyncio
import io
import json
import sys
import time
import urllib.parse
import urllib.request

import librosa
import numpy as np
import soundfile as sf
import websockets

text, lang = sys.argv[1], sys.argv[2]
body = json.dumps({"text": text, "lang": lang}).encode()
wav = urllib.request.urlopen(urllib.request.Request("http://localhost:7861/api/tts", body,
                                                    {"Content-Type": "application/json"})).read()
a, sr = sf.read(io.BytesIO(wav), dtype="float32")
a = librosa.resample(a, orig_sr=sr, target_sr=16000)
a = np.concatenate([np.zeros(8000, np.float32), a, np.zeros(16000, np.float32)])   # lead-in + trailing pause
pcm = (np.clip(a, -1, 1) * 32767).astype(np.int16)


async def main():
    url = f"ws://localhost:7861/ws/score?text={urllib.parse.quote(text)}&lang={lang}"
    async with websockets.connect(url) as ws:
        t0 = time.time()

        async def recv():
            async for m in ws:
                r = json.loads(m)
                words = " ".join(f"{w['word']}:{w['score']:.0f}" for w in r.get("words", []))
                print(f"t={time.time() - t0:5.2f}s {r['type']:8} "
                      + (f"done {r.get('done_words')}/{r.get('n_words')} " if r["type"] == "partial" else f"overall {r['overall']} ")
                      + f"[{r.get('elapsed_ms', '-')}ms] {words}", flush=True)
                if r["type"] in ("final", "error"):
                    return

        rx = asyncio.create_task(recv())
        for i in range(0, len(pcm), 4000):                  # 250 ms chunks in real time
            await ws.send(pcm[i:i + 4000].tobytes())
            await asyncio.sleep(0.25)
        await ws.send("end")
        await rx

asyncio.run(main())
