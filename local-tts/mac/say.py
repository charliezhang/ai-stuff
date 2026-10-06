#!/usr/bin/env python3
"""Speak text on this Mac via the Orin TTS server (OpenAI-compatible /v1/audio/speech). Stdlib only.

Usage: python3 say.py "我今天有个 meeting，要讨论 roadmap。" [--voice Vivian] [--save out.wav]
       ORIN_TTS_URL overrides http://orin.local:7860
"""
import argparse, json, os, subprocess, tempfile, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("text")
ap.add_argument("--voice", default="Vivian")
ap.add_argument("--language", default="Auto")
ap.add_argument("--instructions", default=None, help="style instruction (1.7B only)")
ap.add_argument("--save", help="keep the wav at this path")
a = ap.parse_args()

url = os.environ.get("ORIN_TTS_URL", "http://orin.local:7860") + "/v1/audio/speech"
body = {"input": a.text, "voice": a.voice, "language": a.language, "instructions": a.instructions, "response_format": "wav"}
t0 = time.time()
req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
wav = urllib.request.urlopen(req).read()
dt = time.time() - t0
secs = (len(wav) - 44) / (24000 * 2)
print(f"{secs:.1f}s audio in {dt:.1f}s (RTF {dt / secs:.2f})")
path = a.save or tempfile.mktemp(suffix=".wav")
open(path, "wb").write(wav)
subprocess.run(["afplay", path])
