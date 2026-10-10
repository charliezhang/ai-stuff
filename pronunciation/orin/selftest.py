#!/usr/bin/env python3
"""Sanity-check the scorer without a human speaker, using the local TTS server (:7860) as a "native" voice.

  match       TTS says sentence X, scored against X                  -> should be high (~90+)
  swap-word   TTS says X with one word changed, scored against X     -> that word low, others high
  mismatch    TTS says sentence Y, scored against X (same language)  -> should be low (< 30)

Usage (inside pron:orin, network host):  python3 selftest.py --model /hf/.../snapshots/<rev>
"""
import argparse
import io
import json
import urllib.request

import librosa
import numpy as np
import soundfile as sf

from scorer import PronunciationScorer

VOICE = {"en": ("Aiden", "English"), "zh": ("Vivian", "Chinese"), "es": ("Serena", "Spanish")}
SWAPS = {"en3": ("cat", "bat"), "en4": ("Three", "Tree"), "zh2": ("苹果", "香蕉"), "es2": ("perro", "pero")}


def tts(text, lang, url="http://localhost:7860/v1/audio/speech"):
    voice, language = VOICE[lang]
    req = urllib.request.Request(url, json.dumps({"input": text, "voice": voice, "language": language,
                                                  "response_format": "wav"}).encode(), {"Content-Type": "application/json"})
    a, sr = sf.read(io.BytesIO(urllib.request.urlopen(req).read()), dtype="float32")
    return librosa.resample(a, orig_sr=sr, target_sr=16000)


def show(tag, r):
    words = "  ".join(f"{w['word']}:{w['score']:.0f}" for w in r["words"])
    tone = f" | tone {r['tone']:.0f}" if r.get("tone") is not None else ""
    print(f"  {tag:10} overall {r['overall']:5.1f} | {words}{tone}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--only", default="", help="comma list of sentence ids")
    args = ap.parse_args()
    sc = PronunciationScorer(args.model)
    sents = json.load(open("sentences.json"))
    by_lang = {}
    for s in sents:
        by_lang.setdefault(s["lang"], []).append(s)
    for s in [x for x in sents if not args.only or x["id"] in args.only.split(",")]:
        print(f"{s['id']}  {s['text']}   expected: {' | '.join(' '.join(p) for _, p in sc.g2p(s['text'], s['lang']))}")
        audio = tts(s["text"], s["lang"])
        r = sc.score(audio, s["text"], s["lang"])
        show("match", r)
        print(f"  {'':10} heard: {r['heard']}")
        if s["id"] in SWAPS:
            a, b = SWAPS[s["id"]]
            show(f"say '{b}'", sc.score(tts(s["text"].replace(a, b), s["lang"]), s["text"], s["lang"]))
        other = next(o for o in by_lang[s["lang"]] if o["id"] != s["id"] and len(o["text"]) > 3)
        show("mismatch", sc.score(tts(other["text"], s["lang"]), s["text"], s["lang"]))


if __name__ == "__main__":
    main()
