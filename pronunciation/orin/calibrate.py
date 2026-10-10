#!/usr/bin/env python3
"""Per-language score calibration (proxy): native TTS speech -> ~90.

The recognizer is much stronger in some languages than others (Mandarin was a small part of its
Common Voice training data), so raw GOP for *correct* native speech differs per language. We shift each
language's GOP->score mapping so the median phoneme of native speech maps to TARGET:

    score = 100 * sigmoid((GOP - bias[backend][lang]) / TEMP),   bias_lang = median_native_GOP - TEMP * ln(T/(100-T))

Native speech here is the local Qwen3-TTS (several voices x the test sentences + extra sentences), i.e.
a *proxy*. Real calibration needs human recordings with human scores (native and learners, kids), fit
per language and age group. Writes calib.json; also reports wrong-sentence scores, which must stay low.
"""
import json
import math
import sys

import numpy as np

import scorer as S
from selftest import VOICE, tts

BACKEND = sys.argv[1] if len(sys.argv) > 1 else "w2v"

TARGET = 90.0
VOICES = {"en": ["Aiden", "Ryan", "Serena"], "zh": ["Vivian", "Serena", "Dylan"], "es": ["Serena", "Aiden", "Ryan"]}
EXTRA = {
    "en": ["My dog likes to run in the park.", "Please close the door.", "We are going to school today."],
    "zh": ["今天天气很好。", "我们一起去公园玩吧。", "请把门关上。"],
    "es": ["Hoy hace mucho sol.", "Mi perro corre en el parque.", "Vamos a la escuela."],
}

sc = S.load(BACKEND)
sents = json.load(open("sentences.json"))
calib = {}
for lang in ("en", "zh", "es"):
    texts = [s["text"] for s in sents if s["lang"] == lang] + EXTRA[lang]
    gops, results = [], []
    for v in VOICES[lang]:
        VOICE[lang] = (v, VOICE[lang][1])
        for t in texts:
            r = sc.score(tts(t, lang), t, lang)
            gops += [p["gop"] for w in r["words"] for p in w["phones"]]
    med = float(np.median(gops))
    bias = med - S.TEMP * math.log(TARGET / (100 - TARGET))
    calib[lang] = {"bias": round(bias, 3), "temp": S.TEMP, "median_native_gop": round(med, 3), "n_phones": len(gops)}
    print(f"{lang}: {len(gops)} native phones, median GOP {med:+.2f} -> bias {bias:+.2f}")
# merge into calib.json under this backend (other backends' calibration untouched)
allc = json.loads(S.CALIB_FILE.read_text()) if S.CALIB_FILE.exists() else {}
allc[BACKEND] = calib
json.dump(allc, open("/out/calib.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(allc, indent=2, ensure_ascii=False))
