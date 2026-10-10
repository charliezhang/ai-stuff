#!/usr/bin/env python3
"""Deterministic LJSpeech split for SFT + evaluation.

  test.jsonl        100 held-out clips (never trained on): ASR calibration on real human speech
                    and the speaker-similarity reference (sim_ref) for the target voice
  ref.wav           one fixed reference clip (Qwen3-TTS SFT conditions every sample on it; also the
                    zero-shot cloning prompt for the baseline)
  train_raw.jsonl   N training clips in Qwen3-TTS finetuning format {audio, text, ref_audio}

Uses LJSpeech's *normalized* transcript column (numbers/abbreviations already spelled out).
Usage: python3 prep_ljspeech.py --root /datasets/LJSpeech-1.1 --out /datasets/ljs-sft --train 2000
"""
import argparse
import json
import random
from pathlib import Path

import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/datasets/LJSpeech-1.1")
ap.add_argument("--out", default="/datasets/ljs-sft")
ap.add_argument("--train", type=int, default=2000, help="number of training clips (LJSpeech has 13,100)")
ap.add_argument("--test", type=int, default=100)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

root, out = Path(args.root), Path(args.out)
out.mkdir(parents=True, exist_ok=True)
rows = []
for line in open(root / "metadata.csv", encoding="utf-8"):
    uid, _raw, norm = line.rstrip("\n").split("|")
    rows.append({"id": uid, "text": norm, "audio": str(root / "wavs" / f"{uid}.wav")})

rng = random.Random(args.seed)
rng.shuffle(rows)
test, rest = rows[: args.test], rows[args.test:]

# Reference clip: a medium-length held-out utterance (3-8 s), the same one for every training sample.
ref = next(r for r in test if 3.0 <= sf.info(r["audio"]).duration <= 8.0)
ref_wav = out / "ref.wav"
a, sr = sf.read(ref["audio"])
sf.write(ref_wav, a, sr)
(out / "ref.txt").write_text(ref["text"])

train = rest[: args.train]
hours = sum(sf.info(r["audio"]).duration for r in train) / 3600
with open(out / "train_raw.jsonl", "w") as f:
    for r in train:
        f.write(json.dumps({"audio": r["audio"], "text": r["text"], "ref_audio": str(ref_wav)}) + "\n")
with open(out / "test.jsonl", "w") as f:
    for r in test:
        f.write(json.dumps({"id": r["id"], "kind": "en", "text": r["text"], "audio": r["audio"]}) + "\n")
print(f"train {len(train)} clips ({hours:.2f} h) | test {len(test)} | ref {ref['id']}: {ref['text']!r}")
