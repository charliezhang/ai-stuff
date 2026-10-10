#!/usr/bin/env python3
"""Generate the eval set with a TTS model (or load existing recordings) and score it with all checkers.

Modes:
  custom  CustomVoice-style model (preset or fine-tuned speaker):  --model DIR --speaker NAME
  clone   Base model, zero-shot voice cloning from one clip:       --model DIR --ref-audio WAV --ref-text TEXT
  audio   no generation; score recordings listed in --evalset ("audio" field). Use for ASR calibration
          on real human speech (the noise floor of the checkers).

Outputs runs/<name>/{wav/*.wav, results.jsonl, summary.json}. Compare runs with compare.py.

Example (inside qwen-tts container, see run-eval.sh):
  python3 posttrain/run_eval.py --name base17-vivian --mode custom --model $M17 --speaker Vivian
"""
import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from checkers import (ASRChecker, SanityChecker, SpeakerSimChecker, Utterance, UTMOSChecker, summarize)

HF = Path("/root/.cache/huggingface/hub")


def snapshot(repo: str) -> str:
    snaps = sorted((HF / f"models--{repo.replace('/', '--')}" / "snapshots").glob("*"))
    if not snaps:
        raise SystemExit(f"{repo} not downloaded (HF_HOME=/mnt/models/hf hf download {repo})")
    return str(snaps[-1])


def load_audio(path: str) -> tuple[np.ndarray, int]:
    a, sr = sf.read(path, dtype="float32", always_2d=True)
    return a.mean(axis=1), sr


def generate(args, items, wav_dir: Path) -> list[Utterance]:
    from faster_qwen3_tts import FasterQwen3TTS
    model = FasterQwen3TTS.from_pretrained(args.model, local_files_only=True)
    utts, t_gen, a_gen = [], 0.0, 0.0
    for it in items:
        torch.manual_seed(zlib.crc32(f"{args.seed}:{it['id']}".encode()))   # same sample per run+utterance
        t = time.time()
        if args.mode == "custom":
            wavs, sr = model.generate_custom_voice(text=it["text"], speaker=args.speaker, language=args.language)
        else:
            wavs, sr = model.generate_voice_clone(text=it["text"], language=args.language,
                                                  ref_audio=args.ref_audio, ref_text=args.ref_text)
        a = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        t_gen += time.time() - t; a_gen += len(a) / sr
        sf.write(wav_dir / f"{it['id']}.wav", a, sr)
        utts.append(Utterance(it["id"], it["kind"], it["text"], a, sr))
        print(f"  gen {it['id']} {len(a) / sr:4.1f}s", flush=True)
    print(f"generated {len(utts)} utts, {a_gen:.0f}s audio, RTF {t_gen / max(a_gen, 1e-6):.2f}")
    del model; torch.cuda.empty_cache()
    return utts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=["custom", "clone", "audio"], required=True)
    ap.add_argument("--model", help="local model / checkpoint dir")
    ap.add_argument("--speaker", default="Vivian")
    ap.add_argument("--ref-audio", help="clone mode: reference clip")
    ap.add_argument("--ref-text", default="", help="clone mode: transcript of the reference clip")
    ap.add_argument("--sim-refs", help="dir or jsonl of reference recordings of the target voice (for sim_ref)")
    ap.add_argument("--evalset", default=str(Path(__file__).with_name("evalset.jsonl")))
    ap.add_argument("--language", default="Auto")
    ap.add_argument("--kinds", default="zh,en,mixed")
    ap.add_argument("--limit", type=int, default=0, help="max utterances per kind (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/runs")
    ap.add_argument("--skip", default="", help="comma list of checkers to skip: asr,speaker,utmos,sanity")
    args = ap.parse_args()

    kinds = args.kinds.split(",")
    items = [json.loads(l) for l in open(args.evalset) if l.strip()]
    items = [it for it in items if it["kind"] in kinds]
    if args.limit:
        items = [it for k in kinds for it in [x for x in items if x["kind"] == k][: args.limit]]

    run = Path(args.out) / args.name
    (run / "wav").mkdir(parents=True, exist_ok=True)
    print(f"run {args.name}: {len(items)} utterances, mode={args.mode}")

    if args.mode == "audio":
        utts = [Utterance(it["id"], it["kind"], it["text"], *load_audio(it["audio"])) for it in items]
    else:
        utts = generate(args, items, run / "wav")

    refs = []
    if args.sim_refs:
        p = Path(args.sim_refs)
        paths = sorted(p.glob("*.wav")) if p.is_dir() else [json.loads(l)["audio"] for l in open(p) if l.strip()]
        import librosa
        refs = [librosa.resample(a, orig_sr=sr, target_sr=16000) for a, sr in map(lambda x: load_audio(str(x)), paths)]

    skip = set(filter(None, args.skip.split(",")))
    checkers = [SanityChecker(),
                ASRChecker(snapshot("openai/whisper-large-v3"), snapshot("funasr/paraformer-zh")),
                SpeakerSimChecker(snapshot("microsoft/wavlm-base-plus-sv"), refs),
                UTMOSChecker()]
    for c in checkers:
        if c.name in skip:
            continue
        t = time.time()
        c(utts)
        print(f"checker {c.name:8} {time.time() - t:5.1f}s", flush=True)

    with open(run / "results.jsonl", "w") as f:
        for u in utts:
            f.write(json.dumps({"id": u.id, "kind": u.kind, "text": u.text, **u.scores}, ensure_ascii=False) + "\n")
    summary = {"name": args.name, "mode": args.mode, "model": args.model, "speaker": args.speaker,
               "ref_audio": args.ref_audio, "seed": args.seed, "metrics": summarize(utts)}
    (run / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n{'kind':6} {'n':>3} {'err%':>6} {'alt%':>5} {'S/D/I':>10} {'en_rec':>6} {'sim_ref':>7} {'sim_self':>8} {'utmos':>5} {'rate':>5} {'sanity':>6}")
    for kind, s in summary["metrics"].items():
        sdi = f"{s.get('sub', '-')}/{s.get('dele', '-')}/{s.get('ins', '-')}"
        fmt = lambda k, p=3: f"{s[k]:.{p}f}" if k in s else "-"
        err = f"{100 * s['err_rate']:.1f}" if "err_rate" in s else "-"
        alt = f"{100 * s['err_rate_alt']:.1f}" if "err_rate_alt" in s else "-"
        print(f"{kind:6} {s['n']:>3} {err:>6} {alt:>5} {sdi:>10} {fmt('en_recall', 2):>6} {fmt('sim_ref'):>7} "
              f"{fmt('sim_self'):>8} {fmt('utmos', 2):>5} {fmt('rate', 1):>5} {s['sanity_fail']:>6}")
    print(f"\nwrote {run}/results.jsonl, summary.json")


if __name__ == "__main__":
    main()
