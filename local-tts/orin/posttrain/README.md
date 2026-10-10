# posttrain — checkers, eval, and fine-tuning for Qwen3-TTS on the Orin

The loop: **fixed eval set → generate → automatic checkers → compare runs**, before and after every
training change. Checkers are only trusted as far as they've been calibrated on real human speech.

## Files

| File | What |
|---|---|
| `evalset.jsonl` | 60 held-out sentences: 20 zh, 20 en, 20 mixed (code-switched). No digits. Never trained on. |
| `checkers.py` | ASR (2 systems) → error rate + English-in-mixed recall; WavLM speaker similarity; UTMOS; sanity |
| `run_eval.py` | generate eval set with a model (`custom` / `clone`) or score recordings (`audio`), run checkers |
| `run-eval.sh` | runs `run_eval.py` in the `qwen-tts:orin` container (outputs → `local-tts/runs/<name>/`) |
| `compare.py` | side-by-side table of several runs |
| `prep_ljspeech.py` | deterministic LJSpeech split: 2,000 train clips (3.67 h), 100 test, fixed ref clip |

```bash
# score a model                                   # score human recordings (calibration)
./run-eval.sh --name X --mode custom --model $M --speaker Vivian
./run-eval.sh --name human-ljspeech --mode audio --evalset /datasets/ljs-sft/test.jsonl \
              --sim-refs /datasets/ljs-sft/test.jsonl --kinds en --limit 40
python3 compare.py ../../runs/A ../../runs/B
```

## Checkers

| Metric | How | Notes |
|---|---|---|
| `err%` (CER zh / WER en / MER mixed) | ASR transcript vs input text, Levenshtein S/D/I | tokens: each Chinese char, each English word |
| `alt%` | same with the *other* ASR | both wrong → likely real TTS error; one-sided → ASR quirk |
| `en_rec` | share of English words in mixed sentences transcribed as English | code-switching specific |
| `sim_ref` / `sim_self` | WavLM x-vector cosine vs target-speaker clips / vs run's own centroid | `sim_self` = voice consistency across zh/en/mixed |
| `utmos` | UTMOS22 predicted MOS | catches flat/robotic speech that WER rewards |
| sanity | speaking rate (2–8 syll/s), long silences, clipping | catches truncation and run-on generation |

**ASR choice per language** — Paraformer-zh is primary for zh and mixed, Whisper-large-v3 for en.
Normalization on both sides: NFKC, traditional→simplified (OpenCC), apostrophes dropped, punctuation
removed, and Whisper's EnglishTextNormalizer on English-only text (numbers: "ten" == "10").

## Calibration and what the checkers taught us

| Run | zh err | en err | mixed err | sim | UTMOS |
|---|---|---|---|---|---|
| **human** LJSpeech (40 real clips) | – | **1.6 %** (Whisper) | – | 0.98 | 4.36 |
| Qwen3-TTS 1.7B CustomVoice Vivian (5/kind smoke) | 0.0 % | 0.0 % | **6.4 %** | – | 3.65–4.03 |
| **Baseline** 0.6B-Base zero-shot clone of `ref.wav` (60 utts) | 1.0 % | 0.0 % | **21.3 %** (Whisper 10.9 %) | zh 0.80 / en 0.97 / mixed 0.93 | 4.41 |

Baseline reading: English is already near-perfect and on-voice. **Cloning an American-English reference
gives the Chinese parts of mixed sentences a heavy English accent** — Paraformer hears
"他的 presentation 做得非常 professional" as "tithe presentation do the fate on professional"; both ASRs
fail at the same spots, so it's real, not an ASR quirk. zh `sim_ref` 0.80 vs en 0.97 says the voice also
drifts when speaking Chinese (part of that gap is the speaker-embedding model's own language bias).

- **Whisper alone overstated mixed errors 2.4×** (15.4 % vs 6.4 %): on English-led code-switched
  speech it *translated* the Chinese ("一杯咖啡" → "a cup of coffee"). Paraformer can't translate.
- **Normalization halved the human floor** (3.7 % → 1.6 %): "ten or fifteen" vs "10 or 15".
- **A real finding:** both ASRs hear "这个 bug … pull 一下" as "八股 … 铺/破" — the voice gives
  "bug" a Mandarin syllable shape (bāgǔ). That's a code-switching accent error worth targeting.
- Paraformer is weak on English ("our tame", "dock") — that's why it's only the second opinion there.
- `sim_ref` for humans includes each clip in its own reference set, so 0.98 is slightly inflated.

## Data licensing

Training/eval audio must be ours, public-domain/permissive datasets (LJSpeech is public domain), or
output of permissively licensed models (Qwen3-TTS, Apache-2.0). **Not ElevenLabs output**: their
Prohibited Use Policy §9.k–9.l (17 Aug 2026) bans using Output to train *or test* ML models.

## Training plan (next)

1. **Baseline**: `Qwen3-TTS-12Hz-0.6B-Base` zero-shot cloning from `ref.wav` on the full eval set
   (`--mode clone`), plus `sim_ref` against the LJSpeech test clips.
2. **Data prep**: Qwen's `prepare_data.py` encodes each training clip into 12.5 Hz × 16 audio codes.
3. **SFT**: Qwen's `sft_12hz.py` — full fine-tune of 0.6B, new speaker `lj`, checkpoint per epoch.
   Needs vLLM stopped (~10 GB AdamW state + activations in unified memory).
4. **Eval every checkpoint** with `--mode custom --speaker lj`: expect `sim_ref` ↑ toward 0.98,
   en err ≤ baseline. Watch zh/mixed for **catastrophic forgetting** — LJSpeech is English-only.
