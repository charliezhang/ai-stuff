# pronunciation — local pronunciation assessment demo (EN · ZH · ES)

Scripted pronunciation assessment (like Azure Pronunciation Assessment / SpeechSuper "read-aloud"):
the learner reads a known sentence; every **phoneme, word and the sentence** get a 1–100 score, plus
"sounded like /x/" feedback. Runs on the Jetson Orin; ~80 ms per sentence.

```
 Mac browser ──ssh tunnel──▶ http://localhost:7861   (FastAPI + web/index.html, container "pron")
   mic 16 kHz PCM ──WebSocket /ws/score (250 ms chunks)──▶ re-score every 0.5 s ──▶ live word scores
                               │ text ─eSpeak-NG─▶ canonical IPA phonemes per word
                               │ audio ─wav2vec2-xlsr-53-espeak-cv-ft─▶ CTC phoneme posteriors
                               │ alignment-free GOP: canonical vs substitution/deletion hypotheses
                               └─ 🔊 native example ◀── local TTS (:7860, Qwen3-TTS)
```

**Run it**
```bash
ssh orin ~/ai-stuff/pronunciation/orin/serve-pron.sh       # (re)start both models, ready in ~40 s (MODELS=w2v for one)
ssh -N -L 7861:localhost:7861 orin                         # browsers only allow the mic on https/localhost
open http://localhost:7861                                 # allow microphone access when asked
```
Tap the big round button (or press space), read the sentence; words light up with scores as you reach
them, the button shows a live waveform, and it auto-finishes ~0.9 s after the last word. Opening
`http://orin.local:7861` directly shows "No microphone" — browsers hide mics on non-secure origins.

## Streaming

While you speak, the server re-scores the audio received so far every 0.5 s (35–100 ms per pass):
1. **Progress** — free phoneme decoding is aligned to every prefix of the canonical phonemes (semi-global
   edit distance; tones and accepted variants compare equal); the best prefix = how far you've read.
2. **Prefix scoring** — the same alignment-free GOP, on completed words only. The newest word is
   *provisional* (dashed chip) because its boundary with the next word isn't certain yet — e.g. "The"
   scores 58 at 0.5 s and 94 once "cat" arrives. The final score on stop re-scores the full sentence.

`stream_test.py` replays TTS audio into the WebSocket in real time: words arrive 1/7 → 3/7 → 4/7 → 7/7,
final result ~90 ms after "end".

## Why this design (from the option review)

| Requirement | How this demo addresses it | Status |
|---|---|---|
| Kids 3–8 | No forced alignment (CTC likelihood sums over all alignments — forced alignment is what breaks on kids' pitch/timing); lenient accent variants | ⚠️ recognizer not yet adapted to child speech |
| 1–100, poor → perfect | Per-phoneme log-likelihood ratio → sigmoid, calibrated per language | ⚠️ calibrated on TTS as a *proxy* native; needs human-scored recordings |
| 10+ languages | One multilingual phoneme recognizer + eSpeak-NG G2P (100+ languages) | ✅ EN/ZH/ES now; adding a language = G2P + calibration |

Alternatives considered: GOPT (Kaldi GOP + transformer; English-only adult acoustic model), SSL feature +
hierarchical scorers (3M, HiPAMA, HiPPO; need human-scored training data), end-to-end SSL regression
(no phoneme feedback), speech-LLM LoRA (Phi-4-MM, Qwen2-Audio; good for explanations, weak at phonemes).
This demo is the alignment-free CTC-GOP front-end; a trained hierarchical scorer can sit on top later.

## Scoring

For each canonical phoneme *i*: `GOP_i = log P(canonical | audio) − max over {substitute i with q, delete i}`
using the CTC loss (all alignments). `score = 100·sigmoid((GOP − bias_lang)/2)`. Word = mean of phonemes,
Accuracy = mean of words, Completeness = % words ≥ 30, Overall = 0.8·Accuracy + 0.2·Completeness.
Substitution candidates = the 48 phonemes with most posterior mass in the recording (+ canonical ones).

- **Accepted variants** (`EQUIV` in `scorer.py`) — not errors: American flap (t/d→ɾ), non-rhotic vowels,
  Spanish ð/β/ɣ ↔ d/b/ɡ, Mandarin h written χ vs x.
- **Mandarin tones** are scored separately and **not** in Overall: the recognizer's tone labels are the
  first digit of the Chao contour (tone 1 = 55 and tone 4 = 51 both → `5`), and in self-tests the tone score
  did not separate right from wrong audio. A dedicated tone model is needed.
- **Calibration** (`calibrate.py` → `calib.json`): median native GOP is +3.5 (en), +3.8 (es), only +0.6 (zh)
  — the recognizer is much weaker in Mandarin. Bias per language maps native median → 90.

## Models (UI dropdown)

| | wav2vec2-XLSR-53 eSpeak (`w2v`, default) | POWSM-CTC (`powsm`) |
|---|---|---|
| Source | facebook/wav2vec2-xlsr-53-espeak-cv-ft (2021) | espnet/powsm_ctc (2026, POWSM/PRiSM papers) |
| Size on Orin | 316M params, 1.26 GB GPU | 426M params, 1.70 GB GPU |
| License | Apache-2.0 | CC-BY-4.0 (commercial OK) |
| Training labels | eSpeak IPA (Common Voice) → eSpeak phonemes used 1:1 | IPAPack++ (17k h); eSpeak mapped to its segments (`POWSMBackend.units`) |
| Mandarin tones | weak tone digits → experimental tone score | none emitted → segments only |
| Median GOP, native TTS (en / zh / es) | +3.5 / +0.6 / +3.8 | **+8.6 / +4.0 / +6.1** |
| Scoring pass | ~85 ms | ~350 ms (20 s padded input) |

POWSM integration notes (each was a measured problem):
- **Language prompt** `<eng>/<cmn>/<spa>` is required (with `<unk>` it guessed German).
- **Pad to 20 s** (training condition): single words "Hello"/"Hola" went 13→74 / 7→64.
- **Prompt tokens** `<lang> <pr> <notimestamps> ▁` are emitted in the first 4 frames (160 ms) and are part
  of every CTC target; 0.3 s of leading silence keeps them from crowding out the first word.
- **Infeasible alignments**: 40 ms frames mean a long target can't fit a short clip; CTC must return
  −inf (not `zero_infinity`, and `torch.nan_to_num` must keep −inf) — otherwise wrong audio scored a flat 36.
- ESPnet's BPE model path is relative and loaded lazily — made absolute after init.

Self-test comparison (TTS as speaker, so each run is a *different* random sample — noisy):
POWSM separates **wrong Mandarin** audio much better (7–17 vs 19–48) and is near-perfect on wrong
EN/ES sentences; w2v is currently better at **single-phoneme swaps** in EN/ES (cat→bat, perro→pero) and
scores native EN/ES higher. A real verdict needs a fixed recording set with human scores (speechocean762,
own recordings) — next step.

## Self-test (TTS as the speaker; `selftest.py`)

| | English | Spanish | Mandarin |
|---|---|---|---|
| native, correct sentence | 75–87 | 68–92 | 82–90 |
| one word swapped (that word) | cat→bat: 37–39 | perro→pero: 78 | 苹果→香蕉: 44 |
| wrong sentence | 1–6 | 1–10 | 22–49 |

## Gotchas fixed on the way
- Ubuntu's eSpeak-NG lacks the extended Mandarin dictionary → pinyin spelled out with tone digits read as
  English ("xi3" → `z ɪ θ ɹ iː`). Built from source with `--with-extdict-cmn` (see Dockerfile).
- phonemizer drops multi-token words silently if word/phone separators collide (喜欢 → `i2χ`).
- Single-word prompts ("Hola") vary a lot between TTS voices — little acoustic context.

## Known issues / next
- Native "chair"/"are" still ~50: r-colored vowels need more variant handling.
- Kids: evaluate on real child recordings (with parental consent), then fine-tune the recognizer on child
  speech (CMU Kids / MyST / ChildMandarin), pitch/formant augmentation.
- Replace proxy calibration with human scores (speechocean762 for EN; own rubric for kids, per age band).
- Train a small hierarchical scorer (GOPT-style) on GOP + SSL features; add fluency (pauses, rate).
- Try stronger multilingual phone recognizers (ZIPA, POWSM), especially for Mandarin.

## Files (`orin/`)
`Dockerfile` (pron:orin, eSpeak-NG from source), `scorer.py` (GOP, progress, streaming), `app.py` (FastAPI
:7861: page, /api/words, /api/tts, /ws/score), `web/index.html` (speak button, waveform, live chips),
`sentences.json` (10 test cases), `selftest.py`, `stream_test.py`, `calibrate.py` → `calib.json`, `serve-pron.sh`.
