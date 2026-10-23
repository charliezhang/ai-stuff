# train-tiny-tts

Train a small [pocket-tts](https://github.com/kyutai-labs/pocket-tts) model on the Jetson AGX Orin.
`pocket-tts/` is the upstream repo as a read-only submodule; nothing in it is modified. Everything
Orin-specific lives next to it:

```
prepare_training_on_orin.sh   Mac: weights, manifests, copy code + audio, build image, preflight
start_train_on_orin.sh        Mac: start/resume a run on the Orin and follow its losses
mac/stage_orin_data.py        fetch + check the Mimi weights; container-path manifests
orin/Dockerfile               tiny-tts:orin, FROM the vLLM image (the box's working CUDA torch)
orin/build_image.sh           compiles the sphn wheel once, builds the image
orin/train_orin.py            preflight + key=value overrides around training.train.main
orin/watch_loss.py            loss/ETA follower over progress.jsonl (stdlib, runs on the host)
orin/configs/orin_small.yaml  6 layers x 512 (30M params)
orin/configs/orin_base.yaml   6 layers x 1024 (89M params, the shipped Pocket TTS shape)
orin/configs/model_*.yaml     inference configs with each run's model shape
test_tts.sh                   Mac: speak a sentence with a run's latest (or --step N) checkpoint
serve_tts.sh, serve/          Mac: web UI (text box, playback, waveform, latency metrics)
```

## Before the first run

1. **Dataset**: `cd pocket-tts && uv run training/scripts/prepare_data.py --hours 200`
   (done: `pocket-tts/data/hifitts2_200h`, 16 GB of chapter mp3s).
2. **Gated weights**: training needs Mimi's encoder, which only the gated `kyutai/pocket-tts` repo
   has (the public one ships it zeroed). Accept the terms at
   <https://huggingface.co/kyutai/pocket-tts>, then `cd pocket-tts && uv run hf auth login`.
3. **Storage on the Orin**: about 25 GB (16 GB audio, ~1 GB latents, checkpoints). Mount the USB
   drive at `/mnt/models` (the eMMC has ~3 GB free), or pass `--orin-data DIR`.

## Run

```bash
./prepare_training_on_orin.sh --subset 600     # optional: trial with 600 rows (~1 GB)
./prepare_training_on_orin.sh                  # everything
./start_train_on_orin.sh                       # orin_small; or: ./start_train_on_orin.sh orin/configs/orin_base.yaml
./start_train_on_orin.sh --watch               # follow again after Ctrl-C (training keeps running)
./start_train_on_orin.sh --stop                # checkpoints, then stops; start again to resume
./test_tts.sh "Some text."                     # listen to the latest checkpoint (build/tts/*.wav)
./test_tts.sh --step 50000 --voice me.wav      # a given checkpoint, cloning your own voice
./serve_tts.sh                                 # web UI at http://127.0.0.1:8765 (--step N, --port P)
```

`start_train_on_orin.sh` takes `key=value` overrides for any config field, e.g. `max_steps=50000`
or `run_dir=/data/runs/try2`. The first start encodes the training audio to Mimi latents (~45 min
for 200 h on the Orin) before step 1. Run output is in `<data dir>/runs/<name>/` on the Orin:
`progress.jsonl`, `logs/`, `samples/step*.wav` (every 5k steps), checkpoints, and
`model.safetensors` (pocket-tts format; load it with a copy of `english.yaml` that has the same
`flow_lm` overrides as the training config and `weights_path` pointing at it).

## Measured on the box

Batch 64, `torch.compile` on, JetPack 7.2.1, torch 2.13 + cu130, MAXN + `jetson_clocks`:

| config | params | steps/s | GPU memory | 100k steps |
|---|---|---|---|---|
| `orin_small` (6 x 512) | 30.5M | ~1.5 | 14 GiB training, 20.5 GiB peak with validation + samples | ~19 h |
| `orin_base` (6 x 1024) | 89.4M | ~1.0 | 17.6 GiB training | ~28 h |

Without `torch.compile`, `orin_small` runs ~1.2 steps/s. The reference recipe (24 x 1024 teacher,
400k steps on H100s, then distillation to 6 layers) is out of reach here; these configs train the
student-sized model directly, which pocket-tts notes gives a plainer voice.

## Why it is built this way

- **Container from the vLLM image**: the only aarch64 CUDA torch known to work on this Orin (same
  base as `local-tts` and `train-tiny-llm`). Packages go in with `--no-deps` so torch is never replaced.
- **sphn compiled from source**: pocket-tts reads audio with `sphn`, which has no aarch64 wheel.
  libsndfile/mpg123 is not a substitute: it starts most HiFiTTS-2 mp3s 1105 samples (25 ms) later
  than sphn, which would shift every word timing.
- **Audio stays on the box**: even when training from precomputed latents, every sample re-encodes
  0.5 s of audio at its voice-prompt cut, so the chapter files are needed, not just latents.
- **Rows without word alignments are dropped** (145 train, 6 valid): pocket-tts' loader raises on
  them by design, and one would stop the run.
- **Encoder check before training**: pocket-tts' training path does not reject the zeroed Mimi
  encoder of the public weights; `train_orin.py` does.
