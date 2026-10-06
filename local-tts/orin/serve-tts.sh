#!/usr/bin/env bash
# Bilingual TTS demo (Qwen3-TTS CustomVoice + faster-qwen3-tts) on :7860 — UI at /, OpenAI API at /v1/audio/speech.
# Usage: ./serve-tts.sh [1.7B|0.6B]     (ready in ~1 min: model load + CUDA-graph capture)
set -euo pipefail
SIZE="${1:-1.7B}"
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
SNAP=$(ls -d /mnt/models/hf/hub/models--Qwen--Qwen3-TTS-12Hz-${SIZE}-CustomVoice/snapshots/* | head -1)
[ -d "$SNAP" ] || { echo "model not found; run: HF_HOME=/mnt/models/hf ~/hfenv/bin/hf download Qwen/Qwen3-TTS-12Hz-${SIZE}-CustomVoice"; exit 1; }
[ -n "$(sudo docker images -q qwen-tts:orin)" ] || sudo docker build -t qwen-tts:orin "$HERE"
sudo docker rm -f tts 2>/dev/null || true
sudo docker run -d --name tts --restart unless-stopped --runtime=nvidia --ipc host -p 7860:7860 \
  -e HF_HUB_OFFLINE=1 -v /mnt/models/hf:/root/.cache/huggingface:ro -v "$HERE":/app:ro \
  qwen-tts:orin python3 app.py --model "${SNAP/\/mnt\/models\/hf//root/.cache/huggingface}" --port 7860 >/dev/null
echo "starting $SIZE; follow with: sudo docker logs -f tts   → http://orin.local:7860"
