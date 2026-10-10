#!/usr/bin/env bash
# Pronunciation assessment demo on :7861 (container "pron"). Uses the TTS server on :7860 for examples.
# Mic needs https or localhost: from the Mac run  ssh -N -L 7861:localhost:7861 orin  → http://localhost:7861
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
SNAP=$(ls -d /mnt/models/hf/hub/models--facebook--wav2vec2-xlsr-53-espeak-cv-ft/snapshots/* | head -1)
[ -n "$(sudo docker images -q pron:orin)" ] || sudo docker build -t pron:orin "$HERE"
sudo docker rm -f pron 2>/dev/null || true
sudo docker run -d --name pron --restart unless-stopped --runtime=nvidia --network host \
  -e HF_HUB_OFFLINE=1 -v /mnt/models/hf:/hf:ro -v "$HERE":/app:ro -w /app \
  pron:orin python3 app.py --model "${SNAP/\/mnt\/models\/hf//hf}" --port 7861 >/dev/null
echo "starting; logs: sudo docker logs -f pron"
