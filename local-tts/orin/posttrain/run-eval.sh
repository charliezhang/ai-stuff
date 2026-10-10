#!/usr/bin/env bash
# Run run_eval.py in the qwen-tts container.  Usage: ./run-eval.sh --name X --mode custom|clone|audio ...
# Paths inside the container: models /root/.cache/huggingface, datasets /datasets, outputs /runs.
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"          # local-tts/orin
RUNS="$HERE/../runs"; mkdir -p "$RUNS" /mnt/models/torch
sudo docker run --rm --runtime=nvidia --ipc host -e HF_HUB_OFFLINE=1 -e TORCH_HOME=/torch \
  -v /mnt/models/hf:/root/.cache/huggingface:ro -v /mnt/models/datasets:/datasets:ro \
  -v /mnt/models/torch:/torch -v "$HERE":/app:ro -v "$RUNS":/runs -w /app/posttrain \
  qwen-tts:orin python3 run_eval.py "$@" 2>&1 | grep -vE "Warning|warn\(|deprecat|^\s*$"
sudo chown -R "$(id -u):$(id -g)" "$RUNS"
