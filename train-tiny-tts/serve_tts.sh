#!/usr/bin/env bash
# Web UI for a checkpoint trained on the Orin: type text, hear it, see the waveform and latencies.
# Runs on the Mac (CPU).
#
#   ./serve_tts.sh [--run orin_small] [--step N] [--port 8765]
#
# Fetches <data dir>/runs/<run>/checkpoint_<step>.pt (default: the latest) into build/checkpoints/
# if it is not there yet, then serves http://127.0.0.1:<port> and opens it. Ctrl-C stops it.
#
# Env: ORIN_HOST (default: orin)
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD
HOST=${ORIN_HOST:-orin}
RUN=orin_small
STEP=""
PORT=8765
while [ $# -gt 0 ]; do
  case "$1" in
    --run) RUN=$2; shift 2 ;;
    --step) STEP=$(printf '%08d' "$2"); shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
MODEL_CONFIG=$HERE/orin/configs/model_$RUN.yaml
[ -f "$MODEL_CONFIG" ] || { echo "no model config $MODEL_CONFIG" >&2; exit 2; }

LOCAL_DIR=$HERE/build/checkpoints/$RUN
if [ -n "$STEP" ] && [ -f "$LOCAL_DIR/checkpoint_$STEP.pt" ]; then
  LOCAL=$LOCAL_DIR/checkpoint_$STEP.pt
elif DATA=$(ssh -o ConnectTimeout=5 "$HOST" 'cat ~/train-tiny-tts/.data_dir' 2>/dev/null); then
  REMOTE=$DATA/runs/$RUN
  CKPT=${STEP:+$REMOTE/checkpoint_$STEP.pt}
  CKPT=${CKPT:-$(ssh "$HOST" "ls $REMOTE/checkpoint_*.pt | tail -1")}
  LOCAL=$LOCAL_DIR/$(basename "$CKPT")
  mkdir -p "$LOCAL_DIR"
  [ -f "$LOCAL" ] || { echo "fetching $HOST:$CKPT"; scp -q "$HOST:$CKPT" "$LOCAL.part" && mv "$LOCAL.part" "$LOCAL"; }
else
  # Orin unreachable: serve the newest checkpoint already on this Mac.
  LOCAL=$(ls "$LOCAL_DIR"/checkpoint_*.pt 2>/dev/null | tail -1) || true
  [ -n "$LOCAL" ] || { echo "cannot reach $HOST and no checkpoint in $LOCAL_DIR" >&2; exit 1; }
  echo "$HOST unreachable; using $LOCAL"
fi

( sleep 1; until curl -sf "http://127.0.0.1:$PORT/api/info" >/dev/null; do sleep 0.5; done
  open "http://127.0.0.1:$PORT" ) &
cd pocket-tts && exec uv run --quiet python "$HERE/serve/server.py" \
  --config "$MODEL_CONFIG" --checkpoint "$LOCAL" --port "$PORT"
