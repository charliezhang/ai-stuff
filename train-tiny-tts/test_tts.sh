#!/usr/bin/env bash
# Speak a sentence with a checkpoint trained on the Orin. Runs on the Mac (CPU), with pocket-tts' CLI.
#
#   ./test_tts.sh ["text"] [--run orin_small] [--step N] [--voice FILE] [-- extra pocket-tts generate options]
#
# Fetches <data dir>/runs/<run>/checkpoint_<step>.pt (default: the latest) into build/checkpoints/,
# then plays it with ffplay as it is generated and keeps a copy in build/tts/<run>_<step>.wav. --voice is the voice to clone (a wav file, or
# a built-in voice name); default: alba. The model shape comes from orin/configs/model_<run>.yaml.
#
# Env: ORIN_HOST (default: orin)
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD
HOST=${ORIN_HOST:-orin}
TEXT="The quick brown fox jumps over the lazy dog. I hope this sounds like a real voice by now."
RUN=orin_small
STEP=""
VOICE=""
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --run) RUN=$2; shift 2 ;;
    --step) STEP=$(printf '%08d' "$2"); shift 2 ;;
    --voice) VOICE=$2; shift 2 ;;
    --) shift; EXTRA=("$@"); break ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) TEXT=$1; shift ;;
  esac
done
MODEL_CONFIG=$HERE/orin/configs/model_$RUN.yaml
[ -f "$MODEL_CONFIG" ] || { echo "no model config $MODEL_CONFIG (copy one and set the run's shape)" >&2; exit 2; }

DATA=$(ssh "$HOST" 'cat ~/train-tiny-tts/.data_dir')
REMOTE=$DATA/runs/$RUN
if [ -z "$STEP" ]; then
  CKPT=$(ssh "$HOST" "ls $REMOTE/checkpoint_*.pt | tail -1") || { echo "no checkpoints in $HOST:$REMOTE" >&2; exit 1; }
else
  CKPT=$REMOTE/checkpoint_$STEP.pt
fi
NAME=$(basename "$CKPT" .pt)
LOCAL=$HERE/build/checkpoints/$RUN/$NAME.pt
mkdir -p "$(dirname "$LOCAL")" build/tts
[ -f "$LOCAL" ] || { echo "fetching $HOST:$CKPT"; scp -q "$HOST:$CKPT" "$LOCAL.part" && mv "$LOCAL.part" "$LOCAL"; }

OUT=$HERE/build/tts/${RUN}_${NAME#checkpoint_}.wav
# A relative voice path is relative to here; pocket-tts runs from its own directory.
if [ -n "$VOICE" ] && [ -f "$VOICE" ]; then VOICE=$(cd "$(dirname "$VOICE")" && pwd)/$(basename "$VOICE"); fi
command -v ffplay >/dev/null || [ -n "${NO_PLAY:-}" ] || { echo "installing ffmpeg (for ffplay)"; brew install ffmpeg; }
# Stream the wav to stdout: tee keeps a copy, ffplay plays it as it is generated (NO_PLAY=1: file only).
play() { if [ -n "${NO_PLAY:-}" ]; then cat >/dev/null; else ffplay -autoexit -nodisp -loglevel error -i -; fi; }
if [ -z "${NO_PLAY:-}" ]; then
  vol=$(osascript -e 'set s to get volume settings' -e '(output volume of s as text) & " " & (output muted of s as text)' 2>/dev/null || true)
  case "$vol" in "0 "*|*" true") echo "WARNING: the Mac's sound output is muted or at volume 0 (volume, muted: $vol)" >&2 ;; esac
  echo "playing with ffplay..." >&2
fi
(cd pocket-tts && uv run --quiet pocket-tts generate --config "$MODEL_CONFIG" --checkpoint "$LOCAL" \
  --text "$TEXT" ${VOICE:+--voice "$VOICE"} --output-path - ${EXTRA[@]+"${EXTRA[@]}"}) | tee "$OUT.stream" | play
# A streamed wav header carries no length: rewrite it so the saved copy has the real duration.
ffmpeg -y -loglevel error -i "$OUT.stream" -c copy "$OUT" && rm "$OUT.stream"
echo "wrote $OUT"
