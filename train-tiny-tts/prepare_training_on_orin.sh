#!/usr/bin/env bash
# Prepare the Orin for pocket-tts training. Runs on the Mac; safe to re-run (copies resume, finished
# steps are skipped).
#
#   ./prepare_training_on_orin.sh [--dataset hifitts2_200h] [--orin-data DIR] [--weights FILE] [--subset N]
#
#   1. checks the dataset made by pocket-tts' prepare_data.py (pocket-tts/data/<dataset>)
#   2. fetches the gated kyutai/pocket-tts weights (for the Mimi codec) and the tokenizer
#   3. writes container-path manifests, dropping rows without word alignments
#   4. copies code, weights, manifests and the chapter audio (~16 GB for 200 h) to the Orin: to
#      /mnt/models/train-tiny-tts when the USB drive is mounted, else ~/train-tiny-tts/data
#   5. builds the tiny-tts:orin image there (compiling the sphn wheel the first time)
#   6. runs the training preflight in it
#
# --subset N   only the first N train rows (and N/8 valid rows) and their audio: a quick trial run
#              of the whole pipeline before copying everything.
# Env: ORIN_HOST (default: orin)
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD
HOST=${ORIN_HOST:-orin}
DATASET=hifitts2_200h
ORIN_DATA=""
WEIGHTS=""
SUBSET=0
HEADROOM_GB=8   # latents (~1 GB per 200 h) + checkpoints (2 kept + export: ~3.5 GB for orin_base)

while [ $# -gt 0 ]; do
  case "$1" in
    --dataset) DATASET=$2; shift 2 ;;
    --orin-data) ORIN_DATA=$2; shift 2 ;;
    --weights) WEIGHTS=$2; shift 2 ;;
    --subset) SUBSET=$2; shift 2 ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ "$SUBSET" = 0 ] || HEADROOM_GB=2   # a trial run keeps a few small checkpoints
POCKET=$HERE/pocket-tts
SRC=$POCKET/data/$DATASET
AUDIO=$POCKET/data/downloads/hifitts2_audio
STAGE=$HERE/build/orin-data
step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }
stage() { (cd "$POCKET" && PYTHONPATH="$POCKET" uv run --quiet python "$HERE/mac/stage_orin_data.py" "$@"); }

step "1/6 Check the local dataset"
for f in "$SRC/train_aligned.jsonl" "$SRC/valid_aligned.jsonl" "$AUDIO"; do
  [ -e "$f" ] || die "$f is missing. Run first:  cd pocket-tts && uv run training/scripts/prepare_data.py --hours 200"
done
ssh -o ConnectTimeout=8 "$HOST" true || die "cannot ssh to $HOST"
echo "$DATASET: $(wc -l < "$SRC/train_aligned.jsonl" | tr -d ' ') train rows, $(wc -l < "$SRC/valid_aligned.jsonl" | tr -d ' ') valid rows"

step "2/6 Fetch Mimi weights and tokenizer"
stage weights --out "$STAGE/weights" ${WEIGHTS:+--weights "$WEIGHTS"}

step "3/6 Write manifests for the container"
stage manifests --src "$SRC" --out "$STAGE/$DATASET" --subset "$SUBSET"
LIST=$STAGE/$DATASET/audio_files.txt
NEED_GB=$( (cd "$AUDIO" && tr '\n' '\0' < "$LIST" | xargs -0 du -ck | tail -1 | awk '{print $1}'; du -sk "$STAGE" | awk '{print $1}') \
  | awk -v h="$HEADROOM_GB" '{ s += $1 } END { printf "%.1f", s / 1048576 + h }')
echo "needs ${NEED_GB} GB on $HOST (audio + weights + ${HEADROOM_GB} GB for latents and checkpoints)"

step "4/6 Copy code and data to $HOST"
# ssh joins its arguments into one string the remote shell re-splits: quote each one for it.
ORIN_DATA=$(ssh "$HOST" bash -s -- $(printf '%q ' "$ORIN_DATA" "$NEED_GB") <<'EOF'
set -euo pipefail
dir=$1 need=$2
if [ -z "$dir" ]; then
  if mountpoint -q /mnt/models; then dir=/mnt/models/train-tiny-tts; else dir=$HOME/train-tiny-tts/data; fi
fi
mkdir -p "$dir" 2>/dev/null || sudo install -d -o "$(id -u)" -g "$(id -g)" "$dir"
# Space already used there counts as available: a re-run overwrites it.
have=$(awk -v a="$(df -Pk "$dir" | awk 'NR == 2 { print $4 }')" -v u="$(du -sk "$dir" | awk '{ print $1 }')" \
  'BEGIN { printf "%.1f", (a + u) / 1048576 }')
if awk -v h="$have" -v n="$need" 'BEGIN { exit !(h < n) }'; then
  echo "$dir has ${have} GB, needs ${need} GB." >&2
  mountpoint -q /mnt/models || echo "The USB drive is not mounted at /mnt/models: plug it in and 'sudo mount /mnt/models', or pass --orin-data DIR (or --subset N for a trial)." >&2
  exit 3
fi
mkdir -p ~/train-tiny-tts && echo "$dir" > ~/train-tiny-tts/.data_dir
echo "$dir"
EOF
) || die "not enough space on $HOST (see above)"
echo "data dir on $HOST: $ORIN_DATA"
rsync -az --delete --exclude .git --exclude .venv --exclude '/data/' --exclude '/runs/' \
  --exclude '__pycache__' --exclude '/docs/' --exclude '/.github/' "$POCKET/" "$HOST:train-tiny-tts/pocket-tts/"
rsync -az --delete --exclude '__pycache__' "$HERE/orin/" "$HOST:train-tiny-tts/orin/"
rsync -a --exclude audio_files.txt "$STAGE/" "$HOST:$ORIN_DATA/"
echo "copying $(wc -l < "$LIST" | tr -d ' ') chapter files (resumes if interrupted)..."
rsync -a --files-from="$LIST" "$AUDIO/" "$HOST:$ORIN_DATA/audio/"

step "5/6 Build tiny-tts:orin"
ssh "$HOST" 'bash ~/train-tiny-tts/orin/build_image.sh'

step "6/6 Preflight"
ssh "$HOST" bash -s -- $(printf '%q ' "$ORIN_DATA" "$DATASET") <<'EOF'
set -euo pipefail
docker run --rm --runtime nvidia --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v ~/train-tiny-tts:/work:ro -v "$1":/data tiny-tts:orin \
  python3 /work/orin/train_orin.py /work/orin/configs/orin_small.yaml \
    data.train_jsonl=/data/$2/train_aligned.jsonl data.valid_jsonl=/data/$2/valid_aligned.jsonl --check 2>&1 \
  | grep '^\[orin\]'
EOF
printf '\n\033[32mReady.\033[0m Start training with:  ./start_train_on_orin.sh [orin/configs/orin_small.yaml]\n'
[ "$SUBSET" = 0 ] || echo "(trial data: start with  ./start_train_on_orin.sh orin/configs/orin_small.yaml max_steps=200 data.num_bucket_batches=1 run_dir=/data/runs/trial)"
