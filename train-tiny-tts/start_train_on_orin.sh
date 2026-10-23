#!/usr/bin/env bash
# Start (or resume) pocket-tts training on the Orin and follow the losses. Runs on the Mac.
#
#   ./start_train_on_orin.sh [config] [key=value ...]   start or resume, then follow the losses
#   ./start_train_on_orin.sh --watch [config]            follow a running or finished run
#   ./start_train_on_orin.sh --stop                      stop; training checkpoints before exiting
#   ./start_train_on_orin.sh --logs                      raw container output
#
# config: orin/configs/orin_small.yaml (default; 6x512, ~1.5 it/s) or orin/configs/orin_base.yaml
# (6x1024, the shipped Pocket TTS shape, ~1.0 it/s). key=value overrides any config field, e.g.
# max_steps=50000 or run_dir=/data/runs/try2 (see orin/train_orin.py).
# Ctrl-C only stops watching. Re-running with the same config resumes from the latest checkpoint.
#
# Env: ORIN_HOST (default: orin)
set -euo pipefail
cd "$(dirname "$0")"
HOST=${ORIN_HOST:-orin}
NAME=tiny-tts-train
MODE=start
CONFIG=orin/configs/orin_small.yaml
ARGS=()
# Mode flags may come anywhere; the first bare argument is the config, the rest go to training.
for a in "$@"; do
  case "$a" in
    --watch) MODE=watch ;;
    --stop) MODE=stop ;;
    --logs) MODE=logs ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *=*|--check|--allow-dummy-encoder) ARGS+=("$a") ;;
    -*) echo "unknown option: $a (see --help)" >&2; exit 2 ;;
    *) CONFIG=$a ;;
  esac
done
set -- ${ARGS[@]+"${ARGS[@]}"}
[ -f "$CONFIG" ] || { echo "no such config: $CONFIG" >&2; exit 2; }
die() { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

case $MODE in
  stop)
    # SIGTERM: training finishes the step, writes a checkpoint, then exits (allow it 2 minutes).
    ssh "$HOST" "docker stop -t 120 $NAME >/dev/null && echo stopped $NAME"; exit 0 ;;
  logs)
    exec ssh -t "$HOST" "docker logs -f --tail 200 $NAME" ;;
esac

DATA=$(ssh "$HOST" 'cat ~/train-tiny-tts/.data_dir 2>/dev/null') || true
[ -n "$DATA" ] || die "the Orin is not prepared yet: run ./prepare_training_on_orin.sh"

# run_dir as the container sees it (/data/...), with any run_dir= override, then as the host sees it.
RUN_DIR=$(sed -n 's/^run_dir:[[:space:]]*//p' "$CONFIG" | tr -d '"'"'")
for kv in "$@"; do [[ $kv == run_dir=* ]] && RUN_DIR=${kv#run_dir=}; done
[[ $RUN_DIR == /data/* ]] || die "run_dir must live under /data (got '$RUN_DIR')"
HOST_RUN=$DATA/${RUN_DIR#/data/}
watch() { exec ssh -t "$HOST" "python3 ~/train-tiny-tts/orin/watch_loss.py '$HOST_RUN' --container $NAME --follow"; }

if [ "$MODE" = watch ]; then watch; fi

if [ "$(ssh "$HOST" "docker inspect -f '{{.State.Running}}' $NAME 2>/dev/null")" = true ]; then
  die "$NAME is already running. Follow it with --watch, or stop it with --stop."
fi

# Configs and code may have changed since prepare: sync them (small).
rsync -az --delete --exclude '__pycache__' orin/ "$HOST:train-tiny-tts/orin/"
rsync -az --delete --exclude .git --exclude .venv --exclude '/data/' --exclude '/runs/' \
  --exclude '__pycache__' --exclude '/docs/' --exclude '/.github/' pocket-tts/ "$HOST:train-tiny-tts/pocket-tts/"

# ssh joins its arguments into one string the remote shell re-splits: quote each one for it.
ssh "$HOST" bash -s -- $(printf '%q ' "$DATA" "$CONFIG" "$NAME" "$HOST_RUN" "$@") <<'EOF'
set -euo pipefail
data=$1 config=$2 name=$3 host_run=$4
shift 4
sudo -n jetson_clocks 2>/dev/null || echo "note: could not run jetson_clocks (clocks not pinned)"
others=$(docker ps --format '{{.Names}}' | grep -vE "^($name|monitoring-.*)$" | tr '\n' ' ')
avail=$(free -g | awk '/^Mem:/ { print $7 }')
echo "memory available: ${avail} GB (shared CPU/GPU)${others:+ | also running: $others}"
[ "$avail" -ge 28 ] || echo "WARNING: under 28 GB free; training peaks at ~21 GB (6x512) and more for 6x1024. Free memory with: docker stop $others"
if [ -d "$host_run" ] && ls "$host_run"/checkpoint_*.pt >/dev/null 2>&1; then
  echo "resuming from $(ls "$host_run"/checkpoint_*.pt | tail -1 | xargs basename)"
fi
docker rm -f "$name" >/dev/null 2>&1 || true
docker run -d --name "$name" --runtime nvidia --ipc host --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v ~/train-tiny-tts:/work:ro -v "$data":/data tiny-tts:orin \
  python3 /work/orin/train_orin.py "/work/$config" "$@" >/dev/null
echo "started $name: $config $*"
echo "run dir: $host_run  (progress.jsonl, logs/, samples/, checkpoints)"
EOF
echo "following losses (Ctrl-C stops watching; training keeps running)"
watch
