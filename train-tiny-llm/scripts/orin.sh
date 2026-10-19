#!/usr/bin/env bash
# Orin helper. Usage:
#   scripts/orin.sh sync                 # rsync project (code + data) to the Orin
#   scripts/orin.sh run <cmd...>         # run a command in the training container
#   scripts/orin.sh bg <name> <cmd...>   # detached run (survives ssh disconnect), log in runs/<name>.out
#   scripts/orin.sh logs <name>          # follow a detached run
#   scripts/orin.sh serve [ckpt=...]     # (re)start the model server + mikupad UI on port 8080
#   scripts/orin.sh services stop|start  # free / restore GPU memory (vllm, tts, pron)
set -euo pipefail

HOST=orin
REMOTE_DIR='~/train-tiny-llm'
IMAGE=vllm/vllm-openai:latest      # already on the box: torch 2.13+cu130, triton, tiktoken
SERVICES=(vllm tts pron)

case "${1:-}" in
  sync)
    rsync -az --delete --exclude .venv --exclude .git --exclude runs/ --exclude '__pycache__' \
      "$(dirname "$0")/../" "$HOST:$REMOTE_DIR/"
    ;;
  run)
    shift
    tty=""; [ -t 0 ] && tty="-t"
    ssh $tty "$HOST" "cd $REMOTE_DIR && mkdir -p runs && docker run --rm ${tty:+-it} --runtime nvidia --gpus all --ipc=host \
      -v \$PWD:/work -w /work -e PYTHONUNBUFFERED=1 --entrypoint '' $IMAGE $*"
    ;;
  bg)
    name=$2; shift 2
    ssh "$HOST" "cd $REMOTE_DIR && mkdir -p runs && docker rm tiny-$name >/dev/null 2>&1; docker run -d --name tiny-$name --runtime nvidia --gpus all --ipc=host \
      -v \$PWD:/work -w /work -e PYTHONUNBUFFERED=1 --entrypoint '' $IMAGE bash -c '$* > runs/$name.out 2>&1'"
    ;;
  logs)
    ssh "$HOST" "tail -f $REMOTE_DIR/runs/$2.out"
    ;;
  serve)
    shift
    ssh "$HOST" "cd $REMOTE_DIR && docker rm -f tinyllm-serve >/dev/null 2>&1; docker run -d --name tinyllm-serve \
      --restart unless-stopped --runtime nvidia --gpus all -p 8080:8080 -v \$PWD:/work -w /work \
      -e PYTHONUNBUFFERED=1 --entrypoint '' $IMAGE python3 -m tiny_llm.serve $*"
    ;;
  services)
    ssh "$HOST" "docker ${2:?stop|start} ${SERVICES[*]} && free -g | head -2"
    ;;
  *) sed -n '2,8p' "$0"; exit 1 ;;
esac
