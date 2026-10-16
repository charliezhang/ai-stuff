#!/usr/bin/env bash
# Launch a training run on the Orin (detached container) and follow its live log.
# Usage: scripts/train.sh <name> [field=value ...]     any ModelConfig/TrainConfig field
#   scripts/train.sh base4ep lr=2e-3                      # the full 4-epoch run
#   scripts/train.sh moe8 n_experts=8 train_tokens=100e6  # 1-epoch MoE comparison
# Ctrl-C only stops watching; resume with scripts/watch.sh <name>.
# Stop a run: ssh orin docker stop tiny-<name>
set -euo pipefail
name=${1:?usage: scripts/train.sh <name> [field=value ...]}
shift
cd "$(dirname "$0")/.."

if ssh orin "docker ps --filter name=tiny- --format '{{.Names}}'" | grep -q .; then
  echo "another run is using the GPU:"; ssh orin "docker ps --filter name=tiny- --format '  {{.Names}} {{.Status}}'"
  exit 1
fi
scripts/orin.sh sync
scripts/orin.sh bg "$name" python3 -m tiny_llm.train name="$name" "$@" >/dev/null
echo "launched tiny-$name  (log: orin:~/train-tiny-llm/runs/$name.out, metrics: runs/$name/log.jsonl)"
exec scripts/watch.sh "$name"
