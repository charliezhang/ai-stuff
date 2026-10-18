#!/usr/bin/env bash
# Run every line of an ablation list sequentially (inside the training container).
# Usage: scripts/ablate.sh [list=scripts/ablations.txt]
set -u
COMMON="train_tokens=50e6 save_model=false"
grep -vE '^\s*(#|$)' "${1:-scripts/ablations.txt}" | while read -r name overrides; do
  if grep -q '^{"event": "final"' "runs/$name/log.jsonl" 2>/dev/null; then
    echo "skip $name (already finished)"; continue
  fi
  python3 -m tiny_llm.train name="$name" $COMMON $overrides || echo "FAILED $name"
done
echo "ABLATIONS DONE"
