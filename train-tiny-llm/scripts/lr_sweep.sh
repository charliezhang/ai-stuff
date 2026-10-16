#!/usr/bin/env bash
# Short LR sweep: same model/data/seed, 15M tokens each, full val eval at the end.
set -u
for lr in ${LRS:-1e-3 2e-3 4e-3}; do
  python3 -m tiny_llm.train name=lr_$lr lr=$lr train_tokens=15e6 eval_every=1000
done
