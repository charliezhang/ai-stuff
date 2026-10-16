# train-tiny-llm

Pretrain a tiny LLM from scratch on a Jetson AGX Orin 64GB, one design decision at a time.

**Compatibility target:** [NanoGPT Slowrun](https://github.com/qlabs-eng/slowrun) (tiny track).
- Data: 100M train tokens + 10M val tokens from `HuggingFaceFW/fineweb` `sample-10BT`, in slowrun's `.pt` format
  (same tokens; slowrun's file hashes don't match across torch environments, so `scripts/verify_data.py` checks token-content hashes)
- Tokenizer: GPT-2 (tiktoken), vocab 50257
- Metric: mean cross-entropy (nats/token) on the 10M val tokens, seq len 2048; bits-per-byte logged alongside

Reference points (8×H100, ≤15 min, 300M params): baseline 3.428 → record 3.295.

## Running (from the Mac)
```bash
scripts/train.sh <name> [field=value ...]   # sync, launch detached on the Orin, follow live loss
scripts/watch.sh <name>                     # re-attach to a run's live log (Ctrl-C only stops watching)
ssh orin docker stop tiny-<name>            # stop a run
scripts/orin.sh services stop|start         # free / restore GPU memory (vllm, tts, pron)
```
Overrides are any `ModelConfig`/`TrainConfig` field, e.g. `lr=2e-3`, `n_experts=8`, `train_tokens=100e6`.
Raw log: `orin:~/train-tiny-llm/runs/<name>.out`; metrics: `runs/<name>/log.jsonl`; checkpoint: `runs/<name>/model.pt`.

## Layout
```
tiny_llm/   config.py (ModelConfig + presets), model.py, data.py (loader + eval), train.py
scripts/    train.sh, watch.sh, orin.sh, lr_sweep.sh, prepare_data.py (vendored), verify_data.py,
            size_calc.py, bench_precision.py, check_model.py
data/       fineweb_{train,val}.pt (gitignored)
runs/       logs + checkpoints (gitignored)
DESIGN.md   decision log: every choice, alternatives, why
```
