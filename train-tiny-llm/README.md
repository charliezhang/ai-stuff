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

## Interactive playground (mikupad)
```bash
scripts/orin.sh serve                          # runs runs/best4ep/model.pt; or: scripts/orin.sh serve ckpt=runs/base4ep/model.pt
open http://orin.local:8080/mikupad            # new sessions default to this server (API: llama.cpp)
ssh orin docker logs -f tinyllm-serve          # see each request the UI makes
```
`tiny_llm/serve.py` implements the llama.cpp server subset mikupad uses (`/tokenize`, `/detokenize`,
`/completion` with `n_probs` + SSE streaming). Container `tinyllm-serve` restarts with the box and
doesn't block training runs. UI is [mikupad](https://github.com/lmg-anon/mikupad) vendored unmodified (AGPL-3.0, `serve/`).

- Hover any generated token → top-10 next-token probabilities; click one → regenerate from there with that token.
- Max Predict Tokens = 1 → step one token at a time.
- Prompt Preview (Tokens = 1) → live prediction of the next token while you type (500 ms debounce).
- Turn **off** "Post-sampling probabilities" to see the model's raw distribution instead of the one after temperature/top-k/top-p/min-p.

## Layout
```
tiny_llm/   config.py (ModelConfig + presets), model.py, data.py (loader + eval), train.py
scripts/    train.sh, watch.sh, orin.sh, lr_sweep.sh, prepare_data.py (vendored), verify_data.py,
            size_calc.py, bench_precision.py, check_model.py
data/       fineweb_{train,val}.pt (gitignored)
runs/       logs + checkpoints (gitignored)
DESIGN.md   decision log: every choice, alternatives, why
```
