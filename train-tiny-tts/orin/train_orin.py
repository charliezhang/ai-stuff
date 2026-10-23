"""Run pocket-tts training on the Orin, unchanged, from a config plus command-line overrides.

    python3 /work/orin/train_orin.py /work/orin/configs/orin_small.yaml [key=value ...] [--check]

- `key=value` overrides any TrainArgs field, dotted for nested ones (`batch_size=32`,
  `data.loader_procs=2`, `max_steps=200`); values are parsed as YAML. The resolved config is written
  to <run_dir>/orin_config.yaml and handed to `training.train.main`.
- Preflight before the (slow) model build: CUDA works, the manifests exist, and the weights carry a
  real Mimi encoder. The ungated pocket-tts release zeroes Mimi's encoder, and pocket-tts' training
  path loads it without checking (only the standalone precompute script checks); every latent would
  then be the same and nothing is learned. `--allow-dummy-encoder` skips the check for plumbing and
  speed tests only.
- Imported rather than run as `python -m training.train`, so pocket-tts does not set
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments, which relies on virtual-memory APIs the Jetson's
  integrated GPU may not support. Export PYTORCH_CUDA_ALLOC_CONF yourself to try it.
"""

import json
import sys
from pathlib import Path

import yaml


def set_dotted(cfg: dict, dotted: str, value) -> None:
    # model_overrides is itself keyed by dotted pocket-tts config paths: keep the rest whole.
    if dotted.startswith("model_overrides."):
        cfg.setdefault("model_overrides", {})[dotted.removeprefix("model_overrides.")] = value
        return
    node = cfg
    *parents, leaf = dotted.split(".")
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = value


def encoder_transformer_max(weights: Path) -> float:
    """Largest |w| over mimi.encoder_transformer.*, read without loading the whole file."""
    from safetensors import safe_open

    with safe_open(str(weights), framework="pt") as f:
        keys = [k for k in f.keys() if k.startswith("mimi.encoder_transformer.")]
        if not keys:
            return 0.0
        return max(float(f.get_tensor(k).abs().max()) for k in keys)


def preflight(cfg: dict, allow_dummy_encoder: bool) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible: start the container with --runtime nvidia")
    free, total = torch.cuda.mem_get_info()
    name = torch.cuda.get_device_name(0)
    cap = ".".join(map(str, torch.cuda.get_device_capability(0)))
    print(f"[orin] {name} sm_{cap.replace('.', '')} | torch {torch.__version__} | "
          f"GPU memory free {free / 2**30:.1f} of {total / 2**30:.1f} GiB", flush=True)

    data = cfg.get("data", {})
    for key in ("train_jsonl", "valid_jsonl"):
        path = data.get(key)
        if path and not Path(path).exists():
            raise SystemExit(f"data.{key} not found: {path}\nRun prepare_training_on_orin.sh first.")
        if path:
            with open(path) as f:
                audio = json.loads(f.readline())["path"]
            if not Path(audio).exists():
                raise SystemExit(f"{path} reads audio that is not there: {audio}\n"
                                 "Re-run prepare_training_on_orin.sh (it copies the chapter files).")
    train = Path(data.get("train_jsonl", ""))
    if train.stem.endswith("_latents") and not train.with_suffix(".meta.json").exists():
        raise SystemExit(f"{train} has no {train.with_suffix('.meta.json').name}: the latents "
                         "precompute did not finish. Re-run prepare_training_on_orin.sh.")

    weights = (cfg.get("model_overrides") or {}).get("weights_path")
    if weights is None:
        weights = yaml.safe_load(Path(cfg["model_config"]).read_text()).get("weights_path")
    if weights is None or str(weights).startswith(("hf://", "http")):
        raise SystemExit(f"weights_path must be a local file on the Orin, got {weights!r}")
    if not Path(weights).exists():
        raise SystemExit(f"weights not found: {weights}")
    if encoder_transformer_max(Path(weights)) == 0.0:
        msg = (f"{weights} has an all-zero Mimi encoder transformer (the release without voice "
               "cloning): every latent would come out the same and the model would learn nothing.")
        if not allow_dummy_encoder:
            raise SystemExit(msg + "\nUse the gated kyutai/pocket-tts weights "
                             "(prepare_training_on_orin.sh fetches them).")
        print(f"[orin] WARNING {msg} Continuing because of --allow-dummy-encoder.", flush=True)


def main(argv: list[str]) -> None:
    flags = {a for a in argv if a.startswith("--")}
    args = [a for a in argv if not a.startswith("--")]
    unknown = flags - {"--check", "--allow-dummy-encoder"}
    if not args or unknown:
        raise SystemExit(__doc__.strip().split("\n\n")[1])
    config_path, overrides = args[0], args[1:]
    cfg = yaml.safe_load(Path(config_path).read_text())
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise SystemExit(f"override {item!r} is not key=value")
        set_dotted(cfg, key, yaml.safe_load(raw))

    preflight(cfg, "--allow-dummy-encoder" in flags)
    if "--check" in flags:
        print("[orin] preflight ok", flush=True)
        return

    run_dir = Path(cfg.get("run_dir", "runs/debug"))
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = run_dir / "orin_config.yaml"
    resolved.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"[orin] config {config_path} + {overrides or 'no overrides'} -> {resolved}", flush=True)

    from training import train

    train.main(str(resolved))


if __name__ == "__main__":
    main(sys.argv[1:])
