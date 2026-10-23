"""Stage pocket-tts training inputs for the Orin, on the Mac (run with pocket-tts' venv and PYTHONPATH).

    python stage_orin_data.py weights   --out build/orin-data/weights
    python stage_orin_data.py manifests --src data/hifitts2_200h --out build/orin-data/hifitts2_200h

weights    Fetch the gated kyutai/pocket-tts checkpoint (for its Mimi codec) and the tokenizer, and
           check the Mimi encoder is real: the ungated release zeroes it, and training on that
           learns nothing.
manifests  Rewrite train/valid manifests for the container: audio paths become /data/audio/<file>,
           and rows without word alignments are dropped (pocket-tts' loader raises on them by design:
           a voice prompt needs a word boundary to cut at, so one such row would stop training).
           Also writes audio_files.txt, the chapter files those rows read, for rsync --files-from.
           --subset N keeps the first N train rows (and N/8 valid rows) for a quick trial run.
"""

import argparse
import filecmp
import json
import shutil
import sys
from pathlib import Path

ENGLISH = "pocket_tts/config/english.yaml"
AUDIO_DIR = "/data/audio"  # where the chapter files live inside tiny-tts:orin


def cmd_weights(a: argparse.Namespace) -> None:
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    from safetensors import safe_open

    from pocket_tts.utils.config import load_config
    from pocket_tts.utils.utils import download_if_necessary

    cfg = load_config(ENGLISH)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        weights = Path(a.weights) if a.weights else download_if_necessary(str(cfg.weights_path))
    except (GatedRepoError, HfHubHTTPError) as e:
        sys.exit(
            f"Could not download {cfg.weights_path}:\n  {type(e).__name__}: {str(e).splitlines()[0]}\n"
            "The Mimi encoder needed for training is only in the gated repo. Once:\n"
            "  1. open https://huggingface.co/kyutai/pocket-tts and accept the terms\n"
            "  2. create a read token at https://huggingface.co/settings/tokens\n"
            "  3. cd pocket-tts && uv run hf auth login\n"
            "then re-run. (Or pass --weights /path/to/model.safetensors.)"
        )
    with safe_open(str(weights), framework="pt") as f:
        # Largest |w| per encoder part; the ungated release zeroes both parts entirely.
        peaks = [max((float(f.get_tensor(k).abs().max()) for k in f.keys() if k.startswith(part)),
                     default=0.0) for part in ("mimi.encoder.", "mimi.encoder_transformer.")]
    if min(peaks) == 0.0 and not a.allow_dummy_encoder:
        sys.exit(f"{weights} has all-zero Mimi encoder weights (the release without voice cloning); "
                 "training on it learns nothing. Use the gated kyutai/pocket-tts weights.")
    tokenizer = download_if_necessary(str(cfg.flow_lm.lookup_table.tokenizer_path))
    for src, name in ((weights, "model.safetensors"), (tokenizer, "tokenizer.json")):
        dst = out / name
        if not dst.exists() or not filecmp.cmp(src, dst, shallow=False):
            shutil.copyfile(src, dst)
    print(f"weights -> {out}/model.safetensors ({(out / 'model.safetensors').stat().st_size / 1e6:.0f} MB), "
          f"tokenizer -> {out}/tokenizer.json", flush=True)


def cmd_manifests(a: argparse.Namespace) -> None:
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files: set[str] = set()
    for split, limit in (("train", a.subset), ("valid", a.subset // 8)):
        rows = [json.loads(line) for line in (src / f"{split}_aligned.jsonl").read_text().splitlines()
                if line.strip()]
        kept = [r for r in rows if r.get("words")][: limit or None]
        lines = []
        for r in kept:
            name = Path(r["path"]).name
            files.add(name)
            lines.append(json.dumps({**r, "path": f"{AUDIO_DIR}/{name}"}) + "\n")
        text = "".join(lines)
        dst = out / f"{split}_aligned.jsonl"
        if not dst.exists() or dst.read_text() != text:
            dst.write_text(text)
            # Latents are named by row index: a changed train manifest invalidates them on the Orin.
            for stale in out.glob(f"{split}_aligned_latents*"):
                stale.unlink()
        hours = sum(r["duration"] for r in kept) / 3600
        print(f"{split}: {len(kept)} rows ({hours:.1f} h); {len(rows) - len([r for r in rows if r.get('words')])} "
              f"rows without word alignments dropped", flush=True)
    (out / "audio_files.txt").write_text("".join(f"{n}\n" for n in sorted(files)))
    print(f"{len(files)} chapter files referenced -> {out}/audio_files.txt", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("weights")
    w.add_argument("--out", required=True)
    w.add_argument("--weights", help="use this model.safetensors instead of downloading it")
    w.add_argument("--allow-dummy-encoder", action="store_true", help="plumbing tests only")
    m = sub.add_parser("manifests")
    m.add_argument("--src", required=True, help="pocket-tts data dir with *_aligned.jsonl")
    m.add_argument("--out", required=True)
    m.add_argument("--subset", type=int, default=0, help="first N train rows only (trial run)")
    a = p.parse_args()
    {"weights": cmd_weights, "manifests": cmd_manifests}[a.cmd](a)


if __name__ == "__main__":
    main()
