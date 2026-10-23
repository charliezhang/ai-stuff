"""Follow a pocket-tts run's losses on the Orin host (stdlib only, no container needed).

    python3 watch_loss.py <run_dir> [--container tiny-tts-train] [--follow]

Reads <run_dir>/progress.jsonl (train/valid/checkpoint events written by training.train) and prints
one line per logged step with an ETA, valid losses when they arrive, and errors from the newest
<run_dir>/logs/train_*.log. With --container, also stops when that container exits and shows its
last log lines, since a crash before the first step never reaches progress.jsonl.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ALERT = ("Traceback", "Error", "error:", "non-finite", "Killed", "out of memory", "SystemExit")
NOTABLE = ("first step took", "samples at step", "precomputing latents", "stitch_frames=", "wrote ", "trainable params")


def fmt_eta(seconds: float) -> str:
    if seconds <= 0 or seconds != seconds:
        return "-"
    h, m = divmod(int(seconds) // 60, 60)
    return f"{h // 24}d{h % 24:02d}h" if h >= 24 else f"{h}h{m:02d}m"


def container_running(name: str) -> bool:
    out = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", name],
                         capture_output=True, text=True)
    return out.stdout.strip() == "true"


def container_latest(name: str) -> str:
    """The container's newest output line; tqdm bars redraw with \\r, keep the last redraw."""
    out = subprocess.run(["docker", "logs", "--tail", "3", name], capture_output=True, text=True)
    lines = [seg.strip() for seg in (out.stdout + out.stderr).replace("\r", "\n").splitlines()]
    lines = [s for s in lines if s and "Warning" not in s and "warn(" not in s]
    return lines[-1][:150] if lines else ""


def container_tail(name: str, n: int = 30) -> str:
    out = subprocess.run(["docker", "logs", "--tail", str(n), name], capture_output=True, text=True)
    return (out.stdout + out.stderr).rstrip()


class Watcher:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.progress = run_dir / "progress.jsonl"
        self.pos = 0
        self.log_path: Path | None = None
        self.log_pos = 0
        self.max_steps = 0
        self.last_step = 0
        self.header_done = False
        self.awaiting_step = True

    def poll_progress(self) -> None:
        if not self.progress.exists():
            return
        with self.progress.open() as f:
            f.seek(self.pos)
            for line in f:
                if not line.endswith("\n"):
                    break
                self.pos += len(line.encode())
                try:
                    self.show(json.loads(line))
                except json.JSONDecodeError:
                    continue

    def show(self, ev: dict) -> None:
        kind, step = ev.get("type"), ev.get("step", 0)
        m = ev.get("metrics") or {}
        if kind == "run_start":
            self.max_steps = int(ev.get("max_steps") or 0)
            print(f"\n== run start at step {step} / {self.max_steps}  ({ev.get('time', '')})")
            self.header_done = False
            self.awaiting_step = True
        elif kind == "train":
            if not self.header_done:
                print(f"{'step':>8} {'loss':>8} {'flow':>8} {'eos':>7} {'grad':>6} "
                      f"{'it/s':>6} {'lr':>9}  eta")
                self.header_done = True
            it_s = ev.get("it_s") or 0.0
            eta = (self.max_steps - step) / it_s if it_s and self.max_steps else float("nan")
            print(f"{step:>8} {m.get('loss', float('nan')):>8.4f} {m.get('flow_loss', float('nan')):>8.4f} "
                  f"{m.get('eos_loss', float('nan')):>7.4f} {ev.get('grad_norm', 0):>6.2f} "
                  f"{it_s:>6.2f} {ev.get('lr', 0):>9.2e}  {fmt_eta(eta)}")
            self.last_step = step
            self.awaiting_step = False
        elif kind == "valid":
            parts = "  ".join(f"{k} {v:.4f}" for k, v in m.items() if k in ("loss", "flow_loss", "eos_loss"))
            print(f"{step:>8} ** VALID  {parts}")
        elif kind == "checkpoint":
            print(f"{step:>8} -- checkpoint saved")
        sys.stdout.flush()

    def poll_log(self) -> None:
        logs = sorted((self.run_dir / "logs").glob("train_*.log"))
        if not logs:
            return
        if logs[-1] != self.log_path:
            self.log_path, self.log_pos = logs[-1], 0
        with self.log_path.open(errors="replace") as f:
            f.seek(self.log_pos)
            chunk = f.read()
            self.log_pos = f.tell()
        for line in chunk.splitlines():
            if any(a in line for a in ALERT) or any(k in line for k in NOTABLE):
                print(f"   | {line.strip()}")
        sys.stdout.flush()


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    if not args:
        print(__doc__.strip().split("\n\n")[1])
        return 2
    follow = "--follow" in argv
    container = None
    if "--container" in argv:
        i = argv.index("--container")
        container = argv[i + 1]
        args.remove(container)
    w = Watcher(Path(args[0]))
    waited = 0
    while True:
        w.poll_progress()
        w.poll_log()
        if not follow:
            return 0
        if container and not container_running(container):
            w.poll_progress()
            w.poll_log()
            if w.max_steps and w.last_step >= w.max_steps:
                print(f"\n== training finished at step {w.last_step}; weights: {w.run_dir}/model.safetensors")
                return 0
            print(f"\n== container {container} stopped at step {w.last_step}. Last output:\n"
                  f"{container_tail(container)}")
            return 1
        if w.awaiting_step and waited % 60 == 0:
            # Before the first step: model build, compilation, the one-time latents precompute.
            status = container_latest(container) if container else ""
            print(f"   [{waited // 60:3d} min] waiting for the first step   {status}")
            sys.stdout.flush()
        time.sleep(5)
        waited += 5


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print("\nstopped watching; training keeps running")
    except BrokenPipeError:  # piped into head/less that exited
        sys.stderr.close()
