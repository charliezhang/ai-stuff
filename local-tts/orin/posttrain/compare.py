#!/usr/bin/env python3
"""Side-by-side checker metrics for several eval runs.  Usage: compare.py RUN_DIR [RUN_DIR ...]

Lower is better: err% (WER for en, CER for zh, MER for mixed).  Higher is better: en_recall, sim_*, utmos."""
import json
import sys
from pathlib import Path

COLS = [("err_rate", "err%", lambda v: f"{100 * v:.1f}"), ("err_rate_alt", "alt%", lambda v: f"{100 * v:.1f}"),
        ("en_recall", "en_rec", lambda v: f"{v:.2f}"),
        ("sim_ref", "sim_ref", lambda v: f"{v:.3f}"), ("sim_self", "sim_self", lambda v: f"{v:.3f}"),
        ("utmos", "utmos", lambda v: f"{v:.2f}"), ("rate", "syll/s", lambda v: f"{v:.1f}"),
        ("sanity_fail", "flags", str)]
runs = [json.loads((Path(p) / "summary.json").read_text()) for p in sys.argv[1:]]
w = max(len(r["name"]) for r in runs)
for kind in ("zh", "en", "mixed", "all"):
    print(f"\n[{kind}]  " + " ".join(f"{h:>8}" for _, h, _ in COLS))
    for r in runs:
        s = r["metrics"].get(kind)
        if s:
            print(f"  {r['name']:<{w}} " + " ".join(f"{f(s[k]) if k in s else '-':>8}" for k, _, f in COLS))
