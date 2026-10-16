#!/usr/bin/env bash
# Follow a run's training loss live (Ctrl-C stops watching, not training).
# Usage: scripts/watch.sh <name>        e.g. scripts/watch.sh lrsweep
set -euo pipefail
name=${1:?usage: scripts/watch.sh <name>}

ssh orin "tail -n +1 -F ~/train-tiny-llm/runs/$name.out 2>/dev/null" | awk '
  /Traceback|Error|Killed|OutOfMemory/ { print "!! " $0; fflush(); next }
  /^event=/ {
    delete f
    for (i = 1; i <= NF; i++) { split($i, kv, "="); f[kv[1]] = kv[2] }
    if (f["event"] == "config")
      printf "\n== run %s: %.1fM params, %s steps\n", f["tc.name"], f["params"] / 1e6, f["steps"]
    else if (f["event"] == "train")
      printf "step %5s  loss %-7s lr %-9s %6.1fk tok/s  mfu %-6s mem %sGB %s\n",
             f["step"], f["loss"], f["lr"], f["tok_per_s"] / 1000, f["mfu"], f["mem_gb"],
             ("moe_max_over_mean" in f) ? "moe " f["moe_max_over_mean"] : ""
    else
      printf "step %5s  ** %s val_loss %s  val_bpb %s  (%s min)\n",
             f["step"], toupper(f["event"]), f["val_loss"], f["val_bpb"], f["elapsed_min"]
    fflush()
  }'
