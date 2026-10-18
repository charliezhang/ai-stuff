"""Table of finished runs: final val loss/bpb, delta vs. a reference run, throughput.
Usage: python3 scripts/summarize.py [glob=runs/ab_*] [ref=ab_base]"""
import glob, json, sys

pattern = sys.argv[1] if len(sys.argv) > 1 else "runs/ab_*"
ref_name = sys.argv[2] if len(sys.argv) > 2 else "ab_base"
rows = {}
for path in sorted(glob.glob(f"{pattern}/log.jsonl")):
    recs = [json.loads(l) for l in open(path)]
    cfg = next(r for r in recs if r["event"] == "config")
    fin = next((r for r in recs if r["event"] == "final"), None)
    tps = [r["tok_per_s"] for r in recs if r["event"] == "train" and r["step"] >= 20]
    rows[cfg["tc.name"]] = dict(params=cfg["params"], final=fin, tps=sum(tps) / len(tps) if tps else 0,
                                mem=max((r["mem_gb"] for r in recs if r["event"] == "train"), default=0))
ref = rows.get(ref_name, {}).get("final")
print(f"{'run':16} {'params':>7} {'val_loss':>8} {'Δ vs ref':>8} {'val_bpb':>7} {'tok/s':>7} {'min':>5} {'mem':>5}")
for name, r in sorted(rows.items(), key=lambda kv: kv[1]["final"]["val_loss"] if kv[1]["final"] else 99):
    f = r["final"]
    if f is None:
        print(f"{name:16} {r['params']/1e6:6.1f}M  (running)  {r['tps']/1e3:6.1f}k"); continue
    d = f"{f['val_loss'] - ref['val_loss']:+.3f}" if ref else ""
    print(f"{name:16} {r['params']/1e6:6.1f}M {f['val_loss']:8.3f} {d:>8} {f['val_bpb']:7.3f} "
          f"{r['tps']/1e3:6.1f}k {f['elapsed_min']:5.1f} {r['mem']:4.1f}G")
