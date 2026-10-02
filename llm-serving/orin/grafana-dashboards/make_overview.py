#!/usr/bin/env python3
"""Generate orin-llm-overview.json: one screen for vLLM serving + Orin health.

Edit the panel list below and re-run:  python3 make_overview.py
"""
import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}
W = "[$__rate_interval]"
panels, y = [], 0


def target(expr, legend="", ref="A"):
    return {"datasource": DS, "expr": expr, "legendFormat": legend, "refId": ref}


def stat(title, expr, unit, x, w=4, h=4, decimals=1, thresholds=None, desc=""):
    steps = thresholds or [{"color": "green", "value": None}]
    panels.append({
        "type": "stat", "title": title, "description": desc, "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [target(expr)],
        "fieldConfig": {"defaults": {"unit": unit, "decimals": decimals,
                                     "thresholds": {"mode": "absolute", "steps": steps}},
                        "overrides": []},
        "options": {"colorMode": "background", "graphMode": "area", "reduceOptions": {"calcs": ["lastNotNull"]}},
    })


def ts(title, targets, unit, x, w=12, h=8, desc="", stack=False, max_=None):
    panels.append({
        "type": "timeseries", "title": title, "description": desc, "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [target(e, l, chr(65 + i)) for i, (e, l) in enumerate(targets)],
        "fieldConfig": {"defaults": {"unit": unit, "max": max_, "custom": {
            "lineWidth": 2, "fillOpacity": 15, "showPoints": "never",
            "stacking": {"mode": "normal" if stack else "none"}}}, "overrides": []},
        "options": {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi"}},
    })


def row(title):
    global y
    panels.append({"type": "row", "title": title, "collapsed": False, "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}})
    y += 1


ORANGE_RED = lambda o, r: [{"color": "green", "value": None}, {"color": "orange", "value": o}, {"color": "red", "value": r}]

# ---------------------------------------------------------------- headline numbers
row("LLM serving (vLLM)")
stat("Output tok/s", f"sum(rate(vllm:generation_tokens_total{W}))", "none", 0)
stat("Input tok/s", f"sum(rate(vllm:prompt_tokens_total{W}))", "none", 4)
stat("TTFT p50", f"histogram_quantile(0.5, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))", "s", 8, decimals=2,
     desc="Time to first token (prefill + queueing), median over 5 min")
stat("Inter-token latency p50", f"histogram_quantile(0.5, sum by (le) (rate(vllm:inter_token_latency_seconds_bucket[5m])))", "s", 12, decimals=3,
     desc="Decode step time per token; 1/ITL = single-stream tok/s")
stat("KV cache used", "max(vllm:kv_cache_usage_perc) * 100", "percent", 16, thresholds=ORANGE_RED(70, 90),
     desc="Share of the paged KV cache blocks in use. Near 100% -> requests get preempted")
stat("Prefix cache hit rate", "sum(rate(vllm:prefix_cache_hits_total[5m])) / clamp_min(sum(rate(vllm:prefix_cache_queries_total[5m])), 1) * 100",
     "percent", 20, desc="Prompt tokens served from cached KV blocks instead of recomputed (5 min)")
y += 4

ts("Token throughput", [(f"sum(rate(vllm:generation_tokens_total{W}))", "output tok/s"),
                        (f"sum(rate(vllm:prompt_tokens_total{W}))", "input (prefill) tok/s")], "none", 0)
ts("Requests", [("sum(vllm:num_requests_running)", "running"), ("sum(vllm:num_requests_waiting)", "waiting"),
                (f"sum(increase(vllm:num_preemptions_total{W}))", "preemptions")], "none", 12)
y += 8

ts("Latency", [("histogram_quantile(0.5, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[1m])))", "TTFT p50"),
               ("histogram_quantile(0.95, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[1m])))", "TTFT p95"),
               ("histogram_quantile(0.5, sum by (le) (rate(vllm:inter_token_latency_seconds_bucket[1m])))", "ITL p50")], "s", 0)
ts("KV cache & prefix cache", [("max(vllm:kv_cache_usage_perc) * 100", "KV cache used %"),
                               ("sum(rate(vllm:prefix_cache_hits_total[1m])) / clamp_min(sum(rate(vllm:prefix_cache_queries_total[1m])), 1) * 100", "prefix hit %")],
   "percent", 12, max_=100)
y += 8

# ---------------------------------------------------------------- compute vs memory
row("Compute & memory bandwidth (decode is bandwidth-bound)")
ts("Compute (vLLM estimate)", [(f"sum(rate(vllm:estimated_flops_per_gpu_total{W})) / 1e12", "model TFLOP/s")], "none", 0, w=8,
   desc="FLOPs vLLM attributes to the model (needs --enable-mfu-metrics). Orin GPU peak is far higher; decode leaves it mostly idle")
ts("Memory bandwidth", [("jetson_emc_bandwidth_bytes_per_second / 1e9", "DRAM total (EMC, all clients)"),
                        (f"sum(rate(vllm:estimated_read_bytes_per_gpu_total{W})) / 1e9", "vLLM est. model reads"),
                        ("jetson_emc_bandwidth_max_bytes_per_second / 1e9", "peak @ current EMC clock")], "none", 8, w=10,
   desc="GB/s. EMC = hardware memory controller (includes CPU, display, etc.). Peak ~205 GB/s at 3199 MHz")
stat("EMC utilization", "jetson_emc_usage_percent", "percent", 18, w=6, h=8, thresholds=ORANGE_RED(70, 90))
y += 8

# ---------------------------------------------------------------- box health
row("Orin health")
ts("GPU", [("jetson_gpu_usage_percent", "GPU util %")], "percent", 0, w=8, max_=100)
ts("CPU (per core)", [("jetson_cpu_usage_percent", "core {{core}}")], "percent", 8, w=8, max_=100)
ts("Unified memory", [("jetson_gpu_memory_used_bytes", "used"), ("jetson_gpu_memory_total_bytes", "total")], "bytes", 16, w=8)
y += 8
ts("Power", [("jetson_power_total_watts", "total"), ("jetson_power_rail_watts", "{{rail}}")], "watt", 0, w=8)
ts("Temperatures", [("jetson_temperature_celsius", "{{zone}}")], "celsius", 8, w=8)
ts("Clocks", [("jetson_gpu_freq_hertz", "GPU"), ("jetson_emc_freq_hertz", "EMC"),
              ("avg(jetson_cpu_freq_hertz)", "CPU avg")], "hertz", 16, w=8)
y += 8

dash = {
    "uid": "orin-llm-overview", "title": "Orin LLM Overview", "tags": ["vllm", "jetson"],
    "timezone": "browser", "refresh": "5s", "time": {"from": "now-15m", "to": "now"},
    "schemaVersion": 39, "panels": panels,
}
out = Path(__file__).parent / "orin-llm-overview.json"
out.write_text(json.dumps(dash, indent=2))
print(f"wrote {out} ({len(panels)} panels)")
