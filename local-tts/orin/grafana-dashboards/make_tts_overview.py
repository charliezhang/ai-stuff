#!/usr/bin/env python3
"""Generate tts-overview.json: one screen for the local TTS server (tts_* metrics) + Orin health.

Mirrors llm-serving's "Orin LLM Overview": audio seconds play the role of tokens, TTFA of TTFT,
RTF of inter-token latency. Edit the panels below and re-run:  python3 make_tts_overview.py
"""
import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}
W = "[$__rate_interval]"
CODEC_HZ, CODEBOOKS = 12.5, 16          # Qwen3-TTS-Tokenizer-12Hz
panels, y = [], 0


def target(expr, legend="", ref="A"):
    return {"datasource": DS, "expr": expr, "legendFormat": legend, "refId": ref}


def steps(*pairs):
    return {"mode": "absolute", "steps": [{"color": c, "value": v} for v, c in pairs]}


GREEN = steps((None, "green"))


def stat(title, expr, unit, x, w=4, h=4, decimals=2, thresholds=GREEN, desc=""):
    panels.append({
        "type": "stat", "title": title, "description": desc, "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h}, "targets": [target(expr)],
        "fieldConfig": {"defaults": {"unit": unit, "decimals": decimals, "thresholds": thresholds,
                                     "noValue": "–"}, "overrides": []},
        "options": {"colorMode": "background", "graphMode": "area", "reduceOptions": {"calcs": ["lastNotNull"]}},
    })


def ts(title, targets, unit, x, w=12, h=8, desc="", stack=False, max_=None, bars=False, line_at=None):
    custom = {"lineWidth": 2, "fillOpacity": 15, "showPoints": "never",
              "drawStyle": "bars" if bars else "line",
              "stacking": {"mode": "normal" if stack else "none"}}
    thresholds = GREEN
    if line_at is not None:                       # dashed reference line, e.g. RTF = 1.0
        custom["thresholdsStyle"] = {"mode": "dashed"}
        thresholds = steps((None, "transparent"), (line_at, "red"))
    panels.append({
        "type": "timeseries", "title": title, "description": desc, "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [target(e, l, chr(65 + i)) for i, (e, l) in enumerate(targets)],
        "fieldConfig": {"defaults": {"unit": unit, "max": max_, "custom": custom, "thresholds": thresholds},
                        "overrides": []},
        "options": {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi"}},
    })


def row(title):
    global y
    panels.append({"type": "row", "title": title, "collapsed": False, "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}})
    y += 1


def q(p, metric, window="5m"):
    return f"histogram_quantile({p}, sum by (le) (rate({metric}_bucket[{window}])))"


AUDIO_RATE = f"sum(rate(tts_generated_audio_seconds_total{W}))"
COMPUTE_RATE = f"sum(rate(tts_compute_seconds_total{W}))"

# ---------------------------------------------------------------- headline numbers
row("TTS serving (Qwen3-TTS + faster-qwen3-tts)")
stat("Speed (× real time)", "sum(rate(tts_request_audio_seconds_sum[5m])) / sum(rate(tts_compute_seconds_total[5m]))", "x", 0,
     thresholds=steps((None, "red"), (1, "orange"), (1.1, "green")),
     desc="Audio seconds produced per second of generation (= 1/RTF), last 5 min. >1 keeps up with playback. "
          "Both series step at request end, so rate() extrapolation cancels out in the ratio")
stat("RTF p50", q(0.5, "tts_rtf"), "none", 4, thresholds=steps((None, "green"), (0.9, "orange"), (1, "red")),
     desc="Compute s / audio s per request, median over 5 min. <1 = faster than real time")
stat("Time to first audio p50", q(0.5, "tts_time_to_first_audio_seconds"), "s", 8,
     thresholds=steps((None, "green"), (1, "orange"), (2, "red")),
     desc="From GPU lock acquired to first streamed chunk (prefill + first codec frames + decode)")
stat("Requests / min", "sum(rate(tts_request_duration_seconds_count[5m])) * 60", "none", 12, decimals=1,
     desc="From the unlabeled duration histogram: labeled counters miss the first request per new label set")
stat("GPU lock busy", f"{COMPUTE_RATE} * 100", "percent", 16, decimals=0,
     thresholds=steps((None, "green"), (70, "orange"), (95, "red")),
     desc="Share of time a request is generating. One generation at a time (CUDA graphs + static KV cache)")
stat("Stalled streams (1h)",
     "sum(increase(tts_stream_stalled_requests_total[1h])) / clamp_min(sum(increase(tts_request_duration_seconds_count[1h])), 1) * 100",
     "percent", 20, decimals=0, thresholds=steps((None, "green"), (1, "orange"), (10, "red")),
     desc="Requests where generation fell behind a listener playing from the first chunk")
y += 4

ts("Audio throughput", [(AUDIO_RATE, "audio s generated / s"), (COMPUTE_RATE, "generation s / s (lock busy)")],
   "none", 0, desc="When busy, the gap between the lines is the real-time margin")
ts("Codec tokens", [(f"{AUDIO_RATE} * {CODEC_HZ}", "frames/s = talker steps/s"),
                    (f"{AUDIO_RATE} * {CODEC_HZ * (CODEBOOKS - 1)}", "code-predictor steps/s"),
                    (f"{AUDIO_RATE} * {CODEC_HZ * CODEBOOKS}", "codec tokens/s")], "none", 12,
   desc=f"Each 80 ms frame = 1 talker step (codebook 0) + {CODEBOOKS - 1} code-predictor steps; "
        f"{CODEBOOKS} tokens/frame. Derived from audio seconds × {CODEC_HZ} Hz")
y += 8

ts("Real-time factor per request", [(q(0.5, "tts_rtf", "1m"), "p50"), (q(0.95, "tts_rtf", "1m"), "p95")], "none", 0,
   line_at=1.0, desc="Above the dashed line (1.0) a stream can't keep up with playback")
ts("Latency", [(q(0.5, "tts_time_to_first_audio_seconds", "1m"), "TTFA p50"),
               (q(0.95, "tts_time_to_first_audio_seconds", "1m"), "TTFA p95"),
               (q(0.95, "tts_queue_wait_seconds", "1m"), "queue wait p95"),
               (q(0.5, "tts_request_duration_seconds", "1m"), "request p50")], "s", 12)
y += 8

ts("Requests by text language", [(f"sum by (text_kind) (increase(tts_requests_total{W}))", "{{text_kind}}")],
   "none", 0, w=8, stack=True, bars=True, desc="zh / en / mixed (code-switched) input text")
ts("Requests by voice", [(f"sum by (voice) (increase(tts_requests_total{W}))", "{{voice}}")],
   "none", 8, w=8, stack=True, bars=True)
ts("Requests by endpoint & status", [(f"sum by (endpoint, status) (increase(tts_requests_total{W}))", "{{endpoint}} {{status}}")],
   "none", 16, w=8, stack=True, bars=True, desc="ui = Gradio, api-wav / api-pcm = /v1/audio/speech")
y += 8

ts("Queue", [("sum(tts_requests_running)", "running"), ("sum(tts_requests_waiting)", "waiting")], "none", 0, w=8)
ts("Audio per request", [(q(0.5, "tts_request_audio_seconds", "15m"), "p50 length"),
                         (q(0.95, "tts_request_audio_seconds", "15m"), "p95 length")], "s", 8, w=8)
ts("Stream stalls", [(f"sum(rate(tts_stream_stall_seconds_total{W}))", "stall s / s"),
                     (f"sum(increase(tts_stream_stalled_requests_total{W}))", "stalled requests")], "none", 16, w=8,
   desc="Time a listener would have waited mid-stream")
y += 8

# ---------------------------------------------------------------- compute & memory
row("Compute & memory (TTS decode is CPU-launch bound without CUDA graphs)")
ts("TTS GPU memory", [('tts_gpu_memory_bytes{kind="allocated"}', "allocated"),
                      ('tts_gpu_memory_bytes{kind="reserved"}', "reserved (PyTorch cache)")], "bytes", 0, w=8)
ts("GPU utilization", [("jetson_gpu_usage_percent", "GPU util %")], "percent", 8, w=8, max_=100,
   desc="Shared with vLLM if it's running. Eager TTS sat at ~12% here")
ts("Memory bandwidth", [("jetson_emc_bandwidth_bytes_per_second / 1e9", "DRAM (EMC, all clients) GB/s"),
                        ("jetson_emc_bandwidth_max_bytes_per_second / 1e9", "peak GB/s")], "none", 16, w=8)
y += 8

# ---------------------------------------------------------------- box health
row("Orin health")
ts("CPU (per core)", [("jetson_cpu_usage_percent", "core {{core}}")], "percent", 0, w=8, max_=100,
   desc="One core pinned at 100% while the GPU idles = kernel-launch bound")
ts("Power", [("jetson_power_total_watts", "total"), ("jetson_power_rail_watts", "{{rail}}")], "watt", 8, w=8)
ts("Unified memory", [("jetson_gpu_memory_used_bytes", "used"), ("jetson_gpu_memory_total_bytes", "total")], "bytes", 16, w=8)
y += 8
ts("Temperatures", [("jetson_temperature_celsius", "{{zone}}")], "celsius", 0, w=12)
ts("Clocks", [("jetson_gpu_freq_hertz", "GPU"), ("jetson_emc_freq_hertz", "EMC"),
              ("avg(jetson_cpu_freq_hertz)", "CPU avg")], "hertz", 12, w=12)
y += 8

dash = {
    "uid": "tts-overview", "title": "Local TTS Overview", "tags": ["tts", "jetson"],
    "timezone": "browser", "refresh": "5s", "time": {"from": "now-15m", "to": "now"},
    "schemaVersion": 39, "panels": panels,
}
out = Path(__file__).parent / "tts-overview.json"
out.write_text(json.dumps(dash, indent=2))
print(f"wrote {out} ({len(panels)} panels)")
