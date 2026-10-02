"""Minimal streaming client for the Orin vLLM server: prints answer, TTFT and decode tok/s.

Usage: python3 chat.py "your question"   (ORIN_URL overrides http://orin.local:8000)
"""
import json, os, time, urllib.request, sys
q = sys.argv[1] if len(sys.argv) > 1 else "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Answer briefly."
body = {"model": "qwen3.6-35b", "stream": True, "max_tokens": 1500, "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": q}]}
req = urllib.request.Request(os.environ.get("ORIN_URL", "http://orin.local:8000") + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
t0 = time.time(); first = ttfa = None; R = []; A = []; usage = None
with urllib.request.urlopen(req) as r:
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]": continue
        ev = json.loads(line[6:]); usage = ev.get("usage") or usage
        for ch in ev.get("choices", []):
            d = ch["delta"]; rc = d.get("reasoning_content") or d.get("reasoning")
            if rc: first = first or time.time(); R.append(rc)
            if d.get("content"): first = first or time.time(); ttfa = ttfa or time.time(); A.append(d["content"])
t1 = time.time(); n = usage["completion_tokens"]
print(f"answer: {''.join(A).strip()[:200]}")
print(f"TTFT {first-t0:.2f}s | reasoning {len(''.join(R))} chars | completion {n} tok in {t1-first:.1f}s → {n/(t1-first):.1f} tok/s")
