"""Serve a checkpoint behind a llama.cpp-compatible API, plus the mikupad UI.

Implements the subset of llama-server that mikupad (and loom-style tools) use:
  POST /tokenize    {content}                -> {tokens}
  POST /detokenize  {tokens}                 -> {content}
  POST /completion  {prompt, n_predict, n_probs, stream, temperature, top_k, top_p,
                     min_p, repeat_penalty, repeat_last_n, seed, stop, ignore_eos,
                     post_sampling_probs, logit_bias}
                    -> SSE chunks {content, stop, completion_probabilities:[{content, probs|top_probs}]}
  GET  /health, /mikupad (UI)

Usage: python3 -m tiny_llm.serve [ckpt=runs/best4ep/model.pt] [port=8080]
"""
import json
import os
import sys
import threading

import tiktoken
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .config import ModelConfig
from .model import GPT

args = dict(a.split("=", 1) for a in sys.argv[1:])
CKPT = args.get("ckpt", "runs/best4ep/model.pt")
PORT = int(args.get("port", 8080))
UI = os.path.join(os.path.dirname(__file__), "..", "serve", "mikupad.html")
DEFAULT_N_PREDICT = 256                 # mikupad sends -1 for "until stop"; cap it

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token                     # 50256: training put it before every document, so it's our BOS too
ckpt = torch.load(CKPT, map_location="cpu")
mc = ModelConfig(**ckpt["mc"])
model = GPT(mc)
model.load_state_dict(ckpt["model"])
model = model.cuda().eval()
gpu = threading.Lock()                  # one generation at a time on the GPU
print(f"loaded {CKPT}: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, context {mc.seq_len}")


def piece(tok: int) -> str:
    return enc.decode_single_token_bytes(tok).decode("utf-8", errors="replace")


@torch.inference_mode()
def next_logits(ids: list[int]) -> torch.Tensor:
    x = torch.tensor([ids[-mc.seq_len:]], device="cuda")       # model only knows 2048 positions
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(x)[0, -1].float()


def sample(logits, ids, p):
    """llama.cpp-style sampler chain. Returns (token, distribution actually sampled from)."""
    logits = logits.clone()
    bias = p.get("logit_bias") or []
    for tok, b in (bias.items() if isinstance(bias, dict) else bias):
        logits[int(tok)] = -float("inf") if b is False else logits[int(tok)] + float(b)
    if p.get("ignore_eos", False):                              # never end the document
        logits[EOT] = -float("inf")
    rp, last_n = float(p.get("repeat_penalty", 1.0)), int(p.get("repeat_last_n", 64))
    if rp != 1.0 and last_n != 0:
        recent = torch.tensor(sorted(set(ids[-last_n:] if last_n > 0 else ids)), device=logits.device)
        vals = logits[recent]
        logits[recent] = torch.where(vals > 0, vals / rp, vals * rp)
    temp = float(p.get("temperature", 0.8))
    if temp <= 0:                                               # greedy
        probs = torch.zeros_like(logits)
        probs[logits.argmax()] = 1.0
        return int(logits.argmax()), probs
    probs = F.softmax(logits / temp, dim=-1)
    k = int(p.get("top_k", 40))
    if 0 < k < probs.numel():
        kth = probs.topk(k).values[-1]
        probs = torch.where(probs >= kth, probs, 0.0)
    top_p = float(p.get("top_p", 0.95))
    if top_p < 1.0:
        sp, si = probs.sort(descending=True)
        drop = (sp.cumsum(0) - sp) / sp.sum() > top_p             # keep the smallest prefix reaching top_p
        probs[si[drop]] = 0.0
    min_p = float(p.get("min_p", 0.05))
    if min_p > 0:
        probs = torch.where(probs >= min_p * probs.max(), probs, 0.0)
    probs = probs / probs.sum()
    gen = p["_gen"]
    return int(torch.multinomial(probs, 1, generator=gen)), probs


def top_n(probs, n, post, chosen):
    vals, idx = probs.topk(n)
    if chosen not in idx.tolist():                              # always show the picked token's own probability
        idx = torch.cat([idx, idx.new_tensor([chosen])])
        vals = torch.cat([vals, probs[chosen].view(1)])
    if post:   # llama.cpp post_sampling_probs=true format
        return {"top_probs": [{"id": int(i), "token": piece(int(i)), "prob": float(v)} for v, i in zip(vals, idx) if v > 0]}
    return {"probs": [{"tok_str": piece(int(i)), "prob": float(v)} for v, i in zip(vals, idx)]}


def generate(p):
    prompt = p.get("prompt", "")
    ids = [EOT] + (enc.encode_ordinary(prompt) if isinstance(prompt, str) else [int(t) for t in prompt])
    n_predict = int(p.get("n_predict", -1))
    n_predict = DEFAULT_N_PREDICT if n_predict < 0 else n_predict
    n_probs, post = int(p.get("n_probs", 0)), bool(p.get("post_sampling_probs", False))
    stops = p.get("stop") or []
    stops = [stops] if isinstance(stops, str) else stops
    seed = int(p.get("seed", -1))
    p["_gen"] = torch.Generator(device="cuda")
    p["_gen"].manual_seed(seed if seed >= 0 else torch.seed())
    text = ""
    for _ in range(n_predict):
        with gpu:
            logits = next_logits(ids)
            tok, sampled = sample(logits, ids, p)
            raw = F.softmax(logits, dim=-1)                     # the model's own belief, temperature 1
        if tok == EOT and not p.get("ignore_eos", False):
            yield {"content": "", "stop": True, "stopped_eos": True}
            return
        ids.append(tok)
        s = piece(tok)
        text += s
        chunk = {"content": s, "stop": False}
        if n_probs > 0:
            chunk["completion_probabilities"] = [{"id": tok, "content": s, "token": s,
                                                  **top_n(sampled if post else raw, n_probs, post, tok)}]
        yield chunk
        if any(st and text.endswith(st) for st in stops):
            yield {"content": "", "stop": True, "stopped_word": True}
            return
    yield {"content": "", "stop": True, "stopped_limit": True}


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok", "ckpt": CKPT, "params": sum(p.numel() for p in model.parameters()), "n_ctx": mc.seq_len}


@app.get("/mikupad")
def ui():
    # mikupad defaults its Server to http://127.0.0.1:8080, i.e. the *browser's* machine. Point new
    # sessions at whatever host served this page instead. The vendored file itself stays unmodified.
    html = open(UI, encoding="utf-8").read()
    default = "endpoint: 'http://127.0.0.1:8080'"
    assert default in html, "mikupad default endpoint moved; update this patch"
    return HTMLResponse(html.replace(default, "endpoint: window.location.origin", 1))


@app.post("/tokenize")
async def tokenize(req: Request):
    return {"tokens": enc.encode_ordinary((await req.json()).get("content", ""))}


@app.post("/detokenize")
async def detokenize(req: Request):
    return {"content": enc.decode([int(t) for t in (await req.json()).get("tokens", [])])}


@app.post("/completion")
async def completion(req: Request):
    p = await req.json()
    if p.get("stream"):
        return StreamingResponse((f"data: {json.dumps(c)}\n\n" for c in generate(p)), media_type="text/event-stream")
    chunks = [c for c in generate(p) if c["content"]]
    return JSONResponse({"content": "".join(c["content"] for c in chunks), "stop": True,
                         "completion_probabilities": [c["completion_probabilities"][0] for c in chunks
                                                      if "completion_probabilities" in c]})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
