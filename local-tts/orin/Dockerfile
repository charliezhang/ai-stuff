# Qwen3-TTS on Jetson AGX Orin (JetPack 7.2, CUDA 13, sm_87) with CUDA-graph decoding.
#
# Base: the vLLM image, for its working aarch64 CUDA PyTorch stack (torch 2.13 + torchaudio, cu130,
# transformers 5.x). Python packages are installed with --no-deps and their other deps listed
# explicitly, so pip never replaces torch/torchaudio with a CPU-only or wrong-arch build.
# transformers/hub/tokenizers are pinned to faster-qwen3-tts' tested set (demo/constraints.txt):
# the vLLM image's transformers 5.17 moved Mimi's rope_theta and qwen-tts-hf crashes on it.
#
# faster-qwen3-tts (https://github.com/andimarafioti/faster-qwen3-tts) wraps qwen-tts-hf (the
# transformers-5 port of Qwen's qwen-tts) and captures each decode step in a CUDA graph. Plain
# qwen-tts is launch-bound on Orin's CPU: RTF ~4.8 (5x slower than real time) with the GPU ~12% busy.
FROM vllm/vllm-openai:latest

ARG FQT_VERSION=0.5.3
ARG FQT_COMMIT=7a729d6

RUN apt-get update && apt-get install -y --no-install-recommends sox libsox-fmt-all git \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --no-deps faster-qwen3-tts==${FQT_VERSION} "qwen-tts-hf>=0.1.1.post1,<0.2" nano-parakeet \
 && pip install --no-cache-dir gradio librosa soundfile sox onnxruntime einops python-multipart \
      "huggingface-hub[oauth]==1.28.0" transformers==5.15.1 tokenizers==0.22.2 \
      opencc-python-reimplemented \
 && python3 -c "import torch, torchaudio, faster_qwen3_tts, qwen_tts; assert torch.version.cuda, 'CPU torch!'; print('torch', torch.__version__, torch.version.cuda, 'faster_qwen3_tts ok')"

# Eval checkers: FunASR for Paraformer-zh (Chinese-first ASR that keeps English words; Whisper tends to
# translate code-switched speech). --no-deps again; umap_learn (numba, speaker clustering only) is skipped.
RUN pip install --no-cache-dir --no-deps funasr==1.4.16 torch_complex \
 && pip install --no-cache-dir hydra-core omegaconf kaldiio jieba jamo jaconv sentencepiece tiktoken \
      rapidfuzz tensorboardX modelscope oss2 websockets \
 && python3 -c "import torch, funasr; assert torch.version.cuda; print('funasr', funasr.__version__, 'ok')"

# Demo UI (streaming, TTFA/RTF metrics) + OpenAI-compatible server live in the repo, not the wheel.
RUN git clone https://github.com/andimarafioti/faster-qwen3-tts /opt/fqt && git -C /opt/fqt checkout ${FQT_COMMIT}

ENV HF_HOME=/root/.cache/huggingface \
    DEMO_AVAILABLE_BACKENDS=torch \
    DEMO_DEFAULT_BACKEND=torch
WORKDIR /app
COPY bench_tts.py /app/
ENTRYPOINT []
CMD ["bash"]
