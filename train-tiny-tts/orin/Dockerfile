# tiny-tts:orin — pocket-tts training on the Jetson AGX Orin (sm_87), pocket-tts code mounted at /work.
# Built by build_image.sh, which supplies the sphn wheel in the build context.
# Base: the vLLM image, for the one aarch64 CUDA PyTorch known to work on this box (torch 2.13, cu130;
# same base as local-tts and train-tiny-llm). It already has pydantic, sentencepiece, tokenizers,
# safetensors, typer, einops, huggingface_hub, pyyaml, tqdm and triton. Added: sphn (built from
# source, no aarch64 wheel on PyPI), soundfile and scipy, and jiwer + rapidfuzz (training.train
# imports training/eval/librispeech.py through refl.py even when reward fine-tuning is off).
# --no-deps so pip never swaps torch for a CPU or x86 build.
FROM vllm/vllm-openai:latest
COPY sphn-*.whl /tmp/
RUN pip install --no-cache-dir --no-deps /tmp/sphn-*.whl "soundfile==0.14.0" "scipy>=1.13" \
      "jiwer>=3.0" "rapidfuzz>=3" \
 && rm /tmp/sphn-*.whl \
 && python3 -c "import torch, sphn, soundfile, scipy.signal, jiwer; assert torch.version.cuda, 'CPU torch!'; print('torch', torch.__version__, torch.version.cuda)"
ENV PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONPATH=/work/pocket-tts
WORKDIR /work/pocket-tts
ENTRYPOINT []
