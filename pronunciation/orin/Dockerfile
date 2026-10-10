# Pronunciation assessment demo on Jetson AGX Orin.
# Reuses qwen-tts:orin (vLLM base: working aarch64 CUDA torch 2.13, transformers 5.15, gradio, librosa).
#
# eSpeak-NG is built from source with --with-extdict-cmn: Ubuntu's package lacks the extended Mandarin
# dictionary, so it spells unknown characters as pinyin and reads tone digits as English words
# ("xi3" -> "z ɪ θ ɹ iː"). The phoneme model was trained on eSpeak output *with* that dictionary.
FROM qwen-tts:orin

ARG ESPEAK_VERSION=1.52.0
RUN apt-get update && apt-get install -y --no-install-recommends \
      git make autoconf automake libtool pkg-config gcc g++ \
 && git clone --depth 1 --branch ${ESPEAK_VERSION} https://github.com/espeak-ng/espeak-ng /tmp/espeak-ng \
 && cd /tmp/espeak-ng && ./autogen.sh && ./configure --prefix=/usr/local --with-extdict-cmn \
 && make -j"$(nproc)" && make install && ldconfig && cd / && rm -rf /tmp/espeak-ng \
 && rm -rf /var/lib/apt/lists/*
ENV PHONEMIZER_ESPEAK_LIBRARY=/usr/local/lib/libespeak-ng.so

# phonemizer (G2P wrapper) and pypinyin; --no-deps so pip never touches torch.
RUN pip install --no-cache-dir --no-deps phonemizer segments dlinfo csvw clldutils colorlog isodate \
      rfc3986 uritemplate language-tags pypinyin \
 && python3 -c "from phonemizer.backend import EspeakBackend as E; print(E('cmn').phonemize(['我喜欢吃苹果'])); import torch; assert torch.version.cuda"
# ESPnet for POWSM-CTC (espnet/powsm_ctc). --no-deps again; flash_attn can't build on Jetson and ESPnet
# falls back to its own attention ("Failed to import Flash Attention, using ESPnet default").
RUN pip install --no-cache-dir --no-deps espnet==202610.post2 espnet_model_zoo typeguard humanfriendly omniio \
 && python3 -c "from espnet2.bin.s2t_inference_ctc import Speech2Text; import torch; assert torch.version.cuda; print('espnet ok')"

WORKDIR /app
ENTRYPOINT []
CMD ["bash"]
