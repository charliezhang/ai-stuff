#!/usr/bin/env bash
# Build tiny-tts:orin on the Orin. sphn (pocket-tts' audio reader) publishes no aarch64 wheel, so
# the first run compiles one in a throwaway vLLM container (~1 min) and keeps it in
# ~/train-tiny-tts/wheels. It must be the real sphn: other MP3 decoders (libsndfile/mpg123) start
# most HiFiTTS-2 chapters 1105 samples later, which would misalign every word timing.
set -euo pipefail
ROOT=~/train-tiny-tts
WHEELS=$ROOT/wheels
mkdir -p "$WHEELS"
if ! ls "$WHEELS"/sphn-*-linux_aarch64.whl >/dev/null 2>&1; then
  echo "building the sphn wheel (Rust, ~1 min)"
  # CMAKE_POLICY_VERSION_MINIMUM: the Opus sources bundled by audiopus_sys predate CMake 4.
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e CMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -v "$WHEELS":/out --entrypoint bash vllm/vllm-openai:latest -c '
      set -e
      curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal -q >/dev/null
      . /tmp/.cargo/env
      pip wheel --no-deps --no-cache-dir -q sphn==0.2.1 -w /out'
fi
ctx=$(mktemp -d)
trap 'rm -rf "$ctx"' EXIT
cp "$ROOT/orin/Dockerfile" "$ctx/"
cp "$WHEELS"/sphn-*-linux_aarch64.whl "$ctx/"
docker build -q -t tiny-tts:orin "$ctx" >/dev/null
echo "built tiny-tts:orin"
