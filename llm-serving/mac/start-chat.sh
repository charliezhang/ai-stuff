#!/usr/bin/env bash
# Open WebUI on this Mac, talking to vLLM on the Jetson (OpenAI-compatible API).
# First run downloads ~2-3 GB of Python deps into uv's cache; later starts take ~20 s.
set -euo pipefail
cd "$(dirname "$0")"
export DATA_DIR="$PWD/open-webui-data"          # chats, settings (gitignored)
export OPENAI_API_BASE_URL="${ORIN_URL:-http://orin.local:8000/v1}"
export OPENAI_API_KEY="none"                     # vLLM runs without auth
export ENABLE_OLLAMA_API=False
export WEBUI_AUTH=False                          # single local user, no login
export ENABLE_OPENAI_API=True
export BYPASS_EMBEDDING_AND_RETRIEVAL=True       # no RAG -> skip loading embedding models (saves RAM)
export ANONYMIZED_TELEMETRY=False DO_NOT_TRACK=True SCARF_NO_ANALYTICS=True
mkdir -p "$DATA_DIR"
echo "Open WebUI -> http://localhost:${PORT:-8080}  (backend: $OPENAI_API_BASE_URL)"
exec uvx --python 3.11 open-webui@latest serve --port "${PORT:-8080}"
