#!/usr/bin/env bash
# Qwen3.6-35B-A3B (NVFP4) on vLLM — Jetson AGX Orin 64GB, JetPack 7.2.1
# --enable-mfu-metrics: exports estimated FLOPs/bytes counters used by the Grafana overview.
# --enable-auto-tool-choice/--tool-call-parser: tool calling; Open WebUI sends tool_choice=auto and fails without it.
# VLLM_DISABLED_KERNELS: Orin (sm_87) has no FP8 tensor cores, but vLLM 0.30 picks Cutlass/Torch FP8
# kernels anyway and crashes; disabling them forces Marlin weight-only FP8 (W8A16).
sudo jetson_clocks
sudo docker rm -f vllm 2>/dev/null
sudo docker run -d --name vllm --restart unless-stopped --runtime=nvidia --network host --ipc host \
  -v /mnt/models/hf:/root/.cache/huggingface -v /mnt/models/vllm-cache:/root/.cache/vllm \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel,CutlassFP8ScaledMMLinearKernel,B12xTensorFP8ScaledMMLinearKernel,PerTensorTorchFP8ScaledMMLinearKernel,ChannelWiseTorchFP8ScaledMMLinearKernel \
  vllm/vllm-openai:latest nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --served-model-name qwen3.6-35b \
  --max-model-len 8192 --gpu-memory-utilization 0.7 \
  --reasoning-parser qwen3 \
  --enable-mfu-metrics \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --limit-mm-per-prompt "{\"image\":0,\"video\":0}" "$@"
echo "starting; follow with: sudo docker logs -f vllm   (ready in ~5 min)"
