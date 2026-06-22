#!/usr/bin/env bash
# Serve Gemma4 MTP with official Google backbone (text-only from multimodal ckpt).
set -euo pipefail

MODEL="${1:-google/gemma-4-31b-it}"
CONFIG="${2:-/home/ubuntu/config_google_gemma4_text_mtp.yaml}"
PORT="${3:-8000}"
TP="${4:-4}"

export TRTLLM_GEMMA4_TEXT_ONLY=1

exec trtllm-serve "${MODEL}" \
  --backend pytorch \
  --tp_size "${TP}" \
  --config "${CONFIG}" \
  --host 0.0.0.0 \
  --port "${PORT}"
