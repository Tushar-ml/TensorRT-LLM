#!/usr/bin/env bash
# Build optimized Whisper draft engine variants for DTM experiments.
set -euo pipefail

DRAFT_CKPT="${DRAFT_CKPT:-whisper_turbo_weights}"
DRAFT_MODEL="${DRAFT_MODEL:-large-v3-turbo}"
DRAFT_ENGINE="${DRAFT_ENGINE:-whisper_turbo_engine}"
ASSETS="${ASSETS:-assets}"
MAX_BATCH="${MAX_BATCH:-8}"
PRECISION="${PRECISION:-float16}"

python3 convert_checkpoint.py \
  --model_name "${DRAFT_MODEL}" \
  --model_dir "${ASSETS}" \
  --output_dir "${DRAFT_CKPT}"

trtllm-build --checkpoint_dir "${DRAFT_CKPT}/encoder" \
  --output_dir "${DRAFT_ENGINE}/encoder" \
  --moe_plugin disable --max_batch_size "${MAX_BATCH}" --gemm_plugin disable \
  --bert_attention_plugin float16 --max_input_len 3000 --max_seq_len 3000 \
  --remove_input_padding enable --kv_cache_type disabled

DECODER_FLAGS=(--moe_plugin disable --max_batch_size "${MAX_BATCH}" --max_beam_width 1 \
  --max_seq_len 114 --max_input_len 14 --max_encoder_input_len 3000 \
  --gemm_plugin "${PRECISION}" --bert_attention_plugin float16 \
  --gpt_attention_plugin float16 --remove_input_padding enable \
  --kv_cache_type paged --use_paged_context_fmha enable)

trtllm-build --checkpoint_dir "${DRAFT_CKPT}/decoder" \
  --output_dir "${DRAFT_ENGINE}/decoder" "${DECODER_FLAGS[@]}"

echo "Built draft engine at ${DRAFT_ENGINE} (model=${DRAFT_MODEL}, max_batch=${MAX_BATCH}, precision=${PRECISION})"
echo "Optional: PRECISION=float8 for FP8 draft GEMMs; DRAFT_MODEL=distil-large-v3 for alternate draft."
