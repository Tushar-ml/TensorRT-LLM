#!/usr/bin/env bash
# Rebuild DTM target decoder with configurable max_draft_len (default 32).
set -euo pipefail

TARGET_CKPT="${TARGET_CKPT:-whisper_large_v3_weights}"
MAX_DRAFT_LEN="${MAX_DRAFT_LEN:-16}"
MAX_BATCH="${MAX_BATCH:-8}"
TARGET_ENGINE="${TARGET_ENGINE:-whisper_large_v3_dtm_bs8_engine}"
# Decoder context must fit prompt prefix + draft tokens in one spec-dec verify step.
MAX_DECODER_INPUT_LEN="${MAX_DECODER_INPUT_LEN:-$((14 + MAX_DRAFT_LEN))}"

trtllm-build --checkpoint_dir "${TARGET_CKPT}/encoder" \
  --output_dir "${TARGET_ENGINE}/encoder" \
  --moe_plugin disable --max_batch_size "${MAX_BATCH}" --gemm_plugin disable \
  --bert_attention_plugin float16 --max_input_len 3000 --max_seq_len 3000 \
  --remove_input_padding enable --kv_cache_type disabled

trtllm-build --checkpoint_dir "${TARGET_CKPT}/decoder" \
  --output_dir "${TARGET_ENGINE}/decoder" \
  --moe_plugin disable --max_batch_size "${MAX_BATCH}" --max_beam_width 1 \
  --max_seq_len $((114 + MAX_DRAFT_LEN)) --max_input_len "${MAX_DECODER_INPUT_LEN}" --max_encoder_input_len 3000 \
  --speculative_decoding_mode draft_tokens_external --max_draft_len "${MAX_DRAFT_LEN}" \
  --gemm_plugin float16 --bert_attention_plugin float16 --gpt_attention_plugin float16 \
  --remove_input_padding enable --kv_cache_type paged --use_paged_context_fmha enable

echo "Built DTM target at ${TARGET_ENGINE} (max_batch=${MAX_BATCH}, max_draft_len=${MAX_DRAFT_LEN})"
