/*
 * Copyright (c) 2020-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/kernelUtils.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/utils.h"
#include "tensorrt_llm/kernels/multiHeadAttentionCommon.h"
#include <list>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace jit
{

namespace
{

using tensorrt_llm::common::contains;

bool supportConfigCommon(XQAParams const& xqaParams, bool forConfigurePlugin)
{
    if (xqaParams.unidirectional != 1)
    {
        return false;
    }
    if (xqaParams.mask_type != tensorrt_llm::kernels::AttentionMaskType::CAUSAL)
    {
        return false;
    }
    if (xqaParams.cross_attention)
    {
        return false;
    }
    if (xqaParams.position_shift_enabled || xqaParams.sink_token_length > 0)
    {
        return false;
    }
    if (xqaParams.num_kv_heads != 0 && xqaParams.num_q_heads % xqaParams.num_kv_heads != 0)
    {
        return false;
    }
    bool const is_vanilla_mha = !xqaParams.multi_query_tokens
        && (xqaParams.num_kv_heads == 0 || xqaParams.num_q_heads == xqaParams.num_kv_heads);
    if (is_vanilla_mha && xqaParams.beam_width == 1)
    {
        // Do not use XQA kernel for vanilla MHA case for performance reasons.
        return false;
    }
    if (is_vanilla_mha && xqaParams.head_size <= 128)
    {
        // TODO: remove this when the kernel bug for num_kv_heads <= 128 gets fixed.
        return false;
    }
    if (!contains({PositionEmbeddingType::kROPE_GPTJ, PositionEmbeddingType::kROPE_GPT_NEOX,
                      PositionEmbeddingType::kROPE_M, PositionEmbeddingType::kLONG_ROPE,
                      PositionEmbeddingType::kLEARNED_ABSOLUTE, PositionEmbeddingType::kYARN},
            xqaParams.position_embedding_type))
    {
        return false;
    }
    if (xqaParams.chunked_attention_size != INT_MAX)
    {
        // TODO: chunked attention is not supported yet.
        return false;
    }
    return true;
}

} // anonymous namespace

bool appliesRoPEInXqaKernel(XQAParams const& xqaParams, bool isQGMMAKernel)
{
    // In-kernel RoPE is only implemented by the Hopper QGMMA kernel, and only for non-spec-dec, non-MLA
    // cases.
    if (!isQGMMAKernel || xqaParams.multi_query_tokens || xqaParams.isMLA())
    {
        return false;
    }
    // The in-kernel RoPE rotates the first rotary_embedding_dim head elements and copies the rest
    // unrotated; it requires the rope region to be 16B-aligned for any supported cache dtype
    // (rotary_embedding_dim a multiple of 16). Unsupported shapes fall back to invokeQKVPreprocessing.
    bool const isSupportedRotary = xqaParams.rotary_embedding_dim > 0
        && xqaParams.rotary_embedding_dim <= xqaParams.head_size && xqaParams.rotary_embedding_dim % 16 == 0;
    return isSupportedRotary
        && tensorrt_llm::common::contains({PositionEmbeddingType::kLONG_ROPE, PositionEmbeddingType::kROPE_GPT_NEOX,
                                              PositionEmbeddingType::kROPE_GPTJ},
            xqaParams.position_embedding_type);
}

bool supportConfigQGMMA(XQAParams const& xqaParams, int SM, bool forConfigurePlugin)
{
    if (!supportConfigCommon(xqaParams, forConfigurePlugin))
    {
        return false;
    }
    if (SM != kSM_90)
    {
        return false;
    }
    if (!contains({DATA_TYPE_FP16, DATA_TYPE_BF16}, xqaParams.data_type))
    {
        return false;
    }
    if (!contains({DATA_TYPE_FP16, DATA_TYPE_BF16, DATA_TYPE_E4M3}, xqaParams.kv_cache_data_type))
    {
        return false;
    }
    int32_t head_grp_size = xqaParams.num_kv_heads == 0 ? 1 : xqaParams.num_q_heads / xqaParams.num_kv_heads;
    // Gemma4 global-attention layers (head_size=512) in spec-dec: route to the Hopper warp-spec kernel
    // (mha_sm90.cu).  It is compiled with SPEC_Q_SEQ_LEN (SWAP_AB=1, set in nvrtcWrapper) which omits the
    // headElems-scaled VTBuffer, so head_size=512 fits SMEM and works with a bf16/fp16 KV cache.  The
    // SWAP_AB tile requires ctaNbValidQHeads = head_grp_size * q_seq_len <= 32.
    bool const isHd512SpecDec = xqaParams.multi_query_tokens && xqaParams.head_size == 512;
    if (isHd512SpecDec)
    {
        // ctaNbValidQHeads = head_grp_size * specDecQLen must be <= 32 (SWAP_AB tile).  specDecQLen is
        // baked from spec_decoding_max_generation_length (see compileEngine), which is the real
        // draft_len+1; generation_input_length is a placeholder at configure time.
        int32_t const specQLen = xqaParams.spec_decoding_max_generation_length > 0
            ? xqaParams.spec_decoding_max_generation_length
            : xqaParams.generation_input_length;
        if (head_grp_size * specQLen > 32)
        {
            return false;
        }
    }
    bool const is_skip_softmax = xqaParams.skip_softmax_threshold_scale_factor != 0;
    if (!is_skip_softmax && xqaParams.kv_cache_data_type != DATA_TYPE_E4M3 && !isHd512SpecDec)
    {
        // Only use hopper kernel with fp16/bf16 kv cache data type when skip softmax is enabled
        // (or for the hd512 spec-dec path above, which is validated for a bf16/fp16 KV cache).
        return false;
    }
    if (xqaParams.beam_width != 1)
    {
        return false;
    }
    uint32_t const maxHeadSize = isHd512SpecDec ? 512 : 256;
    if (xqaParams.head_size % 16 != 0 || xqaParams.head_size < 16 || xqaParams.head_size > maxHeadSize)
    {
        return false;
    }
    if (head_grp_size * xqaParams.beam_width > 32)
    {
        return false;
    }
    if (xqaParams.paged_kv_cache && !contains({8, 16, 32, 64, 128}, xqaParams.tokens_per_block))
    {
        return false;
    }
    return true;
}

// NOTE on head_size 512 (Gemma4 global attention):
//  - QGMMA (Hopper warp-spec, mha_sm90.cu): supports head_size=512 in the SPEC-DEC path via the
//    SWAP_AB=1 variant (SPEC_Q_SEQ_LEN set in nvrtcWrapper), which omits the headElems-scaled VTBuffer
//    so SMEM fits.  See the isHd512SpecDec branch in supportConfigQGMMA above.  Non-spec-dec hd512 on
//    QGMMA is NOT enabled (the non-swapAB VTBuffer overflows SMEM at 512).
//  - HMMA (Ampere-style, mha.cu): NOT templated for head_size>256 (gemm1WarpsPerGrp = headElems/warpTile.x
//    breaks the warp-group tiling at 512), so the gate below stays at 256.  Plain hd512 decode falls back
//    to MMHA (decoderMaskedMultiheadAttention), which supports head_size=512.  tensorMapUtils.cpp is
//    extended for 512.
bool supportConfigHMMA(XQAParams const& xqaParams, int SM, bool forConfigurePlugin)
{
    if (!supportConfigCommon(xqaParams, forConfigurePlugin))
    {
        return false;
    }
    if (SM < kSM_80)
    {
        return false;
    }
    if (!contains({DATA_TYPE_FP16, DATA_TYPE_BF16}, xqaParams.data_type))
    {
        return false;
    }
    if (!contains({DATA_TYPE_FP16, DATA_TYPE_BF16, DATA_TYPE_INT8, DATA_TYPE_E4M3}, xqaParams.kv_cache_data_type))
    {
        return false;
    }
    if (xqaParams.beam_width != 1 && xqaParams.beam_width != 4)
    {
        return false;
    }
    if (!forConfigurePlugin)
    {
        // Inference time checks.
        if (xqaParams.host_past_key_value_lengths == nullptr)
        {
            return false;
        }
        if (!xqaParams.multi_query_tokens && xqaParams.beam_width != 1
            && xqaParams.max_past_kv_length + 1 > xqaParams.cyclic_attention_window_size)
        {
            return false;
        }
    }
    if (xqaParams.head_size % 16 != 0 || xqaParams.head_size < 16 || xqaParams.head_size > 256)
    {
        return false;
    }
    int32_t head_grp_size = xqaParams.num_kv_heads == 0 ? 1 : xqaParams.num_q_heads / xqaParams.num_kv_heads;
    if (head_grp_size * xqaParams.beam_width > 32)
    {
        return false;
    }
    if (xqaParams.paged_kv_cache && !contains({16, 32, 64, 128}, xqaParams.tokens_per_block))
    {
        return false;
    }
    bool const is_skip_softmax = xqaParams.skip_softmax_threshold_scale_factor != 0;
    if (is_skip_softmax)
    {
        return false;
    }
    return true;
}

bool supportConfigMLA(XQAParams const& xqaParams, int SM, bool forConfigurePlugin)
{
    if (!supportConfigCommon(xqaParams, forConfigurePlugin))
    {
        return false;
    }
    if (SM != kSM_120)
    {
        return false;
    }
    if (xqaParams.data_type != DATA_TYPE_E4M3)
    {
        return false;
    }
    if (xqaParams.kv_cache_data_type != DATA_TYPE_E4M3)
    {
        return false;
    }
    if (xqaParams.beam_width != 1)
    {
        return false;
    }
    if (!xqaParams.isMLA())
    {
        return false;
    }
    if (xqaParams.paged_kv_cache && !contains({8, 16, 32, 64, 128}, xqaParams.tokens_per_block))
    {
        return false;
    }
    bool const is_skip_softmax = xqaParams.skip_softmax_threshold_scale_factor != 0;
    if (is_skip_softmax)
    {
        return false;
    }
    return true;
}

} // namespace jit
} // namespace kernels

TRTLLM_NAMESPACE_END
