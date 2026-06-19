# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re

import pytest
from defs.common import convert_weights, venv_check_call
from defs.conftest import get_sm_version, skip_post_blackwell
from defs.trt_test_alternative import check_call

if get_sm_version() >= 103:
    pytest.skip(
        "TRT workflow tests are not supported on post Blackwell-Ultra architecture",
        allow_module_level=True)


def _build_whisper_enc_dec_engine(converted_weight_dir,
                                  output_dir,
                                  component,
                                  extra_decoder_flags=None,
                                  llm_venv=None):
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={converted_weight_dir}/{component}",
        f"--output_dir={output_dir}/{component}",
        "--moe_plugin=disable",
        "--max_batch_size=1",
        "--bert_attention_plugin=float16",
    ]
    if component == "encoder":
        build_cmd.extend([
            "--gemm_plugin=disable",
            "--max_input_len=3000",
            "--max_seq_len=3000",
            "--remove_input_padding=enable",
            "--kv_cache_type=disabled",
        ])
    else:
        build_cmd.extend([
            "--gemm_plugin=float16",
            "--gpt_attention_plugin=float16",
            "--max_input_len=14",
            "--max_seq_len=114",
            "--max_encoder_input_len=3000",
            "--max_beam_width=1",
            "--remove_input_padding=enable",
            "--kv_cache_type=paged",
            "--use_paged_context_fmha=enable",
        ])
        if extra_decoder_flags:
            build_cmd.extend(extra_decoder_flags)
    check_call(build_cmd, env=llm_venv._new_env if llm_venv else None)


@skip_post_blackwell
@pytest.mark.parametrize("draft_model_name", ["large-v3-turbo", "large-v3"])
@pytest.mark.parametrize("whisper_model_root", ["large-v3"], indirect=True)
def test_whisper_dtm(llm_venv, engine_dir, whisper_example_root,
                     whisper_model_root, whisper_example_audio_file,
                     draft_model_name):
    """Build and run Whisper draft-target speculative decoding."""
    target_model_name, target_ckpt_dir = whisper_model_root
    draft_ckpt_parent = os.path.join(
        os.path.dirname(target_ckpt_dir.rstrip('/')),
        f"whisper-{draft_model_name}")
    if not os.path.isdir(draft_ckpt_parent):
        pytest.skip(f"Draft checkpoint {draft_ckpt_parent} not available")

    target_engine_dir = os.path.join(
        engine_dir, f"{target_model_name}_dtm_target_float16")
    draft_engine_dir = os.path.join(
        engine_dir, f"{draft_model_name}_dtm_draft_float16")

    target_converted = convert_weights(
        llm_venv=llm_venv,
        example_root=whisper_example_root,
        cmodel_dir=target_engine_dir,
        model=target_model_name,
        model_path=target_ckpt_dir,
        use_weight_only=False,
        weight_only_precision=None,
    )
    draft_converted = os.path.join(draft_engine_dir, draft_model_name, "float16")
    venv_check_call(
        llm_venv, [
            f"{whisper_example_root}/convert_checkpoint.py",
            "--model_dir", draft_ckpt_parent,
            "--model_name", draft_model_name,
            "--output_dir", draft_converted,
        ])

    for component in ["encoder", "decoder"]:
        _build_whisper_enc_dec_engine(target_converted, target_engine_dir,
                                      component,
                                      extra_decoder_flags=[
                                          "--speculative_decoding_mode=draft_tokens_external",
                                          "--max_draft_len=4",
                                      ] if component == "decoder" else None,
                                      llm_venv=llm_venv)
        _build_whisper_enc_dec_engine(draft_converted, draft_engine_dir,
                                    component,
                                    llm_venv=llm_venv)

    baseline_cmd = [
        f"{whisper_example_root}/run.py",
        f"--input_file={whisper_example_audio_file}",
        f"--engine_dir={target_engine_dir}",
        f"--assets_dir={target_ckpt_dir}",
        f"--num_beams=1",
        f"--batch_size=1",
        "--log_level=info",
    ]
    venv_check_call(llm_venv, baseline_cmd)

    dtm_cmd = [
        f"{whisper_example_root}/run_dtm.py",
        f"--draft_engine_dir={draft_engine_dir}",
        f"--target_engine_dir={target_engine_dir}",
        f"--input_file={whisper_example_audio_file}",
        f"--assets_dir={target_ckpt_dir}",
        '--draft_target_model_config=[4,[0],[0],False]',
        '--draft_mode=turbo',
        '--draft_kv_cache_free_gpu_memory_fraction=0.18',
        '--target_kv_cache_free_gpu_memory_fraction=0.28',
        "--log_level=info",
    ]
    output = venv_check_call(llm_venv, dtm_cmd)

    assert "Acceptance rate:" in output
    match = re.search(r"Acceptance rate:\s*([\d.]+)%", output)
    assert match is not None
    acceptance = float(match.group(1))
    rtf_match = re.search(r"RTF:\s*([\d.]+)", output)
    assert rtf_match is not None
    rtf = float(rtf_match.group(1))
    assert rtf < 1.0, f"DTM RTF smoke check failed: {rtf}"
    if draft_model_name == target_model_name:
        assert acceptance > 0.0
    else:
        assert acceptance >= 0.0
