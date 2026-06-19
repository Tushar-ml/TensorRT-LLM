# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark turbo, large-v3 baseline, and optimized DTM."""

import argparse
import statistics
import time
from pathlib import Path

import torch

from run import read_config
from run_dtm import _prepare_mel, _run_once
from tokenizer import get_tokenizer

from tensorrt_llm.runtime import ModelRunnerCpp


def _bench_cpp(engine_dir, input_file, assets_dir, text_prefix, max_output_len):
    enc_cfg = read_config('encoder', Path(engine_dir))
    tok = get_tokenizer('multilingual',
                        num_languages=enc_cfg['num_languages'],
                        tokenizer_dir=assets_dir)
    end_id = tok.encode('<|endoftext|>',
                        allowed_special=tok.special_tokens_set)[0]
    pid = tok.encode(text_prefix, allowed_special=tok.special_tokens_set)
    mel, mel_lens, duration = _prepare_mel(input_file, enc_cfg['n_mels'],
                                           'float16', assets_dir, 'max')
    common = dict(is_enc_dec=True,
                  max_batch_size=1,
                  max_input_len=3000,
                  max_output_len=max_output_len,
                  max_beam_width=1,
                  kv_cache_enable_block_reuse=True,
                  kv_cache_free_gpu_memory_fraction=0.35,
                  cross_kv_cache_fraction=0.5,
                  is_orchestrator_mode=False,
                  rank=0)
    runner = ModelRunnerCpp.from_dir(engine_dir, device_ids=[0], **common)
    gen = dict(encoder_input_features=mel,
               encoder_output_lengths=(mel_lens // 2).tolist(),
               end_id=end_id,
               pad_id=end_id,
               temperature=0.0,
               top_k=1,
               top_p=0.0,
               num_beams=1,
               streaming=False,
               output_sequence_lengths=True,
               return_dict=True)
    runner.generate(batch_input_ids=[torch.tensor(pid, dtype=torch.int32)],
                    max_new_tokens=max_output_len,
                    **gen)
    times = []
    for _ in range(5):
        t0 = time.time()
        runner.generate(batch_input_ids=[torch.tensor(pid, dtype=torch.int32)],
                        max_new_tokens=max_output_len,
                        **gen)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    return statistics.mean(times), statistics.pstdev(times), duration


def _bench_dtm(args, duration, end_id, prefix, mel, mel_lens):
    draft_len, draft_devices, target_devices, _ = eval(
        args.draft_target_model_config)

    _run_once(args, draft_len, draft_devices, target_devices, prefix, mel,
              mel_lens, duration, end_id)
    times = []
    last_stats = {}
    last_profile = {}
    for i in range(5):
        args.profile = (i == 4)
        _, stats = _run_once(args, draft_len, draft_devices, target_devices,
                             prefix, mel, mel_lens, duration, end_id)
        torch.cuda.synchronize()
        times.append(stats['elapsed'])
        last_stats = stats
        last_profile = stats.get('profile') or {}
    return statistics.mean(times), statistics.pstdev(times), duration, last_stats, last_profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--assets_dir', default='assets')
    parser.add_argument('--draft_engine_dir', required=True)
    parser.add_argument('--target_engine_dir', required=True)
    parser.add_argument('--baseline_engine_dir', required=True)
    parser.add_argument('--draft_target_model_config', default='[16,[0],[0],False]')
    parser.add_argument('--draft_mode', default='hybrid')
    parser.add_argument('--draft_backend', default='py')
    parser.add_argument('--ngram_max_matching_size', type=int, default=4)
    parser.add_argument('--kv_cache_enable_block_reuse', action='store_true', default=True)
    parser.add_argument('--draft_kv_cache_free_gpu_memory_fraction', type=float, default=0.18)
    parser.add_argument('--target_kv_cache_free_gpu_memory_fraction', type=float, default=0.28)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=0.0)
    parser.add_argument('--max_output_len', type=int, default=96)
    parser.add_argument('--text_prefix',
                        default='<|startoftranscript|><|en|><|transcribe|><|notimestamps|>')
    parser.add_argument('--dtype', default='float16')
    parser.add_argument('--profile', action='store_true', default=False)
    args = parser.parse_args()
    args.profile = False

    enc_cfg = read_config('encoder', Path(args.target_engine_dir))
    tok = get_tokenizer('multilingual',
                        num_languages=enc_cfg['num_languages'],
                        tokenizer_dir=args.assets_dir)
    end_id = tok.encode('<|endoftext|>',
                        allowed_special=tok.special_tokens_set)[0]
    pid = tok.encode(args.text_prefix, allowed_special=tok.special_tokens_set)
    prefix = [torch.tensor(pid, dtype=torch.int32)]
    mel, mel_lens, duration = _prepare_mel(args.input_file, enc_cfg['n_mels'],
                                             args.dtype, args.assets_dir, 'max')

    turbo_mean, turbo_std, _ = _bench_cpp(args.draft_engine_dir,
                                            args.input_file, args.assets_dir,
                                            args.text_prefix,
                                            args.max_output_len)
    v3_mean, v3_std, _ = _bench_cpp(args.baseline_engine_dir, args.input_file,
                                    args.assets_dir, args.text_prefix,
                                    args.max_output_len)
    dtm_mean, dtm_std, _, dtm_stats, profile = _bench_dtm(
        args, duration, end_id, prefix, mel, mel_lens)

    print(f'Audio duration: {duration:.2f}s')
    print(f"{'Mode':<22} {'Time':>12} {'RTF':>10} {'vs turbo':>10}")
    print('-' * 58)
    for label, mean, std in [
        ('Turbo baseline', turbo_mean, turbo_std),
        ('Large-v3 baseline', v3_mean, v3_std),
        ('Optimized DTM', dtm_mean, dtm_std),
    ]:
        print(f'{label:<22} {mean:6.3f}s±{std:.3f} {mean / duration:8.4f} '
              f'{turbo_mean / mean:9.2f}x')

    print('-' * 58)
    print(f"DTM iterations: {dtm_stats['iterations']}")
    print(f"DTM acceptance: {dtm_stats['acceptance_rate'] * 100:.1f}%")
    print(f"DTM ngram hit rate: {dtm_stats.get('ngram_hit_rate', 0) * 100:.1f}%")
    if profile:
        print(f"DTM draft avg ms/iter: {profile.get('draft_ms_avg', 0):.2f}")
        print(f"DTM target avg ms/iter: {profile.get('target_ms_avg', 0):.2f}")
    if dtm_mean >= v3_mean:
        print('WARNING: optimized DTM did not beat large-v3 baseline wall time')


if __name__ == '__main__':
    main()
