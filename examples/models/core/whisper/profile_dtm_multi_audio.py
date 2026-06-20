# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark large-v3 vs DTM (MTP) on batches of different audio files."""

import argparse
import gc
import statistics
import time
from pathlib import Path

import torch

from run import read_config
from profile_dtm_batch import (_make_runner, _pick_dtm_target, _read_engine_max_batch)
from run_dtm import (DTMSession, _prepare_batch_mels, _resolve_batched_target_verify)
from tokenizer import get_tokenizer


def _cycle_files(files, batch_size):
    if batch_size <= len(files):
        return files[:batch_size]
    out = []
    for i in range(batch_size):
        out.append(files[i % len(files)])
    return out


def _bench_v3(runner, prefix_ids, batch_mel, batch_lens, end_id, max_output_len,
              batch_size, warmup=1, iters=3):
    batch_ids = [torch.tensor(prefix_ids, dtype=torch.int32)] * batch_size
    gen = dict(
        batch_input_ids=batch_ids,
        encoder_input_features=batch_mel,
        encoder_output_lengths=(batch_lens // 2).tolist(),
        end_id=end_id,
        pad_id=end_id,
        temperature=0.0,
        top_k=1,
        top_p=0.0,
        num_beams=1,
        streaming=False,
        output_sequence_lengths=True,
        return_dict=True,
    )
    for _ in range(warmup):
        runner.generate(max_new_tokens=max_output_len, **gen)
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.time()
        runner.generate(max_new_tokens=max_output_len, **gen)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    mean = statistics.mean(times)
    return mean, mean / batch_size


def _bench_dtm_batched(args, draft_len, prefix, batch_mel, batch_lens,
                       durations, end_id, batch_size, warmup=1, iters=3):
    """Run true batched DTM on different audios (grouped verify by prefix length)."""
    target_dir = _pick_dtm_target(args, batch_size)
    if target_dir is None:
        return None

    saved_target = args.target_engine_dir
    args.target_engine_dir = target_dir
    draft_len_cfg, draft_devices, target_devices, _ = eval(
        args.draft_target_model_config)
    del draft_len_cfg
    batch_prefix = [prefix[0].clone() for _ in range(batch_size)]
    try:
        session = DTMSession(args,
                             draft_devices,
                             target_devices,
                             batch_prefix,
                             batch_mel,
                             batch_lens,
                             batch_size=batch_size,
                             identical_batch=False)
        session.durations = durations
        session.end_id = end_id
        session.setup_ngram_pool(draft_len, end_id)
        for _ in range(warmup):
            session.run(draft_len, end_id, profile=False)
            torch.cuda.synchronize()
        times = []
        last_stats = {}
        for _ in range(iters):
            _, stats = session.run(draft_len, end_id, profile=False)
            torch.cuda.synchronize()
            times.append(stats['elapsed'])
            last_stats = stats
        mean = statistics.mean(times)
        return {
            'total_s': mean,
            'per_seq_s': mean / batch_size,
            'accept_pct': last_stats.get('acceptance_rate', 0) * 100,
        }
    finally:
        args.target_engine_dir = saved_target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files',
                        required=True,
                        help='Comma-separated audio paths (cycled to fill batch).')
    parser.add_argument('--assets_dir', default='assets')
    parser.add_argument('--draft_engine_dir', required=True)
    parser.add_argument('--target_engine_dir', required=True)
    parser.add_argument('--target_engine_dir_bs8',
                        default='whisper_large_v3_dtm_bs8_engine')
    parser.add_argument('--baseline_engine_dir', required=True)
    parser.add_argument('--batch_sizes', default='1,2,4')
    parser.add_argument('--draft_target_model_config', default='[16,[0],[0],False]')
    parser.add_argument('--draft_mode', default='hybrid')
    parser.add_argument('--draft_backend', default='cpp',
                        choices=['cpp', 'py'],
                        help='cpp IFB draft recommended for multi-audio batches.')
    parser.add_argument('--ngram_max_matching_size', type=int, default=4)
    parser.add_argument('--max_output_len', type=int, default=96)
    parser.add_argument('--text_prefix',
                        default='<|startoftranscript|><|en|><|transcribe|><|notimestamps|>')
    parser.add_argument('--dtype', default='float16')
    parser.add_argument('--draft_kv_cache_free_gpu_memory_fraction', type=float,
                        default=0.08)
    parser.add_argument('--target_kv_cache_free_gpu_memory_fraction', type=float,
                        default=0.10)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=0.0)
    parser.add_argument('--kv_cache_enable_block_reuse', action='store_true',
                        default=True)
    args = parser.parse_args()
    args.profile = False

    input_files = [x.strip() for x in args.input_files.split(',') if x.strip()]
    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]

    enc_cfg = read_config('encoder', Path(args.baseline_engine_dir))
    tok = get_tokenizer('multilingual',
                        num_languages=enc_cfg['num_languages'],
                        tokenizer_dir=args.assets_dir)
    end_id = tok.encode('<|endoftext|>',
                        allowed_special=tok.special_tokens_set)[0]
    prefix_ids = tok.encode(args.text_prefix,
                            allowed_special=tok.special_tokens_set)
    prefix = [torch.tensor(prefix_ids, dtype=torch.int32)]

    v3_max = _read_engine_max_batch(args.baseline_engine_dir)
    dtm_max = _read_engine_max_batch(args.target_engine_dir)
    dtm_bs8_max = (_read_engine_max_batch(args.target_engine_dir_bs8)
                   if Path(args.target_engine_dir_bs8).exists() else 0)

    print('Multi-audio benchmark: batched large-v3 vs batched DTM/MTP')
    print('  accept%% is aggregate draft-token acceptance (short clips <10s run fewer '
          'verify steps with draft_len=16).')
    print(f'Audio pool ({len(input_files)} files):')
    for f in input_files:
        print(f'  - {f}')
    print(f'Engine max_batch_size: v3={v3_max} dtm={dtm_max} dtm_bs8={dtm_bs8_max}')
    print(f'DTM batched verify: {_resolve_batched_target_verify(args, max(batch_sizes))}')
    print()

    print(f"{'bs':>4} {'files':>5} {'v3/seq':>10} {'MTP/seq':>10} {'MTP tot':>10} "
          f"{'accept%':>8} {'MTP vs v3':>10}")
    print('-' * 72)

    for bs in batch_sizes:
        files = _cycle_files(input_files, bs)
        batch_mel, batch_lens, durations = _prepare_batch_mels(
            files, enc_cfg['n_mels'], args.dtype, args.assets_dir)

        if bs <= v3_max:
            runner = _make_runner(args.baseline_engine_dir, max(bs, v3_max),
                                  args.max_output_len)
            v3_total, v3_per = _bench_v3(runner, prefix_ids, batch_mel,
                                           batch_lens, end_id,
                                           args.max_output_len, bs)
            del runner
            gc.collect()
            torch.cuda.empty_cache()
        else:
            v3_total = v3_per = None

        dtm = _bench_dtm_batched(args, 16, prefix, batch_mel, batch_lens,
                                 durations, end_id, bs)
        gc.collect()
        torch.cuda.empty_cache()

        if v3_per is None:
            print(f'{bs:4d} {len(files):5d} {"n/a":>10} ', end='')
        else:
            print(f'{bs:4d} {len(files):5d} {v3_per:10.3f}s ', end='')

        if dtm is None:
            print(f'{"n/a":>10} {"n/a":>10} {"n/a":>8} {"n/a":>10}')
            continue

        vs = dtm['per_seq_s'] / v3_per if v3_per else float('nan')
        print(f'{dtm["per_seq_s"]:10.3f}s {dtm["total_s"]:10.3f}s '
              f'{dtm["accept_pct"]:7.1f}% {vs:9.2f}x')


if __name__ == '__main__':
    main()
