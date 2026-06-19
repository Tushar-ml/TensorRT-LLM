# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Profile baseline vs DTM across inference batch sizes."""

import argparse
import gc
import statistics
import time
from pathlib import Path

import torch

from run import read_config
from run_dtm import _prepare_mel, _run_once
from tokenizer import get_tokenizer

from tensorrt_llm.runtime import ModelRunnerCpp


def _read_engine_max_batch(engine_dir):
    cfg = read_config('decoder', Path(engine_dir))
    return cfg.get('max_batch_size', 1)


def _make_runner(engine_dir, max_batch_size, max_output_len, kv_fraction=0.35):
    return ModelRunnerCpp.from_dir(
        engine_dir,
        is_enc_dec=True,
        max_batch_size=max_batch_size,
        max_input_len=3000,
        max_output_len=max_output_len,
        max_beam_width=1,
        kv_cache_enable_block_reuse=True,
        kv_cache_free_gpu_memory_fraction=kv_fraction,
        cross_kv_cache_fraction=0.5,
        is_orchestrator_mode=False,
        rank=0,
        device_ids=[0],
    )


def _bench_baseline(runner, prefix_ids, mel, mel_lens, end_id, max_output_len,
                    batch_size, warmup=1, iters=5):
    batch_ids = [torch.tensor(prefix_ids, dtype=torch.int32)] * batch_size
    if isinstance(mel, torch.Tensor):
        batch_mel = mel.repeat(batch_size, 1, 1)
        batch_lens = mel_lens.repeat(batch_size)
    else:
        batch_mel = mel
        batch_lens = mel_lens
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


def _bench_dtm(args, draft_len, prefix, mel, mel_lens, duration, end_id,
               warmup=1, iters=5):
    draft_len_cfg, draft_devices, target_devices, _ = eval(
        args.draft_target_model_config)
    del draft_len_cfg
    for _ in range(warmup):
        _run_once(args, draft_len, draft_devices, target_devices, prefix, mel,
                  mel_lens, duration, end_id)
        torch.cuda.synchronize()
    times = []
    last_stats = {}
    last_profile = {}
    for i in range(iters):
        args.profile = (i == iters - 1)
        _, stats = _run_once(args, draft_len, draft_devices, target_devices,
                             prefix, mel, mel_lens, duration, end_id)
        torch.cuda.synchronize()
        times.append(stats['elapsed'])
        last_stats = stats
        last_profile = stats.get('profile') or {}
    return statistics.mean(times), last_stats, last_profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--assets_dir', default='assets')
    parser.add_argument('--draft_engine_dir', required=True)
    parser.add_argument('--target_engine_dir', required=True)
    parser.add_argument('--baseline_engine_dir', required=True)
    parser.add_argument('--batch_sizes', default='1,2,4,8')
    parser.add_argument('--draft_target_model_config', default='[16,[0],[0],False]')
    parser.add_argument('--draft_mode', default='hybrid')
    parser.add_argument('--draft_backend', default='py')
    parser.add_argument('--ngram_max_matching_size', type=int, default=4)
    parser.add_argument('--max_output_len', type=int, default=96)
    parser.add_argument('--text_prefix',
                        default='<|startoftranscript|><|en|><|transcribe|><|notimestamps|>')
    parser.add_argument('--dtype', default='float16')
    parser.add_argument('--draft_kv_cache_free_gpu_memory_fraction', type=float,
                        default=0.18)
    parser.add_argument('--target_kv_cache_free_gpu_memory_fraction', type=float,
                        default=0.28)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=0.0)
    parser.add_argument('--kv_cache_enable_block_reuse', action='store_true',
                        default=True)
    args = parser.parse_args()
    args.profile = False

    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    enc_cfg = read_config('encoder', Path(args.target_engine_dir))
    tok = get_tokenizer('multilingual',
                        num_languages=enc_cfg['num_languages'],
                        tokenizer_dir=args.assets_dir)
    end_id = tok.encode('<|endoftext|>',
                        allowed_special=tok.special_tokens_set)[0]
    prefix_ids = tok.encode(args.text_prefix,
                            allowed_special=tok.special_tokens_set)
    prefix = [torch.tensor(prefix_ids, dtype=torch.int32)]
    mel, mel_lens, duration = _prepare_mel(args.input_file, enc_cfg['n_mels'],
                                             args.dtype, args.assets_dir, 'max')

    turbo_max = _read_engine_max_batch(args.draft_engine_dir)
    v3_max = _read_engine_max_batch(args.baseline_engine_dir)
    dtm_max = _read_engine_max_batch(args.target_engine_dir)

    print(f'Audio: {args.input_file} ({duration:.2f}s)')
    print(f'Engine max_batch_size: turbo={turbo_max} v3={v3_max} dtm_target={dtm_max}')
    print()

    rows = []
    dtm_row = None
    for bs in batch_sizes:
        if bs > v3_max:
            print(f'skip bs={bs}: exceeds large-v3 engine max_batch_size={v3_max}')
            continue
        runner = _make_runner(args.baseline_engine_dir, max(bs, v3_max),
                              args.max_output_len)
        v3_total, v3_per = _bench_baseline(runner, prefix_ids, mel, mel_lens,
                                           end_id, args.max_output_len, bs)
        del runner
        gc.collect()
        torch.cuda.empty_cache()

        turbo_total, turbo_per = None, None
        if bs <= turbo_max:
            runner = _make_runner(args.draft_engine_dir, max(bs, turbo_max),
                                  args.max_output_len, kv_fraction=0.18)
            turbo_total, turbo_per = _bench_baseline(
                runner, prefix_ids, mel, mel_lens, end_id, args.max_output_len,
                bs)
            del runner
            gc.collect()
            torch.cuda.empty_cache()

        rows.append((bs, turbo_total, turbo_per, v3_total, v3_per))

    if dtm_max >= 1:
        dtm_total, dtm_stats, profile = _bench_dtm(
            args, 16, prefix, mel, mel_lens, duration, end_id)
        dtm_row = (dtm_total, dtm_stats, profile)

    print(f"{'bs':>4} {'turbo tot':>10} {'turbo/seq':>10} {'v3 tot':>10} "
          f"{'v3/seq':>10} {'v3 vs turbo':>12} {'draft overhead':>14}")
    print('-' * 78)
    for bs, turbo_total, turbo_per, v3_total, v3_per in rows:
        turbo_tot_s = f'{turbo_total:.3f}s' if turbo_total is not None else 'n/a'
        turbo_seq_s = f'{turbo_per:.3f}s' if turbo_per is not None else 'n/a'
        ratio = f'{turbo_per / v3_per:.2f}x' if turbo_per is not None else 'n/a'
        draft_oh = 'n/a (batched)' if bs > 1 else ''
        if bs == 1 and dtm_row is not None:
            dtm_total = dtm_row[0]
            draft_oh = f'+{(dtm_total - v3_total) * 1000:.0f}ms ({dtm_total / v3_total:.2f}x)'
        print(f'{bs:4d} {turbo_tot_s:>10} {turbo_seq_s:>10} {v3_total:10.3f}s '
              f'{v3_per:10.3f}s {ratio:>12} {draft_oh:>14}')

    if dtm_row is not None:
        dtm_total, dtm_stats, profile = dtm_row
        v3_bs1 = next(v3_per for bs, _, _, _, v3_per in rows if bs == 1)
        print()
        print('DTM (batch_size=1 only; target engine max_batch_size=1):')
        print(f'  wall time: {dtm_total:.3f}s  vs large-v3 bs1: {v3_bs1:.3f}s  '
              f'({dtm_total / v3_bs1:.2f}x)')
        print(f'  iterations: {dtm_stats["iterations"]}  '
              f'acceptance: {dtm_stats["acceptance_rate"] * 100:.1f}%')
        if profile:
            draft_avg = profile.get('draft_ms_avg', 0)
            target_avg = profile.get('target_ms_avg', 0)
            draft_tot = profile.get('draft_ms_total', 0)
            target_tot = profile.get('target_ms_total', 0)
            loop_tot = draft_tot + target_tot
            print(f'  profile (ms/iter): draft={draft_avg:.1f}  target={target_avg:.1f}')
            print(f'  profile (ms total): draft={draft_tot:.1f}  target={target_tot:.1f}  '
                  f'sum={loop_tot:.1f}  unaccounted≈{(dtm_total * 1000 - loop_tot):.0f}ms')
            print(f'  draft share of loop: {100 * draft_tot / loop_tot:.1f}%  '
                  f'target share: {100 * target_tot / loop_tot:.1f}%')
            saved_vs_v3 = (v3_bs1 - dtm_total) * 1000
            print(f'  vs no-draft v3: saves {saved_vs_v3:.0f}ms wall  '
                  f'but draft adds {draft_tot:.0f}ms vs turbo-only path')


if __name__ == '__main__':
    main()
