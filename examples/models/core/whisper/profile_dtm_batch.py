# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Profile turbo, large-v3 baseline, and DTM across inference batch sizes."""

import argparse
import gc
import json
import statistics
import subprocess
import sys
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


def _pick_dtm_target(args, batch_size):
    if batch_size <= _read_engine_max_batch(args.target_engine_dir):
        return args.target_engine_dir
    if args.target_engine_dir_bs8 and batch_size <= _read_engine_max_batch(
            args.target_engine_dir_bs8):
        return args.target_engine_dir_bs8
    return None


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
                    batch_size, warmup=1, iters=3):
    batch_ids = [torch.tensor(prefix_ids, dtype=torch.int32)] * batch_size
    batch_mel = mel.repeat(batch_size, 1, 1)
    batch_lens = mel_lens.repeat(batch_size)
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
               batch_size, warmup=1, iters=3):
    target_dir = _pick_dtm_target(args, batch_size)
    if target_dir is None:
        return None

    saved_target = args.target_engine_dir
    args.target_engine_dir = target_dir
    draft_len_cfg, draft_devices, target_devices, _ = eval(
        args.draft_target_model_config)
    del draft_len_cfg
    try:
        for _ in range(warmup):
            _run_once(args, draft_len, draft_devices, target_devices, prefix,
                      mel, mel_lens, duration, end_id, batch_size=batch_size)
            torch.cuda.synchronize()
        times = []
        last_stats = {}
        last_profile = {}
        for i in range(iters):
            args.profile = (i == iters - 1)
            _, stats = _run_once(args, draft_len, draft_devices, target_devices,
                                 prefix, mel, mel_lens, duration, end_id,
                                 batch_size=batch_size)
            torch.cuda.synchronize()
            times.append(stats['elapsed'])
            last_stats = stats
            last_profile = stats.get('profile') or {}
        mean = statistics.mean(times)
        return {
            'total_s': mean,
            'per_seq_s': mean / batch_size,
            'accept_pct': last_stats.get('acceptance_rate', 0) * 100,
            'draft_ms_avg': last_profile.get('draft_ms_avg', 0),
            'target_ms_avg': last_profile.get('target_ms_avg', 0),
            'draft_ms_total': last_profile.get('draft_ms_total', 0),
            'target_ms_total': last_profile.get('target_ms_total', 0),
            'iterations': last_stats.get('iterations', 0),
        }
    finally:
        args.target_engine_dir = saved_target


def _run_worker(args):
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

    bs = args.worker_batch_size
    row = {'batch_size': bs}

    turbo_max = _read_engine_max_batch(args.draft_engine_dir)
    v3_max = _read_engine_max_batch(args.baseline_engine_dir)

    if bs <= v3_max:
        runner = _make_runner(args.baseline_engine_dir, max(bs, v3_max),
                              args.max_output_len)
        v3_total, v3_per = _bench_baseline(runner, prefix_ids, mel, mel_lens,
                                           end_id, args.max_output_len, bs)
        row['v3_total_s'] = v3_total
        row['v3_per_seq_s'] = v3_per
        del runner
        gc.collect()
        torch.cuda.empty_cache()
    else:
        row['error'] = f'bs>{v3_max} for v3'

    if bs <= turbo_max:
        runner = _make_runner(args.draft_engine_dir, max(bs, turbo_max),
                              args.max_output_len, kv_fraction=0.18)
        turbo_total, turbo_per = _bench_baseline(runner, prefix_ids, mel,
                                                 mel_lens, end_id,
                                                 args.max_output_len, bs)
        row['turbo_total_s'] = turbo_total
        row['turbo_per_seq_s'] = turbo_per
        del runner
        gc.collect()
        torch.cuda.empty_cache()

    dtm = _bench_dtm(args, 16, prefix, mel, mel_lens, duration, end_id, bs)
    if dtm is None:
        row['dtm_error'] = f'no engine for bs={bs}'
    else:
        row.update({f'dtm_{k}': v for k, v in dtm.items()})

    print(json.dumps(row))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--assets_dir', default='assets')
    parser.add_argument('--draft_engine_dir', required=True)
    parser.add_argument('--target_engine_dir', required=True)
    parser.add_argument(
        '--target_engine_dir_bs8',
        default='whisper_large_v3_dtm_bs8_engine',
        help='DTM target engine with max_batch_size>=8.')
    parser.add_argument('--baseline_engine_dir', required=True)
    parser.add_argument('--batch_sizes', default='1,2,4,8')
    parser.add_argument('--worker_batch_size', type=int, default=None)
    parser.add_argument('--draft_target_model_config', default='[16,[0],[0],False]')
    parser.add_argument('--draft_mode', default='hybrid')
    parser.add_argument('--draft_backend', default='py')
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

    if args.worker_batch_size is not None:
        _run_worker(args)
        return

    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    turbo_max = _read_engine_max_batch(args.draft_engine_dir)
    v3_max = _read_engine_max_batch(args.baseline_engine_dir)
    dtm_max = _read_engine_max_batch(args.target_engine_dir)
    dtm_bs8_max = (_read_engine_max_batch(args.target_engine_dir_bs8)
                   if Path(args.target_engine_dir_bs8).exists() else 0)

    print(f'Audio: {args.input_file}')
    print(f'Engine max_batch_size: turbo={turbo_max} v3={v3_max} '
          f'dtm={dtm_max} dtm_bs8={dtm_bs8_max}')
    print('(Identical clip repeated per batch slot; DTM bs>1 uses prefix sync '
          'for benchmark — batched enc-dec spec-dec verify may reduce acceptance.)')
    print()

    rows = []
    script = str(Path(__file__).resolve())
    base_cmd = [
        sys.executable,
        script,
        '--input_file',
        args.input_file,
        '--assets_dir',
        args.assets_dir,
        '--draft_engine_dir',
        args.draft_engine_dir,
        '--target_engine_dir',
        args.target_engine_dir,
        '--target_engine_dir_bs8',
        args.target_engine_dir_bs8,
        '--baseline_engine_dir',
        args.baseline_engine_dir,
        '--draft_target_model_config',
        args.draft_target_model_config,
        '--draft_mode',
        args.draft_mode,
        '--draft_backend',
        args.draft_backend,
        '--draft_kv_cache_free_gpu_memory_fraction',
        str(args.draft_kv_cache_free_gpu_memory_fraction),
        '--target_kv_cache_free_gpu_memory_fraction',
        str(args.target_kv_cache_free_gpu_memory_fraction),
    ]
    for bs in batch_sizes:
        cmd = base_cmd + ['--worker_batch_size', str(bs)]
        proc = subprocess.run(cmd,
                              capture_output=True,
                              text=True,
                              cwd=str(Path(__file__).resolve().parent))
        if proc.returncode != 0:
            print(f'worker bs={bs} failed:', proc.stderr[-500:], file=sys.stderr)
            continue
        line = proc.stdout.strip().splitlines()[-1]
        rows.append(json.loads(line))

    print(f"{'bs':>4} {'turbo/seq':>10} {'v3/seq':>10} {'DTM/seq':>10} "
          f"{'DTM tot':>10} {'accept%':>8} {'draft ms':>9} {'target ms':>10} "
          f"{'DTM vs v3':>10}")
    print('-' * 96)
    for row in rows:
        bs = row['batch_size']
        turbo = row.get('turbo_per_seq_s')
        v3 = row.get('v3_per_seq_s')
        dtm_per = row.get('dtm_per_seq_s')
        dtm_tot = row.get('dtm_total_s')
        accept = row.get('dtm_accept_pct')
        draft_ms = row.get('dtm_draft_ms_avg')
        target_ms = row.get('dtm_target_ms_avg')
        turbo_s = f'{turbo:.3f}s' if turbo is not None else 'n/a'
        v3_s = f'{v3:.3f}s' if v3 is not None else 'n/a'
        if dtm_per is None:
            print(f'{bs:4d} {turbo_s:>10} {v3_s:>10} {"n/a":>10} {"n/a":>10} '
                  f'{"n/a":>8} {"n/a":>9} {"n/a":>10} {"n/a":>10}')
            continue
        vs_v3 = dtm_per / v3 if v3 else float('nan')
        print(f'{bs:4d} {turbo_s:>10} {v3_s:>10} {dtm_per:10.3f}s '
              f'{dtm_tot:10.3f}s {accept:7.1f}% {draft_ms:8.1f} '
              f'{target_ms:9.1f} {vs_v3:9.2f}x')

    bs1 = next((r for r in rows if r['batch_size'] == 1), None)
    if bs1 and bs1.get('dtm_draft_ms_total'):
        draft_tot = bs1['dtm_draft_ms_total']
        target_tot = bs1['dtm_target_ms_total']
        loop_tot = draft_tot + target_tot
        print()
        print('DTM bs=1 draft vs target (ms):')
        print(f'  draft {draft_tot:.0f}ms ({100 * draft_tot / loop_tot:.0f}%)  '
              f'target {target_tot:.0f}ms ({100 * target_tot / loop_tot:.0f}%)')
        print(f'  turbo-only {bs1.get("turbo_per_seq_s", 0):.3f}s  '
              f'v3-only {bs1.get("v3_per_seq_s", 0):.3f}s  '
              f'DTM {bs1.get("dtm_per_seq_s", 0):.3f}s')


if __name__ == '__main__':
    main()
