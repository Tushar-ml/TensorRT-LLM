# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Whisper draft-target speculative decoding (DTM) example."""

import argparse
import ast
import re
import time
from pathlib import Path

import torch

from tokenizer import get_tokenizer
from whisper_dtm_utils import (CppDraftRunner, IterationProfiler,
                               NgramDraftPool, PersistentWhisperDraftDecoder)
from whisper_utils import log_mel_spectrogram

import tensorrt_llm
import tensorrt_llm.logger as logger
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.runtime import ModelRunnerCpp

from run import (WhisperEncoding, WhisperTRTLLM, decode_wav_file, read_config)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--draft_engine_dir',
                        type=str,
                        required=True,
                        help='TRT engine dir for draft model (encoder + turbo decoder).')
    parser.add_argument('--target_engine_dir',
                        type=str,
                        required=True,
                        help='TRT engine dir for target model (encoder + large-v3 decoder '
                        'built with draft_tokens_external).')
    parser.add_argument('--draft_target_model_config',
                        type=str,
                        required=True,
                        help='DTM config: [draft_len, draft_device_list, target_device_list, '
                        'use_logits]. Example: "[16,[0],[0],False]"')
    parser.add_argument('--draft_len',
                        type=int,
                        default=None,
                        help='Override draft_len from draft_target_model_config (engine max_draft_len must be >= this value).')
    parser.add_argument('--draft_mode',
                        type=str,
                        default='hybrid',
                        choices=['turbo', 'ngram', 'hybrid'],
                        help='Draft source: turbo neural, ngram pool, or hybrid (default).')
    parser.add_argument('--draft_backend',
                        type=str,
                        default='py',
                        choices=['cpp', 'py'],
                        help='Draft runtime: py encoder-once session (default) or cpp ModelRunnerCpp.')
    parser.add_argument('--ngram_max_matching_size',
                        type=int,
                        default=4,
                        help='Max ngram pattern size when draft_mode is ngram/hybrid.')
    parser.add_argument('--profile',
                        action='store_true',
                        help='Log per-iteration draft/target latency breakdown.')
    parser.add_argument('--sweep_draft_len',
                        action='store_true',
                        help='Sweep draft_len in {4,6,8,10,12,16,24,32,48} and print benchmark table.')
    parser.add_argument('--assets_dir', type=str, default='./assets')
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--padding_strategy',
                        default='max',
                        help='Mel padding: max (pad to 30s, default), longest, or nopad.')
    parser.add_argument('--baseline_engine_dir',
                        type=str,
                        default=None,
                        help='Optional plain large-v3 engine dir for quality validation.')
    parser.add_argument('--max_output_len', type=int, default=96)
    parser.add_argument('--text_prefix',
                        default='<|startoftranscript|><|en|><|transcribe|><|notimestamps|>',
                        help='Decoder prefix tokens shared by draft and target.')
    parser.add_argument('--kv_cache_enable_block_reuse',
                        action='store_true',
                        default=True)
    parser.add_argument('--kv_cache_free_gpu_memory_fraction',
                        type=float,
                        default=None,
                        help='Shared KV fraction when draft/target fractions unset.')
    parser.add_argument('--draft_kv_cache_free_gpu_memory_fraction',
                        type=float,
                        default=0.18)
    parser.add_argument('--target_kv_cache_free_gpu_memory_fraction',
                        type=float,
                        default=0.28)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=0.0)
    parser.add_argument('--dtype', type=str, default='float16')
    return parser.parse_args()


def _prepare_mel(input_file, n_mels, dtype, assets_dir, padding_strategy='max'):
    mel, duration = log_mel_spectrogram(input_file,
                                        n_mels,
                                        device='cuda',
                                        return_duration=True,
                                        mel_filters_dir=assets_dir)
    mel = mel.type(str_dtype_to_torch(dtype)).unsqueeze(0)
    if padding_strategy != 'longest':
        mel = torch.nn.functional.pad(mel, (0, 3000 - mel.shape[2]))
    if mel.shape[2] % 2:
        mel = torch.nn.functional.pad(mel, (0, 1))
    mel_input_lengths = torch.tensor([mel.shape[2]],
                                     dtype=torch.int32,
                                     device='cuda')
    mel = mel.transpose(1, 2)
    return mel, mel_input_lengths, duration


def _count_accepted(d_ids, t_ids):
    accept_len = min(len(d_ids[0]), len(t_ids))
    accepted = 0
    for i in range(accept_len):
        if d_ids[0][i] == t_ids[i]:
            accepted += 1
        else:
            break
    return accepted


def run_whisper_dtm(prefix,
                    target_runner,
                    draft_decoder,
                    ngram_pool,
                    draft_mode,
                    draft_len,
                    end_id,
                    max_output_len,
                    mel,
                    mel_input_lengths,
                    target_encoder_output=None,
                    target_encoder_output_lengths=None,
                    temperature=0.0,
                    top_k=1,
                    top_p=0.0,
                    profile=False):
    input_len = len(prefix[0])
    max_seq_len = input_len + max_output_len
    prefix = [p.clone() for p in prefix]

    output_ids = torch.full((max_seq_len, ), end_id, dtype=torch.int32)
    output_ids[:input_len] = prefix[0]
    sequence_length = input_len

    n_draft_tokens = 0
    n_accept_tokens = 0
    n_ngram_hits = 0
    n_iteration = 0
    profiler = IterationProfiler(enabled=profile)

    if target_encoder_output is not None:
        target_gen_common = dict(
            encoder_outputs=[target_encoder_output],
            encoder_output_lengths=target_encoder_output_lengths,
            end_id=end_id,
            pad_id=end_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_beams=1,
            streaming=False,
            output_sequence_lengths=True,
            return_dict=True,
        )
    else:
        target_gen_common = dict(
            encoder_input_features=mel,
            encoder_output_lengths=(mel_input_lengths // 2).tolist(),
            end_id=end_id,
            pad_id=end_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_beams=1,
            streaming=False,
            output_sequence_lengths=True,
            return_dict=True,
        )

    while True:
        n_iteration += 1
        prefix_len = len(prefix[0])
        if prefix_len >= max_seq_len - 1:
            break

        draft_t0 = time.time()
        d_ids = [[end_id]]
        d_len = 0
        used_ngram = False
        if draft_mode in ('ngram', 'hybrid') and ngram_pool is not None:
            d_ids = ngram_pool.get_draft_tokens(prefix)
            d_len = len(d_ids[0])
            used_ngram = d_len > 0 and d_ids[0][0] != end_id
            if used_ngram:
                n_ngram_hits += 1
        if draft_mode == 'ngram' or (draft_mode == 'hybrid'
                                     and not used_ngram):
            if draft_decoder is not None:
                d_ids, d_len = draft_decoder.propose(prefix, end_id, draft_len)
        elif draft_mode == 'turbo' and draft_decoder is not None:
            d_ids, d_len = draft_decoder.propose(prefix, end_id, draft_len)
        draft_ms = (time.time() - draft_t0) * 1000.0

        target_t0 = time.time()
        target = target_runner.generate(
            batch_input_ids=prefix,
            draft_tokens_list=d_ids,
            draft_logits_list=None,
            max_new_tokens=draft_len + 1,
            **target_gen_common,
        )
        torch.cuda.synchronize()
        target_ms = (time.time() - target_t0) * 1000.0

        t_seq_len = target['sequence_lengths'][0, 0].item()
        t_seq_len = min(t_seq_len, max_seq_len)
        t_ids = target['output_ids'][0, 0, prefix_len:t_seq_len].tolist()
        accepted = _count_accepted(d_ids, t_ids)

        output_ids[prefix_len:t_seq_len] = torch.tensor(t_ids,
                                                          dtype=torch.int32)
        sequence_length = t_seq_len
        n_draft_tokens += max(d_len, 0)
        n_accept_tokens += accepted
        profiler.record(n_iteration, prefix_len, draft_ms, target_ms, accepted,
                        d_len)

        if accepted < d_len and draft_decoder is not None:
            draft_decoder.on_rejection()
        elif draft_decoder is not None:
            draft_decoder.commit_prefix(t_seq_len)

        if t_seq_len <= prefix_len:
            break
        if end_id in target['output_ids'][0, 0, prefix_len:t_seq_len]:
            break
        prefix = [target['output_ids'][0, 0, :t_seq_len].clone()]

    stats = {
        'iterations': n_iteration,
        'draft_tokens': n_draft_tokens,
        'accepted_tokens': n_accept_tokens,
        'acceptance_rate':
        (n_accept_tokens / n_draft_tokens if n_draft_tokens > 0 else 0.0),
        'ngram_hits': n_ngram_hits,
        'ngram_hit_rate':
        (n_ngram_hits / n_iteration if n_iteration > 0 else 0.0),
        'profile': profiler.summary(),
    }
    if profile:
        profiler.print_summary()
    return output_ids[:sequence_length], stats


def _build_runners(args, draft_device_list, target_device_list, runtime_rank,
                   mel, mel_input_lengths):
    shared_fraction = getattr(args, 'kv_cache_free_gpu_memory_fraction', None)
    draft_fraction = shared_fraction if shared_fraction is not None else args.draft_kv_cache_free_gpu_memory_fraction
    target_fraction = shared_fraction if shared_fraction is not None else args.target_kv_cache_free_gpu_memory_fraction

    common_runner_kwargs = dict(
        is_enc_dec=True,
        max_batch_size=1,
        max_input_len=3000,
        max_output_len=args.max_output_len,
        max_beam_width=1,
        kv_cache_enable_block_reuse=args.kv_cache_enable_block_reuse,
        cross_kv_cache_fraction=0.5,
        is_orchestrator_mode=False,
        rank=runtime_rank,
    )

    draft_decoder = None
    if args.draft_mode in ('turbo', 'hybrid'):
        if args.draft_backend == 'py':
            runtime_mapping = tensorrt_llm.Mapping(1, runtime_rank)
            draft_decoder = PersistentWhisperDraftDecoder(
                Path(args.draft_engine_dir), runtime_mapping)
        else:
            draft_runner = ModelRunnerCpp.from_dir(
                engine_dir=args.draft_engine_dir,
                device_ids=draft_device_list,
                kv_cache_free_gpu_memory_fraction=draft_fraction,
                **common_runner_kwargs,
            )
            draft_decoder = CppDraftRunner(draft_runner,
                                           mel,
                                           mel_input_lengths,
                                           args.temperature,
                                           args.top_k,
                                           args.top_p,
                                           draft_engine_dir=Path(
                                               args.draft_engine_dir))

    target_runner = ModelRunnerCpp.from_dir(
        engine_dir=args.target_engine_dir,
        device_ids=target_device_list,
        kv_cache_free_gpu_memory_fraction=target_fraction,
        **common_runner_kwargs,
    )
    return draft_decoder, target_runner


def _run_once(args, draft_len, draft_device_list, target_device_list, prefix,
              mel, mel_input_lengths, duration, end_id):
    runtime_rank = tensorrt_llm.mpi_rank()
    draft_decoder, target_runner = _build_runners(
        args, draft_device_list, target_device_list, runtime_rank, mel,
        mel_input_lengths)
    if isinstance(draft_decoder, PersistentWhisperDraftDecoder):
        draft_decoder.encode_once(mel.transpose(1, 2), mel_input_lengths)

    ngram_pool = None
    if args.draft_mode in ('ngram', 'hybrid'):
        ngram_pool = NgramDraftPool(draft_len, args.ngram_max_matching_size,
                                    end_id)

    target_encoder_output = None
    target_encoder_output_lengths = None
    target_encoder = WhisperEncoding(Path(args.target_engine_dir))
    target_encoder_output, target_encoder_output_lengths = target_encoder.get_audio_features(
        mel.transpose(1, 2), mel_input_lengths)

    start = time.time()
    output_ids, stats = run_whisper_dtm(
        prefix,
        target_runner,
        draft_decoder,
        ngram_pool,
        args.draft_mode,
        draft_len,
        end_id,
        args.max_output_len,
        mel,
        mel_input_lengths,
        target_encoder_output=target_encoder_output,
        target_encoder_output_lengths=target_encoder_output_lengths.tolist(),
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        profile=args.profile,
    )
    elapsed = time.time() - start
    stats['elapsed'] = elapsed
    stats['rtf'] = elapsed / duration
    return output_ids, stats


def main():
    args = parse_arguments()
    tensorrt_llm.logger.set_level(args.log_level)

    config_draft_len, draft_device_list, target_device_list, use_logits = ast.literal_eval(
        args.draft_target_model_config)
    draft_len = args.draft_len if args.draft_len is not None else config_draft_len
    assert not use_logits, 'Whisper DTM example supports token acceptance only.'
    assert draft_len > 0, 'draft_len must be > 0'

    target_engine_dir = Path(args.target_engine_dir)
    encoder_config = read_config('encoder', target_engine_dir)

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(1, runtime_rank)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    is_multilingual = read_config('decoder',
                                  target_engine_dir)['vocab_size'] >= 51865
    tokenizer_name = 'multilingual' if is_multilingual else 'gpt2'
    tokenizer = get_tokenizer(name=tokenizer_name,
                              num_languages=encoder_config['num_languages'],
                              tokenizer_dir=args.assets_dir)
    end_id = tokenizer.encode('<|endoftext|>',
                              allowed_special=tokenizer.special_tokens_set)[0]
    prompt_id = tokenizer.encode(
        args.text_prefix, allowed_special=tokenizer.special_tokens_set)
    prefix = [torch.tensor(prompt_id, dtype=torch.int32)]

    mel, mel_input_lengths, duration = _prepare_mel(args.input_file,
                                                    encoder_config['n_mels'],
                                                    args.dtype,
                                                    args.assets_dir,
                                                    args.padding_strategy)

    if args.baseline_engine_dir:
        logger.info('Running plain large-v3 baseline from %s',
                    args.baseline_engine_dir)
        baseline_model = WhisperTRTLLM(Path(args.baseline_engine_dir),
                                       assets_dir=args.assets_dir,
                                       batch_size=1,
                                       num_beams=1,
                                       use_py_session=True)
        baseline_results, _ = decode_wav_file(
            args.input_file,
            baseline_model,
            text_prefix=args.text_prefix,
            dtype=args.dtype,
            mel_filters_dir=args.assets_dir,
            padding_strategy=args.padding_strategy)
        baseline_text = ' '.join(baseline_results[0][2])
        print(f'Baseline (large-v3 greedy): {baseline_text}')

    if args.sweep_draft_len:
        print(f'Sweeping draft_len on {args.input_file} ({duration:.2f}s audio)')
        print(f"{'draft_len':>9} {'iters':>6} {'accept%':>8} {'time':>8} {'RTF':>8}")
        for sweep_len in [4, 6, 8, 10, 12, 16, 24, 32, 48]:
            output_ids, stats = _run_once(args, sweep_len, draft_device_list,
                                          target_device_list, prefix, mel,
                                          mel_input_lengths, duration, end_id)
            print(f'{sweep_len:9d} {stats["iterations"]:6d} '
                  f'{stats["acceptance_rate"] * 100:7.1f}% '
                  f'{stats["elapsed"]:7.3f}s {stats["rtf"]:7.4f}')
        return

    logger.info('Draft len: %s', draft_len)
    logger.info('Draft mode: %s', args.draft_mode)
    logger.info('Draft devices: %s', draft_device_list)
    logger.info('Target devices: %s', target_device_list)

    output_ids, stats = _run_once(args, draft_len, draft_device_list,
                                  target_device_list, prefix, mel,
                                  mel_input_lengths, duration, end_id)

    text = tokenizer.decode(output_ids.tolist()).strip()
    text = re.sub(r'<\|.*?\|>', '', text)
    print(f'Transcript: {text}')
    print(f'Iterations: {stats["iterations"]}')
    print(f'Acceptance rate: {stats["acceptance_rate"] * 100:.2f}% '
          f'({stats["accepted_tokens"]}/{stats["draft_tokens"]} draft tokens)')
    if stats.get('ngram_hits', 0) > 0 or args.draft_mode in ('ngram', 'hybrid'):
        print(f'Ngram hit rate: {stats["ngram_hit_rate"] * 100:.2f}% '
              f'({stats["ngram_hits"]}/{stats["iterations"]} iterations)')
    print(f'RTF: {stats["rtf"]:.4f}')


if __name__ == '__main__':
    main()
