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
                               NgramDraftPool, PersistentWhisperDraftDecoder,
                               replicate_encoder_for_batch)
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
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Number of identical sequences to run in one DTM batch.')
    parser.add_argument('--sync_identical_batch',
                        action='store_true',
                        help='Debug: force all batch slots to follow slot-0 prefix.')
    parser.add_argument('--sequential_target_verify',
                        action='store_true',
                        help='Verify each batch slot separately (slower; debug).')
    parser.add_argument('--batched_target_verify',
                        action='store_true',
                        help='Force batched target verify even at batch_size=1.')
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


def _group_indices_by_prefix_len(prefix):
    groups = {}
    for bi, p in enumerate(prefix):
        plen = len(p)
        groups.setdefault(plen, []).append(bi)
    return groups


def _subset_target_gen_common(target_gen_common, indices):
    slot_gen = dict(target_gen_common)
    if isinstance(target_gen_common.get('encoder_outputs'), list):
        slot_gen['encoder_outputs'] = [
            target_gen_common['encoder_outputs'][i] for i in indices
        ]
        enc_lens = target_gen_common['encoder_output_lengths']
        if isinstance(enc_lens, list):
            slot_gen['encoder_output_lengths'] = [enc_lens[i] for i in indices]
        else:
            slot_gen['encoder_output_lengths'] = [
                int(enc_lens[i]) for i in indices
            ]
    return slot_gen


def _merge_target_outputs(batch_size, indices, slot_out, output_ids_parts,
                          seq_lens_parts):
    for local_i, bi in enumerate(indices):
        output_ids_parts[bi] = slot_out['output_ids'][local_i:local_i + 1]
        seq_lens_parts[bi] = slot_out['sequence_lengths'][local_i:local_i + 1]


def _batched_target_verify(target_runner, prefix, draft_tokens_list, draft_len,
                           target_gen_common, batched_target_verify):
    batch_size = len(prefix)
    output_ids_parts = [None] * batch_size
    seq_lens_parts = [None] * batch_size
    groups = _group_indices_by_prefix_len(prefix)

    for _plen, indices in groups.items():
        sub_prefix = [prefix[i] for i in indices]
        sub_drafts = [draft_tokens_list[i] for i in indices]
        slot_gen = _subset_target_gen_common(target_gen_common, indices)
        if len(indices) == 1 or not batched_target_verify:
            for bi, p, d in zip(indices, sub_prefix, sub_drafts):
                one_gen = _subset_target_gen_common(target_gen_common, [bi])
                slot_out = target_runner.generate(
                    batch_input_ids=[p],
                    draft_tokens_list=[d],
                    draft_logits_list=None,
                    max_new_tokens=draft_len + 1,
                    **one_gen,
                )
                _merge_target_outputs(batch_size, [bi], slot_out,
                                      output_ids_parts, seq_lens_parts)
        else:
            slot_out = target_runner.generate(
                batch_input_ids=sub_prefix,
                draft_tokens_list=sub_drafts,
                draft_logits_list=None,
                max_new_tokens=draft_len + 1,
                **slot_gen,
            )
            _merge_target_outputs(batch_size, indices, slot_out,
                                  output_ids_parts, seq_lens_parts)

    return {
        'output_ids': torch.cat(output_ids_parts, dim=0),
        'sequence_lengths': torch.cat(seq_lens_parts, dim=0),
    }


def _split_encoder_outputs(encoder_output, batch_size, encoder_output_lengths=None):
    if batch_size == 1:
        return [encoder_output]
    if encoder_output.dim() == 2 and encoder_output_lengths is not None:
        outputs = []
        offset = 0
        for length in encoder_output_lengths:
            length = int(length)
            outputs.append(encoder_output[offset:offset + length])
            offset += length
        return outputs
    return [encoder_output[i] for i in range(batch_size)]


def _resolve_batched_target_verify(args, batch_size):
    if getattr(args, 'sequential_target_verify', False):
        return False
    if getattr(args, 'batched_target_verify', False):
        return True
    return batch_size > 1


def _prepare_batch_mels(input_files, n_mels, dtype, assets_dir,
                        padding_strategy='max'):
    """Load and stack mel features for a multi-audio batch."""
    mels = []
    lengths = []
    durations = []
    for input_file in input_files:
        mel, mel_lens, duration = _prepare_mel(input_file, n_mels, dtype,
                                               assets_dir, padding_strategy)
        mels.append(mel)
        lengths.append(mel_lens)
        durations.append(duration)
    return torch.cat(mels, dim=0), torch.cat(lengths, dim=0), durations


def _encode_target_for_batch(target_engine_dir,
                             mel,
                             mel_input_lengths,
                             batch_size,
                             identical_batch=True):
    """Encode target audio; replicate bs=1 features for identical batch slots."""
    target_encoder = WhisperEncoding(Path(target_engine_dir))
    if identical_batch and batch_size > 1:
        single_mel = mel[:1]
        single_lens = mel_input_lengths[:1]
        encoder_output, encoder_output_lengths = target_encoder.get_audio_features(
            single_mel.transpose(1, 2), single_lens)
        encoder_outputs, encoder_lengths = replicate_encoder_for_batch(
            encoder_output, encoder_output_lengths, batch_size)
        encoder_max_input_length = max(encoder_lengths)
        return encoder_outputs, encoder_lengths, encoder_max_input_length

    encoder_output, encoder_output_lengths = target_encoder.get_audio_features(
        mel.transpose(1, 2), mel_input_lengths)
    if batch_size == 1:
        enc_len = int(encoder_output_lengths[0])
        return [encoder_output], [enc_len], enc_len
    encoder_outputs = _split_encoder_outputs(encoder_output, batch_size,
                                             encoder_output_lengths.tolist())
    encoder_lengths = [int(x) for x in encoder_output_lengths.tolist()]
    return encoder_outputs, encoder_lengths, max(encoder_lengths)


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
                    profile=False,
                    sync_identical_batch=False,
                    batched_target_verify=False,
                    encoder_max_input_length=None,
                    identical_batch=True):
    batch_size = len(prefix)
    input_len = len(prefix[0])
    max_seq_len = input_len + max_output_len
    prefix = [p.clone() for p in prefix]

    n_draft_tokens = 0
    n_accept_tokens = 0
    n_ngram_hits = 0
    n_iteration = 0
    profiler = IterationProfiler(enabled=profile)

    if target_encoder_output is not None:
        encoder_outputs = (
            target_encoder_output if isinstance(target_encoder_output, list)
            else _split_encoder_outputs(target_encoder_output, batch_size,
                                        target_encoder_output_lengths))
        target_gen_common = dict(
            encoder_outputs=encoder_outputs,
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
        if all(len(p) >= max_seq_len - 1 for p in prefix):
            break

        draft_t0 = time.time()
        draft_tokens_list = [[end_id] for _ in range(batch_size)]
        d_len = 0
        turbo_indices = list(range(batch_size))
        if draft_mode in ('ngram', 'hybrid') and ngram_pool is not None:
            for bi in range(batch_size):
                ngram_draft = ngram_pool.get_draft_tokens([prefix[bi]])[0]
                if len(ngram_draft) > 0 and ngram_draft[0] != end_id:
                    n_ngram_hits += 1
                    draft_tokens_list[bi] = ngram_draft[:]
                    d_len = max(d_len, len(ngram_draft))
                    if draft_mode == 'ngram':
                        turbo_indices = []
                    elif bi in turbo_indices:
                        turbo_indices.remove(bi)
        if draft_decoder is not None and draft_mode in ('turbo', 'hybrid'):
            turbo_drafts, td = draft_decoder.propose(
                prefix, end_id, draft_len, identical_batch=identical_batch)
            for bi in range(batch_size):
                if draft_mode == 'turbo' or bi in turbo_indices:
                    draft_tokens_list[bi] = turbo_drafts[bi]
            d_len = max(d_len, td)
        draft_ms = (time.time() - draft_t0) * 1000.0

        target_t0 = time.time()
        target = _batched_target_verify(
            target_runner,
            prefix,
            draft_tokens_list,
            draft_len,
            target_gen_common,
            batched_target_verify=(batch_size == 1 or batched_target_verify),
        )
        torch.cuda.synchronize()
        target_ms = (time.time() - target_t0) * 1000.0

        batch_accepted = 0
        batch_draft_count = 0
        new_prefix = []
        all_done = True
        mean_prefix_len = 0
        for bi in range(batch_size):
            prefix_len = len(prefix[bi])
            mean_prefix_len += prefix_len
            t_seq_len = target['sequence_lengths'][bi, 0].item()
            t_seq_len = min(t_seq_len, input_len + max_output_len)
            t_ids = target['output_ids'][bi, 0, prefix_len:t_seq_len].tolist()
            accepted = _count_accepted([draft_tokens_list[bi]], t_ids)
            batch_accepted += accepted
            batch_draft_count += max(len(draft_tokens_list[bi]), 0)

            if accepted < len(draft_tokens_list[bi]) and draft_decoder is not None:
                draft_decoder.on_rejection(bi)
            elif draft_decoder is not None:
                draft_decoder.commit_prefix(t_seq_len, bi)

            new_prefix.append(
                target['output_ids'][bi, 0, :t_seq_len].clone())
            if t_seq_len <= prefix_len:
                continue
            if end_id in target['output_ids'][bi, 0, prefix_len:t_seq_len]:
                continue
            all_done = False

        prefix = new_prefix
        mean_prefix_len = mean_prefix_len // batch_size
        if sync_identical_batch and batch_size > 1:
            leader = prefix[0].clone()
            prefix = [leader.clone() for _ in range(batch_size)]
        n_draft_tokens += batch_draft_count
        n_accept_tokens += batch_accepted
        profiler.record(n_iteration, mean_prefix_len, draft_ms, target_ms,
                        batch_accepted, batch_draft_count)

        if all_done:
            break

    output_ids = prefix[0]
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
        'batch_size': batch_size,
    }
    if profile:
        profiler.print_summary()
    return output_ids, stats


def _scaled_kv_fractions(batch_size, draft_fraction, target_fraction):
    scale = max(1.0, batch_size / 2.0)
    return draft_fraction / scale, target_fraction / scale


def _build_runners(args, draft_device_list, target_device_list, runtime_rank,
                   mel, mel_input_lengths, batch_size=1, identical_batch=True):
    shared_fraction = getattr(args, 'kv_cache_free_gpu_memory_fraction', None)
    draft_fraction = shared_fraction if shared_fraction is not None else args.draft_kv_cache_free_gpu_memory_fraction
    target_fraction = shared_fraction if shared_fraction is not None else args.target_kv_cache_free_gpu_memory_fraction
    draft_fraction, target_fraction = _scaled_kv_fractions(
        batch_size, draft_fraction, target_fraction)

    use_batched_target = _resolve_batched_target_verify(args, batch_size)
    target_max_batch = batch_size if use_batched_target else 1

    common_runner_kwargs = dict(
        is_enc_dec=True,
        max_batch_size=batch_size,
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
                Path(args.draft_engine_dir), runtime_mapping,
                max_batch_size=batch_size)
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
                                               args.draft_engine_dir),
                                           batch_size=batch_size,
                                           identical_batch=identical_batch)

    target_runner = ModelRunnerCpp.from_dir(
        engine_dir=args.target_engine_dir,
        device_ids=target_device_list,
        kv_cache_free_gpu_memory_fraction=target_fraction,
        is_enc_dec=True,
        max_batch_size=target_max_batch,
        max_input_len=3000,
        max_output_len=args.max_output_len,
        max_beam_width=1,
        kv_cache_enable_block_reuse=args.kv_cache_enable_block_reuse,
        cross_kv_cache_fraction=0.5,
        is_orchestrator_mode=False,
        rank=runtime_rank,
    )
    return draft_decoder, target_runner


class DTMSession:
    """Reusable DTM runners and target encoder cache for repeated runs."""

    def __init__(self, args, draft_device_list, target_device_list, prefix,
                 mel, mel_input_lengths, batch_size=1, identical_batch=True):
        self.args = args
        self.draft_len_cfg = None
        self.draft_device_list = draft_device_list
        self.target_device_list = target_device_list
        self.batch_size = batch_size
        self.identical_batch = identical_batch
        self.duration = None
        self.end_id = None

        if batch_size > 1 and identical_batch:
            mel = mel.repeat(batch_size, 1, 1)
            mel_input_lengths = mel_input_lengths.repeat(batch_size)
            prefix = [prefix[0].clone() for _ in range(batch_size)]
        elif batch_size > 1:
            prefix = [prefix[0].clone() for _ in range(batch_size)]

        self.prefix = prefix
        self.mel = mel
        self.mel_input_lengths = mel_input_lengths

        runtime_rank = tensorrt_llm.mpi_rank()
        self.draft_decoder, self.target_runner = _build_runners(
            args, draft_device_list, target_device_list, runtime_rank, mel,
            mel_input_lengths, batch_size=batch_size,
            identical_batch=self.identical_batch)
        if isinstance(self.draft_decoder, PersistentWhisperDraftDecoder):
            if self.identical_batch:
                self.draft_decoder.encode_once(self.mel[:1].transpose(1, 2),
                                               self.mel_input_lengths[:1],
                                               identical_batch=True)
            else:
                self.draft_decoder.encode_once(self.mel.transpose(1, 2),
                                               self.mel_input_lengths,
                                               identical_batch=False)

        self.ngram_pool = None
        self.encoder_max_input_length = None
        self.encoder_outputs = None
        self.encoder_output_lengths = None
        self._runs_done = 0
        self._encode_target_once()

    def _encode_target_once(self):
        self.encoder_outputs, self.encoder_output_lengths, self.encoder_max_input_length = (
            _encode_target_for_batch(self.args.target_engine_dir, self.mel,
                                     self.mel_input_lengths, self.batch_size,
                                     identical_batch=self.identical_batch))

    def setup_ngram_pool(self, draft_len, end_id):
        if self.args.draft_mode in ('ngram', 'hybrid'):
            self.ngram_pool = NgramDraftPool(draft_len,
                                             self.args.ngram_max_matching_size,
                                             end_id)

    def _reset_for_new_run(self, draft_len, end_id):
        """Reset draft/target state between full transcription passes."""
        if isinstance(self.draft_decoder, PersistentWhisperDraftDecoder):
            self.draft_decoder.reset_session()
        runtime_rank = tensorrt_llm.mpi_rank()
        _, self.target_runner = _build_runners(
            self.args,
            self.draft_device_list,
            self.target_device_list,
            runtime_rank,
            self.mel,
            self.mel_input_lengths,
            batch_size=self.batch_size,
            identical_batch=self.identical_batch,
        )
        if self.args.draft_mode in ('ngram', 'hybrid'):
            self.setup_ngram_pool(draft_len, end_id)
        else:
            self.ngram_pool = None

    def run(self, draft_len, end_id, profile=False, sync_identical_batch=False,
            batched_target_verify=None):
        if batched_target_verify is None:
            batched_target_verify = _resolve_batched_target_verify(
                self.args, self.batch_size)
        self.end_id = end_id
        if self._runs_done > 0:
            self._reset_for_new_run(draft_len, end_id)
        self._runs_done += 1
        start = time.time()
        output_ids, stats = run_whisper_dtm(
            self.prefix,
            self.target_runner,
            self.draft_decoder,
            self.ngram_pool,
            self.args.draft_mode,
            draft_len,
            end_id,
            self.args.max_output_len,
            self.mel,
            self.mel_input_lengths,
            target_encoder_output=self.encoder_outputs,
            target_encoder_output_lengths=self.encoder_output_lengths,
            temperature=self.args.temperature,
            top_k=self.args.top_k,
            top_p=self.args.top_p,
            profile=profile,
            sync_identical_batch=sync_identical_batch,
            batched_target_verify=batched_target_verify,
            encoder_max_input_length=self.encoder_max_input_length,
            identical_batch=self.identical_batch,
        )
        elapsed = time.time() - start
        stats['elapsed'] = elapsed
        stats['elapsed_per_seq'] = elapsed / self.batch_size
        mean_duration = self.duration
        if getattr(self, 'durations', None):
            mean_duration = sum(self.durations) / len(self.durations)
        stats['rtf'] = elapsed / (mean_duration * self.batch_size)
        stats['rtf_per_seq'] = stats['elapsed_per_seq'] / mean_duration
        return output_ids, stats


def _run_once(args, draft_len, draft_device_list, target_device_list, prefix,
              mel, mel_input_lengths, duration, end_id, batch_size=1,
              sync_identical_batch=False, session=None):
    if session is not None:
        session.duration = duration
        session.end_id = end_id
        return session.run(draft_len,
                           end_id,
                           profile=getattr(args, 'profile', False),
                           sync_identical_batch=sync_identical_batch,
                           batched_target_verify=_resolve_batched_target_verify(
                               args, batch_size))

    runtime_rank = tensorrt_llm.mpi_rank()
    if batch_size > 1:
        mel = mel.repeat(batch_size, 1, 1)
        mel_input_lengths = mel_input_lengths.repeat(batch_size)
        prefix = [prefix[0].clone() for _ in range(batch_size)]

    draft_decoder, target_runner = _build_runners(
        args, draft_device_list, target_device_list, runtime_rank, mel,
        mel_input_lengths, batch_size=batch_size)
    if isinstance(draft_decoder, PersistentWhisperDraftDecoder):
        draft_decoder.encode_once(mel[:1].transpose(1, 2),
                                  mel_input_lengths[:1])

    ngram_pool = None
    if args.draft_mode in ('ngram', 'hybrid'):
        ngram_pool = NgramDraftPool(draft_len, args.ngram_max_matching_size,
                                    end_id)

    encoder_outputs, encoder_output_lengths, encoder_max_input_length = (
        _encode_target_for_batch(args.target_engine_dir, mel, mel_input_lengths,
                                 batch_size))

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
        target_encoder_output=encoder_outputs,
        target_encoder_output_lengths=encoder_output_lengths,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        profile=getattr(args, 'profile', False),
        sync_identical_batch=sync_identical_batch,
        batched_target_verify=_resolve_batched_target_verify(args, batch_size),
        encoder_max_input_length=encoder_max_input_length,
    )
    elapsed = time.time() - start
    stats['elapsed'] = elapsed
    stats['elapsed_per_seq'] = elapsed / batch_size
    stats['rtf'] = elapsed / (duration * batch_size)
    stats['rtf_per_seq'] = stats['elapsed_per_seq'] / duration
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
                                  mel_input_lengths, duration, end_id,
                                  batch_size=args.batch_size,
                                  sync_identical_batch=args.sync_identical_batch)

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
