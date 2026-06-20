# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from ordered_set import OrderedSet

from run import WhisperDecoding, WhisperEncoding


def prefixes_equal(prefix):
    if len(prefix) <= 1:
        return True
    leader = prefix[0]
    return all(torch.equal(leader, p) for p in prefix[1:])


def build_cross_attention_masks(batch_size, prefix_len, draft_len, encoder_len):
    """Optional cross-attention masks for batched external-draft verify.

    Custom masks currently break external-draft acceptance; callers should rely
    on the C++ default padding mask until mask row indexing is fixed for verify.
    """
    return None


def replicate_encoder_for_batch(encoder_output, encoder_output_lengths,
                                batch_size):
    """Replicate bs=1 encoder features for identical batch slots."""
    if isinstance(encoder_output_lengths, torch.Tensor):
        enc_len = int(encoder_output_lengths[0].item())
    elif isinstance(encoder_output_lengths, (list, tuple)):
        enc_len = int(encoder_output_lengths[0])
    else:
        enc_len = int(encoder_output_lengths)
    if batch_size == 1:
        return [encoder_output], [enc_len]
    return [encoder_output] * batch_size, [enc_len] * batch_size


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


class NgramDraftPool:
    """Lightweight ngram draft pool adapted from examples/ngram/run_dtm_ngram.py."""

    def __init__(self, max_draft_len, max_matching_ngram_size, end_id):
        self.max_draft_len = max_draft_len
        self.max_matching_ngram_size = max_matching_ngram_size
        self.end_id = end_id
        self.pool = {}
        self.start_index = 0

    def get_draft_tokens(self, prefix):
        prefix_len = len(prefix[0])
        if prefix_len <= self.start_index:
            return [[self.end_id]]

        sequence = prefix[0][self.start_index:].tolist()
        for size in range(min(self.max_matching_ngram_size, prefix_len - 1), 0,
                          -1):
            for left in range(len(sequence) - size):
                right = min(left + size + self.max_draft_len, len(sequence))
                key = tuple(sequence[left:left + size])
                value = tuple(sequence[left + size:right])
                if key not in self.pool or len(self.pool[key][0]) < self.max_draft_len:
                    self.pool[key] = OrderedSet((value, ))
                elif value not in self.pool[key]:
                    self.pool[key].add(value)

        chosen = [self.end_id]
        for size in range(min(self.max_matching_ngram_size, prefix_len - 1), 0,
                          -1):
            pattern = tuple(prefix[0][-size:].tolist())
            if pattern in self.pool:
                chosen = self.pool[pattern][0]
                break

        self.start_index = max(
            0, prefix_len - (self.max_draft_len + self.max_matching_ngram_size -
                             1))
        return [chosen]


class CppDraftRunner:
    """ModelRunnerCpp draft proposer with optional encoder-once cache."""

    def __init__(self,
                 runner,
                 mel,
                 mel_input_lengths,
                 temperature,
                 top_k,
                 top_p,
                 draft_engine_dir=None,
                 batch_size=1):
        self.runner = runner
        self.mel = mel
        self.mel_input_lengths = mel_input_lengths
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.batch_size = batch_size
        self._committed_prefix_lens = [0] * batch_size
        self._encoder_output = None
        self._encoder_output_lengths = None
        if draft_engine_dir is not None:
            encoder = WhisperEncoding(draft_engine_dir)
            single_mel = mel[:1] if mel.shape[0] > 1 else mel
            single_lens = (mel_input_lengths[:1]
                           if mel_input_lengths.shape[0] > 1 else mel_input_lengths)
            self._encoder_output, self._encoder_output_lengths = encoder.get_audio_features(
                single_mel.transpose(1, 2), single_lens)

    def propose(self, prefix, end_id, draft_len):
        if len(prefix) > 1 and prefixes_equal(prefix):
            draft_tokens_list, max_d_len = self.propose([prefix[0]], end_id,
                                                        draft_len)
            return [draft_tokens_list[0][:] for _ in prefix], max_d_len

        gen_kwargs = dict(
            batch_input_ids=prefix,
            max_new_tokens=draft_len,
            end_id=end_id,
            pad_id=end_id,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            num_beams=1,
            streaming=False,
            output_sequence_lengths=True,
            return_dict=True,
        )
        if self._encoder_output is not None:
            gen_kwargs['encoder_outputs'] = _split_encoder_outputs(
                self._encoder_output, len(prefix),
                self._encoder_output_lengths.tolist())
            gen_kwargs['encoder_output_lengths'] = (
                self._encoder_output_lengths.tolist())
        else:
            gen_kwargs['encoder_input_features'] = self.mel
            gen_kwargs['encoder_output_lengths'] = (
                self.mel_input_lengths // 2).tolist()

        draft = self.runner.generate(**gen_kwargs)
        draft_tokens_list = []
        max_d_len = 0
        for bi in range(len(prefix)):
            prefix_len = len(prefix[bi])
            d_seq_len = draft['sequence_lengths'][bi, 0].item()
            if d_seq_len <= prefix_len:
                draft_tokens_list.append([end_id])
                continue
            draft_ids = draft['output_ids'][bi, 0,
                                            prefix_len:d_seq_len].tolist()
            draft_tokens_list.append(draft_ids)
            max_d_len = max(max_d_len, len(draft_ids))
        return draft_tokens_list, max_d_len

    def commit_prefix(self, prefix_len, batch_index=0):
        if batch_index < len(self._committed_prefix_lens):
            self._committed_prefix_lens[batch_index] = prefix_len

    def on_rejection(self, batch_index=0):
        if batch_index < len(self._committed_prefix_lens):
            self._committed_prefix_lens[batch_index] = 0


class PersistentWhisperDraftDecoder:
    """Draft decoder with encoder-once py_session path and session rewind on rejection."""

    def __init__(self, draft_engine_dir, runtime_mapping, debug_mode=False,
                 max_batch_size=8):
        self.encoder = WhisperEncoding(draft_engine_dir)
        self.decoder = WhisperDecoding(draft_engine_dir,
                                       runtime_mapping,
                                       debug_mode=debug_mode)
        self.max_batch_size = max_batch_size
        self._encoder_output = None
        self._encoder_output_lengths = None
        self._encoder_max_input_length = None
        self._committed_prefix_lens = [0] * max_batch_size
        self._needs_reset = True
        self._needs_reset_slots = set()

    def encode_once(self, mel, mel_input_lengths):
        single_mel = mel[:1] if mel.dim() >= 3 and mel.shape[0] > 1 else mel
        single_lens = (mel_input_lengths[:1]
                       if mel_input_lengths.shape[0] > 1 else mel_input_lengths)
        encoder_output, encoder_output_lengths = self.encoder.get_audio_features(
            single_mel, single_lens)
        self._encoder_output = encoder_output
        self._encoder_output_lengths = encoder_output_lengths
        self._encoder_max_input_length = torch.max(
            encoder_output_lengths).item()
        self._committed_prefix_lens = [0] * self.max_batch_size
        self._needs_reset_slots.clear()
        self._needs_reset = True

    def reset_session(self, batch_index=None):
        if batch_index is None:
            self._needs_reset_slots.clear()
            self._committed_prefix_lens = [0] * self.max_batch_size
        else:
            self._needs_reset_slots.add(batch_index)
            if batch_index < len(self._committed_prefix_lens):
                self._committed_prefix_lens[batch_index] = 0
        self._needs_reset = True
        self.decoder.rewind_draft_session()

    def _should_force_reset(self, prefix_len, batch_size):
        if self._needs_reset or self._needs_reset_slots:
            return True
        for bi in range(batch_size):
            if prefix_len < self._committed_prefix_lens[bi]:
                return True
        return False

    def propose(self, prefix, end_id, draft_len):
        batch_size = len(prefix)
        prefix_len = len(prefix[0])
        force_reset = self._should_force_reset(prefix_len, batch_size)
        if force_reset:
            self._needs_reset = False
            self._needs_reset_slots.clear()
            for bi in range(batch_size):
                self._committed_prefix_lens[bi] = 0

        propose_prefix = prefix
        if batch_size > 1 and prefixes_equal(prefix):
            propose_prefix = [prefix[0]]

        decoder_input_ids = propose_prefix[0].unsqueeze(0).cuda()

        output_ids = self.decoder.propose_draft(
            decoder_input_ids,
            self._encoder_output,
            self._encoder_max_input_length,
            self._encoder_output_lengths,
            end_id,
            draft_len,
            force_reset=force_reset,
        )
        draft_tokens_list = []
        max_d_len = 0
        for bi in range(len(propose_prefix)):
            seq = output_ids[bi][0]
            if isinstance(seq, torch.Tensor):
                seq = seq.tolist()
            seq_len = len(seq)
            if seq_len <= prefix_len:
                draft_tokens_list.append([end_id])
                continue
            draft_ids = seq[prefix_len:seq_len]
            draft_tokens_list.append(draft_ids)
            max_d_len = max(max_d_len, len(draft_ids))
            self._committed_prefix_lens[bi] = prefix_len
        if batch_size > len(draft_tokens_list):
            leader = draft_tokens_list[0][:]
            draft_tokens_list = [leader[:] for _ in range(batch_size)]
            for bi in range(batch_size):
                self._committed_prefix_lens[bi] = prefix_len
        return draft_tokens_list, max_d_len

    def commit_prefix(self, prefix_len, batch_index=0):
        if batch_index < len(self._committed_prefix_lens):
            self._committed_prefix_lens[batch_index] = prefix_len

    def on_rejection(self, batch_index=0):
        self.reset_session(batch_index)


class IterationProfiler:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.records = []

    def record(self, iteration, prefix_len, draft_ms, target_ms, accepted,
               draft_count):
        if not self.enabled:
            return
        self.records.append({
            'iteration': iteration,
            'prefix_len': prefix_len,
            'draft_ms': draft_ms,
            'target_ms': target_ms,
            'accepted': accepted,
            'draft_count': draft_count,
        })

    def summary(self):
        if not self.records:
            return {}
        draft_total = sum(r['draft_ms'] for r in self.records)
        target_total = sum(r['target_ms'] for r in self.records)
        return {
            'iterations': len(self.records),
            'draft_ms_total': draft_total,
            'target_ms_total': target_total,
            'draft_ms_avg': draft_total / len(self.records),
            'target_ms_avg': target_total / len(self.records),
        }

    def print_summary(self):
        if not self.enabled or not self.records:
            return
        summary = self.summary()
        print('Profile summary:')
        print(f"  iterations: {summary['iterations']}")
        print(f"  draft total/avg ms: {summary['draft_ms_total']:.2f} / "
              f"{summary['draft_ms_avg']:.2f}")
        print(f"  target total/avg ms: {summary['target_ms_total']:.2f} / "
              f"{summary['target_ms_avg']:.2f}")
        for rec in self.records:
            print(f"  iter {rec['iteration']:2d} prefix={rec['prefix_len']:3d} "
                  f"draft={rec['draft_ms']:.2f}ms target={rec['target_ms']:.2f}ms "
                  f"accept={rec['accepted']}/{rec['draft_count']}")
