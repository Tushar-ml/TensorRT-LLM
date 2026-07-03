#!/usr/bin/env python3
"""Microtest FlashInfer fa2 head_dim=512 paged prefill+decode on SM90."""
import math
import torch
import flashinfer

NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 512
PAGE_SIZE = 16
DTYPE = torch.bfloat16
DEVICE = "cuda"


def run_test(use_sliding_window: bool = False):
    label = "sliding" if use_sliding_window else "global"
    print(f"\n=== head_dim=512 {label} ===")

    batch_size = 2
    seq_lens = [32, 48]
    max_seq = max(seq_lens)
    num_pages = sum((s + PAGE_SIZE - 1) // PAGE_SIZE for s in seq_lens)

    # Build paged KV cache
    kv_cache = torch.randn(
        num_pages, 2, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM,
        device=DEVICE, dtype=DTYPE,
    )

    qo_indptr = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lens), 0)),
                             device=DEVICE, dtype=torch.int32)
    kv_indptr = torch.zeros(batch_size + 1, device=DEVICE, dtype=torch.int32)
    kv_indices = torch.arange(num_pages, device=DEVICE, dtype=torch.int32)
    kv_last_page_len = torch.tensor(
        [s - (s // PAGE_SIZE) * PAGE_SIZE or PAGE_SIZE for s in seq_lens],
        device=DEVICE, dtype=torch.int32,
    )
    offset = 0
    for i, s in enumerate(seq_lens):
        n_pages = (s + PAGE_SIZE - 1) // PAGE_SIZE
        kv_indptr[i + 1] = kv_indptr[i] + n_pages

    total_q = sum(seq_lens)
    q = torch.randn(total_q, NUM_Q_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    out = torch.empty_like(q)

    workspace = torch.empty(128 * 1024 * 1024, device=DEVICE, dtype=torch.uint8)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend="fa2")

    window_left = 256 if use_sliding_window else -1
    wrapper.plan(
        qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
        NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM,
        PAGE_SIZE, causal=True, q_data_type=DTYPE, kv_data_type=DTYPE,
        window_left=window_left,
    )
    wrapper.run(q, kv_cache, out=out)
    torch.cuda.synchronize()
    print(f"  prefill OK: out shape={out.shape}, mean={out.float().mean().item():.4f}")

    # Single-token decode
    dec_q = torch.randn(batch_size, NUM_Q_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    dec_out = torch.empty_like(dec_q)
    dec_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", backend="fa2")
    dec_wrapper.plan(
        kv_indptr, kv_indices, kv_last_page_len,
        NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM,
        PAGE_SIZE, q_data_type=DTYPE, kv_data_type=DTYPE,
        window_left=window_left,
    )
    dec_wrapper.run(dec_q, kv_cache, out=dec_out)
    torch.cuda.synchronize()
    print(f"  decode OK: out shape={dec_out.shape}, mean={dec_out.float().mean().item():.4f}")


if __name__ == "__main__":
    print(f"FlashInfer {flashinfer.__version__}")
    run_test(use_sliding_window=False)
    run_test(use_sliding_window=True)
    print("\nAll hd512 tests passed.")
