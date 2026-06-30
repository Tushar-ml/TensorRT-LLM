#!/usr/bin/env python3
"""Validate FlashInfer paged-prefill (the _forward_gen_prefill path) against an
SDPA reference for the MTP spec-dec multi-query generation scenario.

Scenario: a generation request has `past` cached tokens already in the paged KV
cache, then appends q_len new query tokens (q_len = draft_len + 1). The new
tokens' KV is already written into the cache (append happened before attn).
Causal masking aligns the LAST query token with the LAST kv position.
"""
import math
import torch
import flashinfer

DEVICE = "cuda"
DTYPE = torch.bfloat16
torch.manual_seed(0)


def sdpa_ref(q, k_full, v_full, sm_scale, window_left):
    # q: [q_len, H, D]  k_full/v_full: [kv_len, Hkv, D]
    q_len, H, D = q.shape
    kv_len, Hkv, _ = k_full.shape
    rep = H // Hkv
    k = k_full.repeat_interleave(rep, dim=1)  # [kv_len, H, D]
    v = v_full.repeat_interleave(rep, dim=1)
    qf = q.float().permute(1, 0, 2)        # [H, q_len, D]
    kf = k.float().permute(1, 0, 2)        # [H, kv_len, D]
    vf = v.float().permute(1, 0, 2)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * sm_scale  # [H, q_len, kv_len]
    # causal: query i (0..q_len-1) corresponds to kv position (kv_len-q_len+i)
    q_pos = torch.arange(q_len, device=DEVICE) + (kv_len - q_len)
    k_pos = torch.arange(kv_len, device=DEVICE)
    mask = k_pos[None, :] > q_pos[:, None]  # future
    if window_left >= 0:
        # sliding window: attend only to [q_pos - window_left, q_pos]
        too_old = k_pos[None, :] < (q_pos[:, None] - window_left)
        mask = mask | too_old
    scores = scores.masked_fill(mask[None], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, vf)          # [H, q_len, D]
    return out.permute(1, 0, 2).to(DTYPE)  # [q_len, H, D]


def run(H=8, Hkv=2, D=512, page=16, past=40, q_len=2, window_left=-1, backend="fa2"):
    label = f"D={D} past={past} q_len={q_len} win={window_left} backend={backend}"
    kv_len = past + q_len
    n_pages = (kv_len + page - 1) // page
    sm_scale = 1.0 / math.sqrt(D)

    # paged KV cache layout NHD: [n_pages, 2, page, Hkv, D]
    kv_cache = torch.randn(n_pages, 2, page, Hkv, D, device=DEVICE, dtype=DTYPE)
    # reconstruct contiguous k/v for reference
    k_flat = kv_cache[:, 0].reshape(-1, Hkv, D)[:kv_len]
    v_flat = kv_cache[:, 1].reshape(-1, Hkv, D)[:kv_len]

    q = torch.randn(q_len, H, D, device=DEVICE, dtype=DTYPE)

    ref = sdpa_ref(q, k_flat, v_flat, sm_scale, window_left)

    qo_indptr = torch.tensor([0, q_len], device=DEVICE, dtype=torch.int32)
    kv_indptr = torch.tensor([0, n_pages], device=DEVICE, dtype=torch.int32)
    kv_indices = torch.arange(n_pages, device=DEVICE, dtype=torch.int32)
    last = kv_len - (n_pages - 1) * page
    kv_last_page_len = torch.tensor([last], device=DEVICE, dtype=torch.int32)

    ws = torch.empty(128 * 1024 * 1024, device=DEVICE, dtype=torch.uint8)
    be = backend
    if be == "fa2" and torch.cuda.get_device_capability(0) != (9, 0):
        be = "auto"
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend=be)
    wrapper.plan(
        qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
        H, Hkv, D, page, causal=True, sm_scale=sm_scale,
        window_left=window_left, q_data_type=DTYPE, kv_data_type=DTYPE,
    )
    out = torch.empty(q_len, H, D, device=DEVICE, dtype=DTYPE)
    wrapper.run(q, kv_cache, out=out)
    torch.cuda.synchronize()

    diff = (out.float() - ref.float()).abs()
    rel = diff.max().item() / (ref.float().abs().max().item() + 1e-6)
    ok = rel < 0.02
    print(f"  [{'OK ' if ok else 'BAD'}] {label}: max_abs={diff.max().item():.4f} "
          f"rel={rel:.4f} ref_absmax={ref.float().abs().max().item():.3f}")
    return ok


if __name__ == "__main__":
    print(f"flashinfer {flashinfer.__version__} cap={torch.cuda.get_device_capability(0)}")
    allok = True
    # full-attention 512 layer, the Gemma4 global layers
    allok &= run(D=512, past=40, q_len=2, window_left=-1)
    allok &= run(D=512, past=200, q_len=2, window_left=-1)
    # sliding 256 layer
    allok &= run(D=256, past=40, q_len=2, window_left=512)
    allok &= run(D=256, past=600, q_len=2, window_left=512)
    # single query (decode-equiv)
    allok &= run(D=512, past=40, q_len=1, window_left=-1)
    print("ALL OK" if allok else "FAILURES DETECTED")
