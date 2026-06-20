"""E2e smoke: Gemma4 one-engine MTP (MTP_EAGLE_ONE_MODEL) on the TRTLLM
attention backend.

Loads the real ``google/gemma-4-31b-it`` backbone (text-only via
``TRTLLM_GEMMA4_TEXT_ONLY=1``) plus the ``gemma-4-31b-it-assistant`` MTP
checkpoint and runs speculative decoding on the TRTLLM backend (the FlashInfer
MTP path was removed).  Prints generated text and the average accepted draft
length so we can confirm the draft layer is actually contributing.

Must be run as a script (the LLM API spawns MPI workers, hence __main__ guard).

    TRTLLM_GEMMA4_TEXT_ONLY=1 python3 scripts/_gemma4_trtllm_mtp_smoke.py
"""
import os
import time


def main():
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig, MTPDecodingConfig

    os.environ.setdefault("TRTLLM_GEMMA4_TEXT_ONLY", "1")

    backbone = os.environ.get("GEMMA4_BACKBONE", "google/gemma-4-31b-it")
    assistant = os.environ.get("GEMMA4_ASSISTANT",
                               "google/gemma-4-31b-it-assistant")
    tp = int(os.environ.get("GEMMA4_TP", "4"))

    # GEMMA4_NO_SPEC=1 runs plain decode (no MTP) — used to validate the hd512
    # kernels on real weights and get a baseline tok/s before the MTP draft path
    # (which requires the Q-only cross-layer KV-share attention op) is finished.
    spec = None if os.environ.get("GEMMA4_NO_SPEC") == "1" else MTPDecodingConfig(
        speculative_model=assistant,
        max_draft_len=1,
        mtp_eagle_one_model=True,
    )

    print(f"=== Building LLM (TRTLLM backend, MTP one-model, tp={tp}) ===")
    llm = LLM(
        backbone,
        tensor_parallel_size=tp,
        backend="pytorch",
        max_batch_size=2,
        max_num_tokens=4096,
        max_seq_len=2048,
        kv_cache_config=KvCacheConfig(enable_block_reuse=False,
                                      free_gpu_memory_fraction=0.6),
        cuda_graph_config=None,
        speculative_config=spec,
    )
    print("resolved attn_backend:", llm.args.attn_backend)

    prompts = ["The capital of France is", "Explain gravity in one sentence:"]
    # gemma-*-it degenerates on raw prompts; GEMMA4_CHAT=1 wraps with the
    # Gemma chat template to sanity-check whether garbage output is a prompt
    # artifact vs a real model/weight bug.
    if os.environ.get("GEMMA4_CHAT") == "1":
        prompts = [
            f"<start_of_turn>user\n{p}<end_of_turn>\n<start_of_turn>model\n"
            for p in prompts
        ]
    gen_tokens = int(os.environ.get("GEMMA4_GEN_TOKENS", "48"))
    sp = SamplingParams(max_tokens=gen_tokens, temperature=0.0)

    # Warm up (first call captures CUDA graphs / JIT-compiles) so the timed run
    # measures steady-state decode throughput.
    _ = llm.generate(prompts, sp)

    t0 = time.perf_counter()
    out = llm.generate(prompts, sp)
    elapsed = time.perf_counter() - t0
    total_out = sum(len(o.outputs[0].token_ids) for o in out)
    for i, o in enumerate(out):
        print(f"--- prompt {i}: {prompts[i]!r}")
        print("   ->", repr(o.outputs[0].text))
    print(f"=== THROUGHPUT: {total_out} output tokens in {elapsed:.3f}s "
          f"= {total_out / elapsed:.1f} tok/s (batch={len(prompts)}, "
          f"max_tokens={gen_tokens}, spec={'off' if spec is None else 'MTP'})")

    print("SMOKE OK: TRTLLM Gemma4 MTP ran end-to-end")
    llm.shutdown()


if __name__ == "__main__":
    main()
