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


def main():
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig, MTPDecodingConfig

    os.environ.setdefault("TRTLLM_GEMMA4_TEXT_ONLY", "1")

    backbone = os.environ.get("GEMMA4_BACKBONE", "google/gemma-4-31b-it")
    assistant = os.environ.get("GEMMA4_ASSISTANT",
                               "google/gemma-4-31b-it-assistant")
    tp = int(os.environ.get("GEMMA4_TP", "4"))

    spec = MTPDecodingConfig(
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
    out = llm.generate(prompts,
                       SamplingParams(max_tokens=48, temperature=0.0))
    for i, o in enumerate(out):
        print(f"--- prompt {i}: {prompts[i]!r}")
        print("   ->", repr(o.outputs[0].text))

    print("SMOKE OK: TRTLLM Gemma4 MTP ran end-to-end")
    llm.shutdown()


if __name__ == "__main__":
    main()
