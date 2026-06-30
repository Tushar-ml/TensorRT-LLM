"""Greedy token-id parity: plain decode vs MTP one-engine (same build)."""
import os
import sys


def main():
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig, MTPDecodingConfig

    os.environ.setdefault("TRTLLM_GEMMA4_TEXT_ONLY", "1")
    tp = int(os.environ.get("GEMMA4_TP", "4"))
    max_tokens = int(os.environ.get("GEMMA4_GEN_TOKENS", "8"))
    prompts = ["The capital of France is"]
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    kv = KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.5)

    def run(no_spec: bool):
        spec = None if no_spec else MTPDecodingConfig(
            speculative_model=os.environ.get(
                "GEMMA4_ASSISTANT", "google/gemma-4-31b-it-assistant"),
            max_draft_len=1,
            mtp_eagle_one_model=True,
        )
        llm = LLM(
            os.environ.get("GEMMA4_BACKBONE", "google/gemma-4-31b-it"),
            tensor_parallel_size=tp,
            backend="pytorch",
            max_batch_size=1,
            max_num_tokens=2048,
            max_seq_len=2048,
            kv_cache_config=kv,
            cuda_graph_config=None,
            speculative_config=spec,
        )
        out = llm.generate(prompts, sp)[0].outputs[0]
        llm.shutdown()
        return list(out.token_ids), out.text

    base_ids, base_text = run(no_spec=True)
    mtp_ids, mtp_text = run(no_spec=False)
    print("BASE ids:", base_ids)
    print("BASE text:", repr(base_text))
    print("MTP  ids:", mtp_ids)
    print("MTP  text:", repr(mtp_text))
    match = base_ids == mtp_ids
    print("TOKEN MATCH:", match)
    return 0 if match else 1


if __name__ == "__main__":
    sys.exit(main())
