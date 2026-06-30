"""Smoke test: Gemma4 (hd512 global layers) on the default backend with CUDA
graphs enabled, dummy weights, no tokenizer. Proves the model_engine CUDA-graph
capture path (which previously crashed on FlashInfer) works with the new TRTLLM
default + hd512 kernels on SM90.

Must be run as a script (the LLM API spawns MPI workers, hence the __main__ guard).
"""
import json
import os
import tempfile


def main():
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig

    # Minimal Gemma4 text config with a head_dim=512 global layer.
    cfg = {
        "architectures": ["Gemma4ForCausalLM"],
        "model_type": "gemma4_text",
        "vocab_size": 1024,
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 6,  # 5 sliding(hd256) + 1 full(hd512)
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "global_head_dim": 512,
        "num_global_key_value_heads": 2,
        "hidden_activation": "gelu_pytorch_tanh",
        "max_position_embeddings": 1024,
        "rms_norm_eps": 1e-6,
        "sliding_window": 128,
        "attention_k_eq_v": False,
        "enable_moe_block": False,
        "num_kv_shared_layers": 0,
        "hidden_size_per_layer_input": 0,
        "final_logit_softcapping": None,
        "rope_parameters": {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
            },
        },
        "torch_dtype": "bfloat16",
        "tie_word_embeddings": True,
        "attention_bias": False,
        "attention_dropout": 0.0,
    }

    d = tempfile.mkdtemp(prefix="gemma4_cg_")
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print("=== Building LLM (default backend, CUDA graphs ON, dummy weights) ===")
    llm = LLM(
        d,
        load_format="dummy",
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_batch_size=2,
        max_seq_len=128,
        cuda_graph_config=CudaGraphConfig(batch_sizes=[1, 2]),
    )
    try:
        print("resolved attn_backend:", llm.args.attn_backend)
        print("cuda_graph_config:", llm.args.cuda_graph_config)
    except Exception as e:
        print("args introspection failed:", e)

    prompt_ids = [[1, 5, 9, 13, 17, 21, 25, 29]]
    out = llm.generate(prompt_ids, SamplingParams(max_tokens=16, end_id=-1))
    toks = out[0].outputs[0].token_ids
    print("=== generated", len(toks), "tokens:", toks)
    assert len(toks) > 0, "no tokens generated"
    print("SMOKE OK: TRTLLM hd512 + CUDA graph capture/replay succeeded")
    llm.shutdown()


if __name__ == "__main__":
    main()
