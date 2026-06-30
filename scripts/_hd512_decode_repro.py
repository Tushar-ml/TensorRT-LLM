"""In-process hd512 decode repro (no LLM API / no MPI) for compute-sanitizer / cuda-gdb.

Builds the Gemma4 hd512 TRTLLM-backend model and runs a generation-phase forward
matching the executor warmup geometry that triggers CUDA 700 in the Dh=512 MMHA
decode kernel.  The crashing case is the *first-token* decode: 0 cached tokens,
1 newly-appended token -> kernel tlength=1, batch=1, grid=(4,1,1).

Env knobs:
  REPRO_BATCH  (default 1)   number of generation requests
  REPRO_CACHED (default 0)   prior cached tokens per seq (0 => tlength=1, the crash)
"""
import os
import sys

os.environ["TRTLLM_MMHA_HEAD_DIM_512"] = "1"

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests", "unittest", "_torch", "modeling"))
from copy import deepcopy

from test_modeling_gemma4 import (  # noqa: E402
    GEMMA4_HD512_CONFIG,
    _build_gemma4_kv_cache_manager,
)

from tensorrt_llm._torch.attention_backend.utils import get_attention_backend  # noqa: E402
from tensorrt_llm._torch.metadata import KVCacheParams  # noqa: E402
from tensorrt_llm._torch.model_config import ModelConfig  # noqa: E402
from tensorrt_llm._torch.models.modeling_gemma4 import Gemma4ForCausalLM  # noqa: E402
from transformers import Gemma4TextConfig  # noqa: E402


def main():
    backend = "TRTLLM"
    device = torch.device("cuda")
    cfg = Gemma4TextConfig(**deepcopy(GEMMA4_HD512_CONFIG))
    dtype = cfg.torch_dtype

    model_config = ModelConfig(pretrained_config=cfg, attn_backend=backend)
    trt = Gemma4ForCausalLM(model_config).to(dtype).to(device)

    batch_size = int(os.environ.get("REPRO_BATCH", "1"))
    cached = int(os.environ.get("REPRO_CACHED", "0"))
    # Reserve room for the cached tokens plus the appended decode token.
    reserve = max(cached + 1, 1)

    num_blocks = int(os.environ.get("REPRO_NUM_BLOCKS", "4"))  # 4*32=128 -> cyc_win=128 (matches smoke)
    kv_cache_manager = _build_gemma4_kv_cache_manager(
        cfg, num_blocks=num_blocks, tokens_per_block=32, batch_size=max(batch_size, 1)
    )

    metadata_cls = get_attention_backend(backend).Metadata

    request_ids = list(range(1, batch_size + 1))
    prompt_lens = [cached] * batch_size
    kv_cache_manager.add_dummy_requests(request_ids, [reserve] * batch_size)

    print(f"=== repro: batch={batch_size} cached={cached} (expect kernel tlength={cached + 1}) ===",
          flush=True)

    gen_input_ids = torch.arange(900, 900 + batch_size, dtype=torch.int32, device=device)
    attn_metadata = metadata_cls(
        seq_lens=torch.tensor([1] * batch_size, dtype=torch.int),
        num_contexts=0,
        kv_cache_params=KVCacheParams(
            use_cache=True, num_cached_tokens_per_seq=[cached] * batch_size
        ),
        kv_cache_manager=kv_cache_manager,
        request_ids=request_ids,
        prompt_lens=prompt_lens,
        max_num_requests=max(batch_size, 1),
        max_num_tokens=8192,
    )
    gen_pos = torch.tensor([[cached] * batch_size], dtype=torch.int32, device=device)
    with torch.inference_mode():
        attn_metadata.prepare()
        out = trt.forward(
            input_ids=gen_input_ids, position_ids=gen_pos, attn_metadata=attn_metadata
        )
    torch.cuda.synchronize()
    print("=== generation phase done, out shape:", out.shape, flush=True)
    print("REPRO OK", flush=True)
    kv_cache_manager.shutdown()


if __name__ == "__main__":
    main()
