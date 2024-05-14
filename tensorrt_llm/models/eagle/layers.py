import torch
from tensorrt_llm.module import Module
from tensorrt_llm.functional import concat, einsum, arange

class EAGLERotaryEmbedding(Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_parameter("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", [t, self.inv_freq])
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_parameter("cos_cached", emb.cos()[None, None, :, :].to(dtype))
        self.register_parameter("sin_cached", emb.sin()[None, None, :, :].to(dtype))

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class EAGLELinearScalingRotaryEmbedding(EAGLERotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = einsum("i,j->ij", [t, self.inv_freq])
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = concat((freqs, freqs), dim=-1)
        self.register_parameter("cos_cached", emb.cos()[None, None, :, :].to(dtype))
        self.register_parameter("sin_cached", emb.sin()[None, None, :, :].to(dtype))


class EAGLEDynamicNTKScalingRotaryEmbedding(EAGLERotaryEmbedding):

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                    (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_parameter("inv_freq", inv_freq)

        t = arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)

        freqs = einsum("i,j->ij", [t, self.inv_freq])
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = concat((freqs, freqs), dim=-1)
        self.register_parameter("cos_cached", emb.cos()[None, None, :, :].to(dtype))
        self.register_parameter("sin_cached", emb.sin()[None, None, :, :].to(dtype))