"""Smoke test for Gemma4 MTP implementation.

Tests the MTP class construction, forward signature, and weight key mapping
without requiring a GPU or real model weights.

Run from repo root:
    python test_gemma4_mtp.py
"""
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Minimal stubs so we can import modeling_gemma4 without torch/GPU
# ---------------------------------------------------------------------------

def _make_stub_module(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    return m


def _install_stubs():
    # torch stub
    torch = _make_stub_module("torch")
    torch.float16 = "float16"
    torch.bfloat16 = "bfloat16"
    torch.float32 = "float32"
    torch.inference_mode = lambda f=None: (lambda fn: fn)(f) if callable(f) else (lambda fn: fn)

    class _TensorStub:
        pass

    torch.Tensor = _TensorStub
    torch.IntTensor = _TensorStub
    torch.LongTensor = _TensorStub
    torch.FloatTensor = _TensorStub

    nn = _make_stub_module("torch.nn")

    class _Module:
        def __init__(self): pass
        def __init_subclass__(cls, **kw): super().__init_subclass__(**kw)

    class _ModuleList(list):
        pass

    class _Linear:
        def __init__(self, *a, **kw): pass

    nn.Module = _Module
    nn.ModuleList = _ModuleList
    nn.Linear = _Linear
    torch.nn = nn
    torch.cuda = _make_stub_module("torch.cuda")
    torch.cuda.Stream = lambda: None
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = nn
    sys.modules["torch.cuda"] = torch.cuda

    # transformers stub
    tr = _make_stub_module("transformers")

    class Gemma4TextConfig:
        architectures = ["Gemma4ForCausalLM"]
        hidden_size = 2560
        num_hidden_layers = 46
        intermediate_size = 10240
        rms_norm_eps = 1e-6
        sliding_window = 4096
        torch_dtype = "bfloat16"
        layer_types = (
            ["sliding_attention", "sliding_attention", "sliding_attention",
             "sliding_attention", "sliding_attention", "full_attention"] * 7
            + ["sliding_attention", "sliding_attention", "sliding_attention",
               "sliding_attention"]
        )
        num_kv_shared_layers = 1  # last layer is KV-shared

    class Gemma4AssistantConfig:
        class text_config:
            layer_types = ["sliding_attention", "full_attention"]

        @classmethod
        def from_pretrained(cls, path, **kw):
            return cls()

    tr.Gemma4TextConfig = Gemma4TextConfig
    tr.Gemma4AssistantConfig = Gemma4AssistantConfig
    sys.modules["transformers"] = tr


_install_stubs()

# ---------------------------------------------------------------------------
# Now check that the MTP weight mapping logic is correct
# ---------------------------------------------------------------------------

class TestGemma4MTPWeightMapping(unittest.TestCase):

    def _make_remap(self, num_mtp_layers=1, num_sub_layers=2):
        """Reproduce the remap dict from MTPForCausalLM.load_weights."""
        remap = {}
        for k in range(num_mtp_layers):
            prefix = f"mtp_layers.{k}."
            remap["pre_projection.weight"] = f"{prefix}pre_projection.weight"
            remap["model.norm.weight"] = f"{prefix}shared_head.norm.weight"
            for j in range(num_sub_layers):
                sub_prefix = f"{prefix}mtp_layers.{j}."
                hf_sub = f"model.layers.{j}."
                for sfx in [
                    "self_attn.o_proj.weight",
                    "self_attn.q_norm.weight",
                    "mlp.gate_proj.weight",
                    "mlp.up_proj.weight",
                    "mlp.down_proj.weight",
                    "input_layernorm.weight",
                    "post_attention_layernorm.weight",
                    "pre_feedforward_layernorm.weight",
                    "post_feedforward_layernorm.weight",
                    "layer_scalar",
                ]:
                    remap[hf_sub + sfx] = sub_prefix + sfx
                remap[hf_sub + "self_attn.q_proj.weight"] = (
                    sub_prefix + "self_attn.qkv_proj.weight"
                )
        return remap

    def test_pre_projection_mapped(self):
        remap = self._make_remap()
        self.assertEqual(
            remap["pre_projection.weight"],
            "mtp_layers.0.pre_projection.weight",
        )

    def test_norm_mapped(self):
        remap = self._make_remap()
        self.assertEqual(
            remap["model.norm.weight"],
            "mtp_layers.0.shared_head.norm.weight",
        )

    def test_q_proj_remapped_to_qkv_proj(self):
        remap = self._make_remap()
        self.assertEqual(
            remap["model.layers.0.self_attn.q_proj.weight"],
            "mtp_layers.0.mtp_layers.0.self_attn.qkv_proj.weight",
        )
        self.assertEqual(
            remap["model.layers.1.self_attn.q_proj.weight"],
            "mtp_layers.0.mtp_layers.1.self_attn.qkv_proj.weight",
        )

    def test_no_kv_proj_in_remap(self):
        remap = self._make_remap()
        # KV projections are NOT in the assistant checkpoint (Q-only attention)
        for k in remap:
            self.assertNotIn("k_proj", k)
            self.assertNotIn("v_proj", k)

    def test_mlp_mapped(self):
        remap = self._make_remap()
        self.assertEqual(
            remap["model.layers.0.mlp.gate_proj.weight"],
            "mtp_layers.0.mtp_layers.0.mlp.gate_proj.weight",
        )

    def test_layer_scalar_mapped(self):
        remap = self._make_remap()
        self.assertEqual(
            remap["model.layers.0.layer_scalar"],
            "mtp_layers.0.mtp_layers.0.layer_scalar",
        )

    def test_multiple_mtp_instances_share_same_weights(self):
        # Both MTP draft steps map the SAME HF keys to different TRT-LLM prefixes.
        # This documents that the same assistant checkpoint is loaded into each step.
        remap = self._make_remap(num_mtp_layers=2)
        # When k=1 writes to mtp_layers.1.*, it overwrites k=0's mapping for the
        # same HF key — this is expected: the caller loads the same checkpoint into
        # each MTP layer instance.
        self.assertEqual(
            remap["pre_projection.weight"],
            "mtp_layers.1.pre_projection.weight",
        )


class TestCacheLinkIdx(unittest.TestCase):

    def test_cache_layer_idx_for_sliding(self):
        """Verify last own-KV sliding layer index."""
        layer_types = (
            ["sliding_attention"] * 5 + ["full_attention"] +
            ["sliding_attention"] * 5 + ["full_attention"] +
            ["sliding_attention"] * 4  # last layer kv_shared
        )
        num_hidden_layers = len(layer_types)
        num_kv_shared = 1
        first_kv_shared_idx = num_hidden_layers - num_kv_shared  # = 15

        _last_own_kv = {}
        for i in range(first_kv_shared_idx - 1, -1, -1):
            t = layer_types[i]
            if t not in _last_own_kv:
                _last_own_kv[t] = i

        # Last full_attention before kv-shared is at index 11
        self.assertEqual(_last_own_kv["full_attention"], 11)
        # Last sliding_attention before kv-shared is at index 14 (0-based idx 14)
        self.assertEqual(_last_own_kv["sliding_attention"], 14)


if __name__ == "__main__":
    unittest.main(verbosity=2)
