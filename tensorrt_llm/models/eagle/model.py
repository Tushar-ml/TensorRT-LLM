# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ...models.llama.model import LLaMAForCausalLM
from .utils import *
from .layers import *
from transformers.configuration_utils import PretrainedConfig
from ...functional import ACT2FN, Tensor, softmax, expand_mask, matmul, unsqueeze, view, shape
from ...layers import ColumnLinear, Attention, AttentionMaskType
from ...layers.attention import make_causal_mask
from ...layers.normalization import RmsNorm
from ...layers.embedding import Embedding
from ...module import Module, ModuleList
from typing import Optional, Tuple, List
from ..._common import default_net
from transformers.models.llama.modeling_llama import repeat_kv
import math
from ...mapping import Mapping
import tensorrt as trt

class EAGLEConfig(PretrainedConfig):
    
    model_type = "llama"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=None,
            hidden_act="silu",
            max_position_embeddings=2048,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            pad_token_id=None,
            bos_token_id=1,
            eos_token_id=2,
            pretraining_tp=1,
            tie_word_embeddings=False,
            rope_scaling=None,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `name` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s name field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be an float > 1, got {rope_scaling_factor}")

class EagleAttention(Module):

    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads, max_position_embeddings,
                 rope_scaling, pretraining_tp):

        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_scaling = rope_scaling
        self.pretraining_tp = pretraining_tp

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        self.q_proj = ColumnLinear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = ColumnLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = ColumnLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = ColumnLinear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        if self.rope_scaling is None:
            self.rotary_emb = EAGLERotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.rope_scaling["type"]
            scaling_factor = self.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = EAGLELinearScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = EAGLEDynamicNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: Tensor,
            attention_mask: Optional[Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [ColumnLinear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
            query_states = concat(query_states, dim=-1)

            key_states = [ColumnLinear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
            key_states = concat(key_states, dim=-1)

            value_states = [ColumnLinear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
            value_states = concat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, (cos, sin), position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = concat([past_key_value[0], key_states], dim=2)
            value_states = concat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = softmax(attn_weights, dim=-1).to(query_states.dtype)
        attn_output = matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
            attn_output = sum([ColumnLinear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class EAGLEMLP(Module):
    def __init__(self, intermediate_size,
                 hidden_size, hidden_act, pretraining_tp):

        super().__init__()
        self.pretraining_tp = pretraining_tp
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.fc = ColumnLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.gate = ColumnLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.proj = ColumnLinear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        if self.pretraining_tp > 1:
            slice = self.intermediate_size // self.pretraining_tp
            gate_proj_slices = self.fc.weight.split(slice, dim=0)
            up_proj_slices = self.gate.weight.split(slice, dim=0)
            down_proj_slices = self.proj.weight.split(slice, dim=1)

            gate_proj = concat(
                [ColumnLinear(x, gate_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1
            )
            up_proj = concat([ColumnLinear(x, up_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                ColumnLinear(intermediate_states[i], down_proj_slices[i]) for i in range(self.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.proj(self.act_fn(self.fc(x)) * self.gate(x))


        return down_proj
      
class EAGLEDecoderLayer(Module):

    def __init__(self,index,
                 hidden_size, num_key_value_heads, num_attention_heads,
                 max_position_embeddings, num_hidden_layers,rope_scaling,
                 rms_norm_eps, intermediate_size, pretraining_tp, hidden_act):

        super().__init__()

        self.hidden_size = hidden_size
        # self.self_attn = EagleAttention(hidden_size, num_attention_heads,
        #                                 num_key_value_heads, max_position_embeddings, rope_scaling, pretraining_tp)
        self.self_attn = Attention(local_layer_idx=index, hidden_size=hidden_size,
                                   num_attention_heads=num_attention_heads, num_kv_heads=num_key_value_heads,
                                   max_position_embeddings=max_position_embeddings, num_layers=num_hidden_layers,
                                   rotary_embedding_scaling=rope_scaling, bias = False, attention_mask_type=AttentionMaskType.causal)
        
        self.mlp = EAGLEMLP(intermediate_size, hidden_size, hidden_act, pretraining_tp)
        self.index = index
        if self.index != 0:
            self.input_layernorm = RmsNorm(hidden_size, eps=rms_norm_eps)
        self.post_layernorm = RmsNorm(hidden_size, eps=rms_norm_eps)

    def forward(
                    self,
                    hidden_states: Tensor,
                    attention_mask: Optional[Tensor] = None,
                    position_ids: Optional[torch.LongTensor] = None,
                    past_key_value: Optional[Tuple[Tensor]] = None,
                    use_cache: Optional[bool] = False,
                    attention_params: Optional[Tensor] = None, kv_cache_params = None, **kwargs
            ):

        residual = hidden_states

        if self.index != 0:
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            attention_params = attention_params,
            kv_cache_params = kv_cache_params, use_cache = use_cache
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class EAGLEModel(Module):

    def __init__(
            self,vocab_size, hidden_size,
            num_hidden_layers,
            hidden_act,
            num_kv_heads,
            num_attention_heads,
            max_position_embeddings, rope_scaling, rms_norm_eps, intermediate_size,
            pretraining_tp,
            bias: bool =True
    ):
        super().__init__()
        self.gradient_checkpointing = False
        self.vocab_size = vocab_size
        self.embed_tokens = Embedding(vocab_size, hidden_size)

        self.layers = ModuleList([
            EAGLEDecoderLayer(index,hidden_size, num_kv_heads, num_attention_heads,
                              max_position_embeddings, num_hidden_layers, rope_scaling, rms_norm_eps,
                              intermediate_size, pretraining_tp, hidden_act) for index in range(1)
        ])

        self.fc = ColumnLinear(2*hidden_size,
                                    hidden_size,
                                    bias=bias)
        
        self.act = ACT2FN[hidden_act]

    def init_tree(self):
        self.tree_buffer = generate_tree_buffers_for_eagle(self.tree, "cuda")
    
    def reset(self):
        self.tree_mask = None

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = make_causal_mask(
                input_shape[0],
                input_shape[1],
                dtype=trt.DataType.FLOAT,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = expand_mask(attention_mask, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        # [MODIFIED] add tree mask
        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            tree_len = tree_mask.size(-1)
            bs=combined_attention_mask.size(0)
            combined_attention_mask[:, :, -tree_len:, -tree_len:][
                tree_mask.repeat(bs,1,1,1) == 0
                ] = torch.finfo(torch.float32).min

        return combined_attention_mask

    def forward(
            self,
            hidden_states,
            input_ids,
            attention_mask: Optional[Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            attention_params: Optional[Tensor] = None, **kwargs
    ):  
        
        batch_size, seq_length = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        with torch.no_grad():
            inputs_embeds = self.embed_tokens(input_ids)

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            device = "cpu"
            position_ids:Tensor = arange(
                past_key_values_length, seq_length + past_key_values_length, dtype="int64"
            )
            position_ids = view(unsqueeze(position_ids,0),(-1, seq_length))
        else:
            position_ids = view(position_ids, (-1, seq_length))

        print("Position IDs: ", position_ids)
        """
        python3 ../run.py --max_output_len=50 \
                  --tokenizer_dir /models/model_input/vicuna-7b-1.3 \
                  --engine_dir=/models/model_output/eagle_model_engine
        """
        # if attention_mask is None:
        #     attention_mask = Tensor("attention_mask",trt.DataType.INT32, (batch_size, seq_length_with_past))

        print("Attention Mask: ", attention_mask)
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (shape(hidden_states,0), shape(hidden_states,1)), hidden_states, past_key_values_length
        )

        print("Attention Mask: ", attention_mask)

        inputs_embeds = inputs_embeds
        hidden_states = self.fc(concat((inputs_embeds, hidden_states), dim=-1))

        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,attention_params = attention_params, kv_cache_params = kwargs["kv_cache_params"]
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        return hidden_states

    def reset_kv(self):
        self.stable_kv = None

    @torch.no_grad()
    def repeat_hidden(self, hidden_state, repeat_num):
        new_hidden = []
        for id, i in enumerate(repeat_num):
            new_hidden.append(hidden_state[:, id:id + 1].repeat(1, i, 1))
        return concat(new_hidden, dim=1)

    def sample(self, logits, logits_processor, k=1):
        bs, seq_len, _ = logits.shape
        logits = logits.view(-1, logits.shape[-1])
        logits = logits_processor(None, logits)
        probabilities = softmax(logits, dim=-1)
        sampled_indices = torch.multinomial(probabilities, k, replacement=False)
        sampled_probs = torch.gather(probabilities, -1, sampled_indices)
        cumulative_sum = torch.cumsum(sampled_probs, dim=-1)
        cumulative_sum = concat(
            (torch.zeros(cumulative_sum.shape[0], 1, device=cumulative_sum.device), cumulative_sum[:, :-1]), dim=-1)
        sampled_probs = sampled_probs / (1 - cumulative_sum)
        sampled_probs[torch.isinf(sampled_probs)] = -1
        sampled_probs[torch.isnan(sampled_probs)] = -1
        sampled_probs = torch.clamp(sampled_probs, min=0.0, max=1.0)
        sampled_indices = sampled_indices.view(bs, seq_len, -1)
        sampled_probs = sampled_probs.view(bs, seq_len, -1)
        probabilities = probabilities.view(bs, seq_len, -1)

        return sampled_indices, sampled_probs, probabilities

    @torch.no_grad()
    def topK_generate(self, hidden_states, input_ids, head, logits_processor, max_length=4, use_cache=True,
                     attention_mask=None, len_posi=None, ):
        top_k = 5
        bs = input_ids.shape[0]
        input_ids = input_ids[:, 1:]
        input_ids = input_ids.to(hidden_states.device)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.to(self.device)
        zero_num = position_ids.shape[1] - position_ids.max(dim=-1).values - 1
        zero_num = zero_num[:, None]
        ss_token, ss_prob, ss_op = [], [], []
        if len_posi is None:
            len_posi = input_ids.shape[1]
        self.reset()
        if use_cache:

            if hasattr(self, "stable_kv") and self.stable_kv is not None:
                kv_len = self.stable_kv[0][0].shape[2]
                position_ids = position_ids[:, kv_len:]
                out_hidden, past_key_values = self(hidden_states, input_ids=input_ids, past_key_values=self.stable_kv,
                                                   use_cache=True, attention_mask=attention_mask,
                                                   position_ids=position_ids)


            else:
                out_hidden, past_key_values = self(hidden_states, input_ids=input_ids, use_cache=True,
                                                   attention_mask=attention_mask, position_ids=position_ids)
            self.stable_kv = past_key_values
            last_nopadding = position_ids.argmax(dim=-1)
            ab = tuple(range(bs))
            last_hidden = out_hidden[ab, last_nopadding][:, None]
            if not self.diff_device:
                last_headout = head(last_hidden)
            else:
                if hasattr(self, "layer_device"):
                    last_headout = head(last_hidden)
                    last_headout = last_headout.to(self.layer_device)
                else:
                    last_headout = ColumnLinear(last_hidden, self.headweight)
            
            for i in range(len(self.tree_buffer['tree_indices'])):
                if logits_processor is not None:
                    topk_index, topk_prob, op = self.sample(last_headout, logits_processor, k=top_k, )
                else:
                    topk_index, topk_prob = torch.topk(last_headout, top_k, dim=-1).indices, torch.topk(last_headout,
                                                                                                        top_k,
                                                                                                        dim=-1).values
                    op = None

                ss_token.append(topk_index)
                ss_prob.append(topk_prob)
                ss_op.append(op)

                input_ids = topk_index.view(bs, -1)[:, self.tree_buffer['tree_indices'][i]]

                attention_mask = concat((attention_mask, torch.ones_like(input_ids, device=attention_mask.device,
                                                                            dtype=attention_mask.dtype)), dim=1)

                if i == 0:
                    hidden_states = last_hidden
                else:
                    hidden_states = out_hidden
                hidden_states = self.repeat_hidden(hidden_states, self.tree_buffer["repeat_nums"][i])
                self.tree_mask = self.tree_buffer['attn_mask'][i]
                position_ids = len_posi + self.tree_buffer["position_ids"][i][None, :] - zero_num
                out_hidden, past_key_values = self(hidden_states, input_ids=input_ids, past_key_values=past_key_values,
                                                   position_ids=position_ids, use_cache=True,
                                                   attention_mask=attention_mask)
                len_posi += 1

                if not self.diff_device:
                    last_headout = head(out_hidden)
                else:
                    if hasattr(self, "layer_device"):
                        last_headout = head(out_hidden)
                        last_headout = last_headout.to(self.layer_device)
                    else:
                        last_headout = ColumnLinear(out_hidden[0], self.headweight)

            if logits_processor is not None:
                topk_index, topk_prob, op = self.sample(last_headout, logits_processor, k=top_k, )
            else:
                topk_index, topk_prob = torch.topk(last_headout, top_k, dim=-1).indices, torch.topk(last_headout, top_k,
                                                                                                    dim=-1).values
                op = None
            ss_token.append(topk_index)
            ss_prob.append(topk_prob)
            ss_op.append(op)

        else:
            # TODO
            pass

        return (concat(ss_token, dim=1), concat(ss_prob, dim=1), ss_op)


class EagleForCausalLM(LLaMAForCausalLM):

    def __init__(self, config, mapping = Mapping()):
        
        super().__init__(config)

        bias = False
        self.ea_layer = EAGLEModel(config.vocab_size, config.hidden_size,
                                   config.num_hidden_layers, config.hidden_act,
                                   config.num_key_value_heads, config.num_attention_heads,
                                   config.max_position_embeddings, config.rope_scaling, config.rms_norm_eps, config.intermediate_size,
                                   config.pretraining_tp)
        device = "cuda"

        self.ea_layer.device = device
        self.ea_layer.diff_device = False
        self.ea_layer.tree = tree_structure
        self.tree = tree_structure

        self.ea_layer.init_tree()
        self.lm_head = ColumnLinear(config.hidden_size,
                                    config.vocab_size,
                                    bias=False,
                                    dtype=config.dtype,
                                    tp_group=mapping.tp_group,
                                    tp_size=mapping.tp_size,
                                    gather_output=True)
        
    def forward(self, *args, **kwargs):
        output_original = True
        hidden_states = super().forward(*args, **kwargs)
        print(args, kwargs)
        if kwargs['use_cache']:
            if default_net().plugin_config.paged_kv_cache:
                lm_logits, hidden_states = hidden_states
            else:
                lm_logits, presents, hidden_states = hidden_states

        if self.mapping.is_last_pp_rank():
            
            eagle_logits: Tensor = self.ea_layer(hidden_states, **kwargs)
            eagle_logits.mark_output('eagle_logits', self.config.logits_dtype)
        else:
            hidden_states.mark_output('hidden_states_output', self.config.dtype)

        print("Eagle Logits: ", eagle_logits, "Hidden States: ", hidden_states)
        if kwargs['use_cache'] and default_net(
        ).plugin_config.paged_kv_cache == False:
            if self.mapping.is_last_pp_rank():
                if output_original:
                    return (eagle_logits, lm_logits, presents)
                return (eagle_logits, presents)
            return (hidden_states, presents)
        else:
            if self.mapping.is_last_pp_rank():
                if output_original:
                    return eagle_logits, lm_logits
                return eagle_logits
            return hidden_states
    
if __name__ == "__main__":

    from transformers import AutoTokenizer, AutoModelForCausalLM

    text = "Hi how are you"
    tokenizer = AutoTokenizer.from_pretrained("/models/model_input/vicuna-7b-v1.3")
    model = EagleForCausalLM.from_checkpoint("/models/model_output/eagle_model")

    input_id = tokenizer(text, return_tensors="pt")

    print(model(**input_id))