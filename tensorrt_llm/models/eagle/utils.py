import torch
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
import copy
from tensorrt_llm.layers.attention import make_causal_mask
from tensorrt_llm.functional import expand_mask
from typing import Optional
import random
from transformers.cache_utils import Cache, DynamicCache
from tensorrt_llm.functional import RopeEmbeddingUtils

tree_structure = [[0], [1], [2], [3], [0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [2, 0], [2, 1], [3, 0]
    , [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 2, 0], [0, 2, 1], [1, 0, 0],
                  [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1]]

chain_structure = [[0], [0, 0], [0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0, 0]]
        
class node:
    def __init__(self, parent=None, value=None, dict_key=None):
        self.parent = parent
        self.value = value
        if parent:
            self.depth = parent.depth + 1
            parent.children.append(self)
        else:
            self.depth = 0
        self.children = []
        self.dict_key = dict_key

    def is_leaf(self):
        return len(self.children) == 0

    def all_index(self):
        if not self.parent.parent:
            return [self.index]
        else:
            return self.parent.all_index() + [self.index]

class Tree:
    def __init__(self, tree_list):
        sorted_tree_list = sorted(tree_list, key=lambda x: (len(x), x))
        self.root = node()
        self.node_dic = {}
        for tree_node in sorted_tree_list:
            cur_value = tree_node[-1]
            if len(tree_node) == 1:
                cur_node = node(parent=self.root, value=cur_value, dict_key=tuple(tree_node))
            else:
                cur_parent = self.node_dic[tuple(tree_node[:-1])]
                cur_node = node(parent=cur_parent, value=cur_value, dict_key=tuple(tree_node))
            self.node_dic[tuple(tree_node)] = cur_node
        self.indexnode()

    def max_depth(self):
        return max([item.depth for item in self.node_dic.values()])

    def num_node_wchild(self):
        num_c = 0
        for item in self.node_dic.values():
            if not item.is_leaf():
                num_c += 1
        return num_c

    def get_node_wchild(self):
        ns = []
        for item in self.node_dic.values():
            if not item.is_leaf():
                ns.append(item)
        return ns

    def indexnode(self):
        cur_index = 0
        for key in self.node_dic:
            cur_node = self.node_dic[key]
            if not cur_node.is_leaf():
                cur_node.index = cur_index
                cur_index += 1

def generate_tree_buffers_for_eagle(tree_choices):
    TOPK = 10
    tree = Tree(tree_choices)
    tree_len = tree.num_node_wchild()

    max_depth = tree.max_depth()
    nodes_wc = tree.get_node_wchild()

    depth_counts = [0 for _ in range(max_depth - 1)]
    for x in nodes_wc:
        depth_counts[x.depth - 1] += 1
    depth_counts_sum = [sum(depth_counts[:i + 1]) for i in range(len(depth_counts))]

    tree_attn_mask = torch.eye(tree_len, tree_len)

    for id, x in enumerate(nodes_wc):
        tree_attn_mask[id, x.all_index()] = 1

    tree_attn_mask_list0 = [tree_attn_mask[:ml, :ml] for ml in depth_counts_sum]
    tree_attn_mask_list = []
    for id, x in enumerate(tree_attn_mask_list0):
        x = x[-depth_counts[id]:]
        tree_attn_mask_list.append(x)

    tree_indices_list = [torch.zeros(ml, dtype=torch.long) for ml in depth_counts]
    repeat_nums = [[] for _ in depth_counts]
    start = 0
    bias = 0
    for i in range(len(depth_counts)):
        bias = 0
        repeat_j = 0
        for j in range(depth_counts[i]):
            cur_node = nodes_wc[start + j]
            cur_parent = cur_node.parent

            if j != 0:
                if cur_parent != parent:
                    bias += 1
                    parent = cur_parent
                    repeat_nums[i].append(j - repeat_j)
                    repeat_j = j
            else:
                parent = cur_parent
            tree_indices_list[i][j] = cur_node.value + TOPK * (bias)
        repeat_nums[i].append(j - repeat_j + 1)
        start += depth_counts[i]

    position_ids = [torch.zeros(ml, dtype=torch.long) for ml in depth_counts]

    tree_buffers = {
        "attn_mask": [i.unsqueeze(0).unsqueeze(0) for i in tree_attn_mask_list],
        "tree_indices": tree_indices_list,
        "position_ids": position_ids,
        "repeat_nums": repeat_nums
    }

    # Move the tensors in the dictionary to the specified device
    tree_buffers = {
        k: [i.clone().numpy().tolist() for i in v]
        if isinstance(v[0], torch.Tensor)
        else (
            v.numpy().tolist()
            if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in tree_buffers.items()
    }
    return tree_buffers


def prepare_logits_processor(
        temperature=0.0, repetition_penalty=0.0, top_p=0.0, top_k=0
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


def pad_path(path, length, pad_value=-2):
    """
    Pad the given path list with a specific value up to a specified length.

    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.

    Returns:
    - list: A new list based on the original path but padded to the desired length.

    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]

    Note:
    If the given path is already longer than the specified length,
    then no padding occurs, and the original path is returned.
    """

    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))


def generate_tree_buffers(tree_choices, device="cuda"):
    TOPK = 5
    sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
    tree_len = len(sorted_tree_choices) + 1

    # Initialize depth_counts to keep track of how many choices have a particular depth
    depth_counts = []
    prev_depth = 0
    for path in sorted_tree_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth

    tree_attn_mask = torch.eye(tree_len, tree_len)
    tree_attn_mask[:, 0] = 1
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_tree_choice = sorted_tree_choices[start + j]
            # retrieve ancestor position
            if len(cur_tree_choice) == 1:
                continue
            ancestor_idx = []
            for c in range(len(cur_tree_choice) - 1):
                ancestor_idx.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]) + 1)
            tree_attn_mask[j + start + 1, ancestor_idx] = 1
        start += depth_counts[i]

    tree_indices = torch.zeros(tree_len, dtype=torch.long)
    p_indices = [0 for _ in range(tree_len - 1)]
    b_indices = [[] for _ in range(tree_len - 1)]
    tree_indices[0] = 0
    start = 0
    bias = 0
    for i in range(len(depth_counts)):
        inlayer_bias = 0
        b = []
        for j in range(depth_counts[i]):
            cur_tree_choice = sorted_tree_choices[start + j]
            cur_parent = cur_tree_choice[:-1]
            if j != 0:
                if cur_parent != parent:
                    bias += 1
                    inlayer_bias += 1
                    parent = cur_parent
                    b = []
            else:
                parent = cur_parent
            tree_indices[start + j + 1] = cur_tree_choice[-1] + TOPK * (i + bias) + 1
            p_indices[start + j] = inlayer_bias
            if len(b) > 0:
                b_indices[start + j] = copy.deepcopy(b)
            else:
                b_indices[start + j] = []
            b.append(cur_tree_choice[-1] + TOPK * (i + bias) + 1)
        start += depth_counts[i]

    p_indices = [-1] + p_indices
    tree_position_ids = torch.zeros(tree_len, dtype=torch.long)
    start = 0
    for i in range(len(depth_counts)):
        tree_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
        start += depth_counts[i]

    retrieve_indices_nest = []
    retrieve_paths = []
    for i in range(len(sorted_tree_choices)):
        cur_tree_choice = sorted_tree_choices[-i - 1]
        retrieve_indice = []
        if cur_tree_choice in retrieve_paths:
            continue
        else:
            for c in range(len(cur_tree_choice)):
                retrieve_indice.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]))
                retrieve_paths.append(cur_tree_choice[:c + 1])
        retrieve_indices_nest.append(retrieve_indice)
    max_length = max([len(x) for x in retrieve_indices_nest])
    retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    retrieve_indices = retrieve_indices + 1
    retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices],
                                 dim=1)

    maxitem = retrieve_indices.max().item() + 5

    def custom_sort(lst):
        # sort_keys=[len(list)]
        sort_keys = []
        for i in range(len(lst)):
            sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
        return sort_keys

    retrieve_indices = retrieve_indices.tolist()
    retrieve_indices = sorted(retrieve_indices, key=custom_sort)
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)

    p_indices = torch.tensor(p_indices)
    p_indices_new = p_indices[retrieve_indices]
    p_indices_new = p_indices_new.tolist()

    b_indices = [[]] + b_indices
    b_indices_new = []
    for ib in range(retrieve_indices.shape[0]):
        iblist = []
        for jb in range(retrieve_indices.shape[1]):
            index = retrieve_indices[ib, jb]
            if index == -1:
                iblist.append([])
            else:
                b = b_indices[index]
                if len(b) > 0:
                    bt = []
                    for bi in b:
                        bt.append(torch.where(tree_indices == bi)[0].item())
                    iblist.append(torch.tensor(bt, device=device))
                else:
                    iblist.append(b)
        b_indices_new.append(iblist)

    # Aggregate the generated buffers into a dictionary
    tree_buffers = {
        "tree_attn_mask": tree_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": tree_indices,
        "tree_position_ids": tree_position_ids,
        "retrieve_indices": retrieve_indices,
    }

    # Move the tensors in the dictionary to the specified device
    tree_buffers = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v, device=device)
        for k, v in tree_buffers.items()
    }
    tree_buffers["p_indices"] = p_indices_new
    tree_buffers["b_indices"] = b_indices_new
    return tree_buffers


def _prepare_decoder_attention_mask(
        attention_mask, tree_mask, input_shape, inputs_embeds, past_key_values_length
):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = make_causal_mask(
            input_shape,
            torch.float16,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        expanded_attn_mask = expand_mask(
            attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
        ).to(inputs_embeds.device)
        combined_attention_mask = (
            expanded_attn_mask
            if combined_attention_mask is None
            else expanded_attn_mask + combined_attention_mask
        )

    if tree_mask is not None:
        tree_len = tree_mask.size(-1)
        bs = combined_attention_mask.size(0)
        combined_attention_mask[:, :, -tree_len:, -tree_len:][
            tree_mask.repeat(bs, 1, 1, 1) == 0
            ] = combined_attention_mask.min()

    return combined_attention_mask


def initialize_tree(input_ids, model, logits_processor, attention_mask=None):
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    hidden_states, past_key_value = forward_with_tree_mask(model.base_model.model, input_ids=input_ids,
                                                           attention_mask=attention_mask, position_ids=position_ids)
    logits=model.base_model.lm_head(hidden_states)

    if logits_processor is not None:
        sample_logits = logits[:, -1]
        sample_logits = logits_processor(None, sample_logits)
        probabilities = torch.nn.functional.softmax(sample_logits, dim=-1)
        token = torch.multinomial(probabilities, 1)
    else:
        token = torch.argmax(logits[:, -1], dim=-1)
        token = token[:, None]
    input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)

    tree_logits = model.ea_layer.topK_genrate(hidden_states, input_ids, model.base_model.lm_head, logits_processor,
                                           attention_mask=attention_mask)



    return tree_logits, logits, hidden_states, token,past_key_value

def generate_candidates(tree_logits, tree_indices, retrieve_indices, sample_token, logits_processor):
    bs = sample_token.shape[0]
    sample_token = sample_token.to(tree_indices.device)

    # candidates_logit = sample_token[0]
    candidates_logit = sample_token

    candidates_tree_logits = tree_logits[0]

    candidates = torch.cat([candidates_logit, candidates_tree_logits.view(bs, -1)], dim=-1)

    tree_candidates = candidates[:, tree_indices]

    tree_candidates_ext = torch.cat(
        [tree_candidates, torch.zeros((bs, 1), dtype=torch.long, device=tree_candidates.device)-1], dim=-1)

    cart_candidates = tree_candidates_ext[:, retrieve_indices]

    if logits_processor is not None:
        candidates_tree_prob = tree_logits[1]
        candidates_prob = torch.cat(
            [torch.ones((bs, 1), device=candidates_tree_prob.device, dtype=torch.float16),
             candidates_tree_prob.view(bs, -1)],
            dim=-1)

        tree_candidates_prob = candidates_prob[:, tree_indices]
        tree_candidates_prob_ext = torch.cat(
            [tree_candidates_prob, torch.ones((bs, 1), dtype=torch.float16, device=tree_candidates_prob.device)],
            dim=-1)
        cart_candidates_prob = tree_candidates_prob_ext[:, retrieve_indices]
    else:
        cart_candidates_prob = None
    # Unsqueeze the tree candidates for dimension consistency.

    #
    # tree_candidates = tree_candidates.unsqueeze(0)
    return cart_candidates, cart_candidates_prob, tree_candidates


def tree_decoding(
        model,
        tree_candidates,
        past_key_values,
        tree_position_ids,
        input_ids,
        retrieve_indices,
        attention_mask=None,
        tree_mask=None,
):

    zero_num = attention_mask.shape[1]-attention_mask.long().sum(-1)
    zero_num = zero_num[:, None]
    position_ids = tree_position_ids[None,:] + input_ids.shape[1]-zero_num


    attention_mask = torch.cat(
        (attention_mask, torch.ones_like(tree_candidates, device=attention_mask.device, dtype=attention_mask.dtype)), dim=1)

    hidden_states, past_key_value = forward_with_tree_mask(model.base_model.model, input_ids=tree_candidates,past_key_values=past_key_values,
                                                           attention_mask=attention_mask, tree_mask=tree_mask,position_ids=position_ids)

    tree_logits = model.base_model.lm_head(hidden_states)




    logits = tree_logits[:, retrieve_indices]
    return logits, hidden_states,past_key_value


def evaluate_posterior(
        logits, candidates, logits_processor, cart_candidates_prob, op, p_indices, tree_candidates, b_indices,
        finish_flag
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
    - posterior_threshold (float): Threshold for posterior probability.
    - posterior_alpha (float): Scaling factor for the threshold.

    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    # Greedy decoding based on temperature value
    if logits_processor is None:
        bs = tree_candidates.size(0)
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (
                candidates[:, :, 1:].to(logits.device) == torch.argmax(logits[:, :, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=-1)).sum(dim=-1)
        accept_length = candidates_accept_length.max(dim=1).values
        best_candidate = torch.argmax(candidates_accept_length, dim=-1).to(torch.long)


        bt = tuple(range(bs))
        logits_batch = logits[bt, best_candidate, accept_length, :]
        accept_length = accept_length.tolist()

        for batch in range(bs):
            if finish_flag[batch]:
                accept_length[batch] = 0

        return best_candidate.tolist(), accept_length, logits_batch

    else:
        cart_candidates_prob = cart_candidates_prob.to(logits.device)
        bs = cart_candidates_prob.size(0)

        logits = logits_processor(None, logits)
        probs = torch.softmax(logits, dim=-1)

        best_candidate_list = []
        accept_length_list = []
        sample_p_list = []

        for batch in range(bs):
            accept_length = 1
            accept_cand = candidates[batch, 0, :1]
            best_candidate = 0
            for i in range(1, candidates.shape[2]):
                if i != accept_length:
                    break
                adjustflag = False
                is_eq = (candidates[batch, :, :accept_length] == accept_cand).all(dim=1)
                fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
                gtp = probs[batch, fi, i - 1]
                candidates_set = []
                for j in range(candidates.shape[1]):
                    if is_eq[j]:
                        x = candidates[batch, j, i]
                        xi = x.item()
                        if xi in candidates_set or xi == -1:
                            continue
                        candidates_set.append(xi)
                        r = random.random()
                        px = gtp[xi]
                        qx = cart_candidates_prob[batch, j, i]
                        if qx <= 0:
                            continue
                        acp = px / qx
                        if r <= acp:
                            accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                            accept_length += 1
                            best_candidate = j
                            break
                        else:
                            q = op[i - 1][batch][p_indices[j][i]].clone()
                            b = b_indices[j][i]
                            if len(b) > 0:
                                mask = tree_candidates[batch][b]
                                q[mask] = 0
                                q = q / q.sum()
                            gtp = gtp - q
                            gtp[gtp < 0] = 0
                            gtp = gtp / gtp.sum()
                            adjustflag = True
            if adjustflag and accept_length != candidates.shape[1]:
                sample_p = gtp
            else:
                sample_p = probs[batch, best_candidate, accept_length - 1]
            best_candidate_list.append(best_candidate)
            accept_length_list.append(accept_length - 1)
            sample_p_list.append(sample_p)

        for batch in range(bs):
            if finish_flag[batch]:
                accept_length_list[batch] = 0

        return best_candidate_list, accept_length_list, sample_p_list


@torch.no_grad()
def update_inference_inputs(
        input_ids,
        attention_mask,
        candidates,
        best_candidate,
        accept_length,
        retrieve_indices,
        logits_processor,
        new_token,
        past_key_values,
        model,
        hidden_state_new,
        sample_p,
        finish_flag

):

    new_outs=[]
    finish_flag=copy.deepcopy(finish_flag)
    bs=len(best_candidate)
    prev_input_len = input_ids.shape[1]
    max_acccept_len=max(accept_length)

    retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices[0]]

    ab=tuple(range(bs))
    select_indices = (
            retrieve_indices.cpu()[ab,best_candidate, : max_acccept_len + 1,...] + prev_input_len
    )
    #select_indices=select_indices.cpu()
    new_input_ids=candidates[ab, best_candidate, : max_acccept_len + 1,...]

    draft_hidden = retrieve_hidden_state_new[ab, best_candidate, :max_acccept_len + 1]

    new_attention_mask = torch.zeros((bs,max_acccept_len+1),dtype=torch.long)


    for batch in range(bs):
        new_attention_mask[batch,:accept_length[batch]+1]=1
        new_o=new_input_ids[batch,: accept_length[batch] + 1].tolist()
        new_outs.append(new_o)
        if model.base_model.config.eos_token_id in new_o:
            finish_flag[batch]=True
        new_token[batch] += accept_length[batch] + 1

    attention_mask = torch.cat((attention_mask, new_attention_mask.to(attention_mask.device)), dim=1)

    batch_dim_indices=torch.tensor(ab)[:,None].expand(-1,max_acccept_len + 1)

    new_kv=()

    for past_key_values_data in past_key_values:
        layer_kv=()
        for korv in past_key_values_data:
            tgt = korv[batch_dim_indices, :, select_indices, :]
            tgt = tgt.permute(0, 2, 1, 3)
            dst = korv[:, :, prev_input_len: prev_input_len + tgt.shape[-2], :]
            dst.copy_(tgt, non_blocking=True)
            layer_kv+=(korv[:, :, : prev_input_len + tgt.shape[-2], :],)
        new_kv+=(layer_kv,)

    input_ids=torch.cat((input_ids,new_input_ids.to(input_ids.device)),dim=1)





    prob = sample_p
    if isinstance(prob,list):
        prob=torch.stack(prob)
    if logits_processor is not None:
        token = torch.multinomial(prob, 1)
    else:
        token = torch.argmax(prob,dim=-1)
        token = token[:,None]

    draft_input_ids=torch.cat((new_input_ids, torch.ones(bs, 1, dtype=torch.long, device=new_input_ids.device)),dim=1)
    token_=token[:,0]

    draft_input_ids[ab,torch.tensor(accept_length,dtype=torch.long)+1]=token_



    tree_logits = model.ea_layer.topK_genrate(draft_hidden,
                                              input_ids=draft_input_ids,
                                              head=model.base_model.lm_head, logits_processor=logits_processor,attention_mask=attention_mask,len_posi=input_ids.shape[1])



    return input_ids, tree_logits, new_token, None, token,attention_mask,finish_flag,new_outs,new_kv

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (RopeEmbeddingUtils.rotate_half(q) * sin)
    k_embed = (k * cos) + (RopeEmbeddingUtils.rotate_half(k) * sin)
    return q_embed, k_embed