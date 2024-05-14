from argparse import Namespace
import copy
import torch

TOPK = 5
# def pad_path(path, length, pad_value=-2):

#     return path + [pad_value] * (length - len(path))

# def get_eagle_topks(num_eagle_heads, paths):
#     eagle_topks = [0] * num_eagle_heads
#     print(paths, num_eagle_heads)
#     for p in paths:
#         for i, k in enumerate(p):
#             eagle_topks[i] = max(eagle_topks[i], k + 1)
#     return eagle_topks

# class node:
#     def __init__(self,parent=None,value=None,dict_key=None):
#         self.parent=parent
#         self.value=value
#         if parent:
#             self.depth=parent.depth+1
#             parent.children.append(self)
#         else:
#             self.depth=0
#         self.children=[]
#         self.dict_key=dict_key
#     def is_leaf(self):
#         return len(self.children)==0

#     def all_index(self):
#         if not self.parent.parent:
#             return [self.index]
#         else:
#             return self.parent.all_index()+[self.index]



# class Tree:
#     def __init__(self,tree_list):
#         sorted_tree_list = sorted(tree_list, key=lambda x: (len(x), x))
#         self.root=node()
#         self.node_dic={}
#         for tree_node in sorted_tree_list:
#             cur_value=tree_node[-1]
#             if len(tree_node)==1:
#                 cur_node=node(parent=self.root,value=cur_value,dict_key=tuple(tree_node))
#             else:
#                 cur_parent=self.node_dic[tuple(tree_node[:-1])]
#                 cur_node = node(parent=cur_parent, value=cur_value,dict_key=tuple(tree_node))
#             self.node_dic[tuple(tree_node)] = cur_node
#         self.indexnode()

#     def max_depth(self):
#         return max([item.depth for item in self.node_dic.values()])

#     def num_node_wchild(self):
#         num_c=0
#         for item in self.node_dic.values():
#             if not item.is_leaf():
#                 num_c+=1
#         return num_c

#     def get_node_wchild(self):
#         ns=[]
#         for item in self.node_dic.values():
#             if not item.is_leaf():
#                 ns.append(item)
#         return ns

#     def indexnode(self):
#         cur_index=0
#         for key in self.node_dic:
#             cur_node=self.node_dic[key]
#             if not cur_node.is_leaf():
#                 cur_node.index=cur_index
#                 cur_index+=1

# def generate_tree_buffers(tree_choices):
#     tree=Tree(tree_choices)
#     sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
#     tree_len = tree.num_node_wchild()


#     max_depth=tree.max_depth()
#     nodes_wc=tree.get_node_wchild()

#     depth_counts=[0 for _ in range(max_depth-1)]
#     for x in nodes_wc:
#         depth_counts[x.depth-1]+=1
#     depth_counts_sum = [sum(depth_counts[:i + 1]) for i in range(len(depth_counts))]


#     tree_attn_mask = torch.eye(tree_len, tree_len)

#     for id,x in enumerate(nodes_wc):
#         tree_attn_mask[id,x.all_index()]=1




#     tree_attn_mask_list0=[tree_attn_mask[:ml,:ml] for ml in depth_counts_sum]
#     tree_attn_mask_list=[]
#     for id,x in enumerate(tree_attn_mask_list0):
#         x=x[-depth_counts[id]:]
#         tree_attn_mask_list.append(x)



#     tree_indices_list = [torch.zeros(ml, dtype=torch.long) for ml in depth_counts]
#     repeat_nums=[[] for _ in depth_counts]
#     start = 0
#     bias = 0
#     for i in range(len(depth_counts)):
#         bias = 0
#         repeat_j=0
#         for j in range(depth_counts[i]):
#             cur_node = nodes_wc[start + j]
#             cur_parent = cur_node.parent

#             if j != 0:
#                 if cur_parent != parent:
#                     bias += 1
#                     parent = cur_parent
#                     repeat_nums[i].append(j-repeat_j)
#                     repeat_j=j
#             else:
#                 parent = cur_parent
#             tree_indices_list[i][j] = cur_node.value + TOPK * (bias)
#         repeat_nums[i].append(j - repeat_j+1)
#         start += depth_counts[i]

#     position_ids = [torch.zeros(ml, dtype=torch.long) for ml in depth_counts]

#     # start = 0
#     # for i in range(len(depth_counts)):
#     #     position_ids[start: start + depth_counts[i]] = i
#     #     start += depth_counts[i]

#     tree_buffers = {
#         "attn_mask": [i.unsqueeze(0).unsqueeze(0) for i in tree_attn_mask_list],
#         "tree_indices": tree_indices_list,
#         "position_ids":position_ids,
#         "repeat_nums":repeat_nums
#     }

#     print("Tree Buffers: ", tree_buffers)
#     return tree_buffers

# def _eagle_setup(choices_or_paths):
#     generate_tree_buffers(choices_or_paths)
#     tree_choices = copy.deepcopy(choices_or_paths)
#     sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
#     tree_len = len(sorted_tree_choices) + 1

#     num_eagle_heads = max([len(c) for c in sorted_tree_choices])
#     depth_counts = []
#     prev_depth = 0
#     for path in sorted_tree_choices:
#         depth = len(path)
#         if depth != prev_depth:
#             depth_counts.append(0)
#         depth_counts[depth - 1] += 1
#         prev_depth = depth
    
#     tree_attn_mask = torch.eye(tree_len, tree_len)
#     tree_attn_mask[:, 0] = 1
#     start = 0
#     for i in range(len(depth_counts)):
#         for j in range(depth_counts[i]):
#             cur_tree_choice = sorted_tree_choices[start + j]
#             # retrieve ancestor position
#             if len(cur_tree_choice) == 1:
#                 continue
#             ancestor_idx = []
#             for c in range(len(cur_tree_choice) - 1):
#                 ancestor_idx.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]) + 1)
#             tree_attn_mask[j + start + 1, ancestor_idx] = 1
#         start += depth_counts[i]
    
#     tree_indices = torch.zeros(tree_len, dtype=torch.long)
#     p_indices = [0 for _ in range(tree_len - 1)]
#     b_indices = [[] for _ in range(tree_len - 1)]
#     tree_indices[0] = 0
#     start = 0
#     bias = 0
#     for i in range(len(depth_counts)):
#         inlayer_bias = 0
#         b = []
#         for j in range(depth_counts[i]):
#             cur_tree_choice = sorted_tree_choices[start + j]
#             cur_parent = cur_tree_choice[:-1]
#             if j != 0:
#                 if cur_parent != parent:
#                     bias += 1
#                     inlayer_bias += 1
#                     parent = cur_parent
#                     b = []
#             else:
#                 parent = cur_parent
#             tree_indices[start + j + 1] = cur_tree_choice[-1] + TOPK * (i + bias) + 1
#             p_indices[start + j] = inlayer_bias
#             if len(b) > 0:
#                 b_indices[start + j] = copy.deepcopy(b)
#             else:
#                 b_indices[start + j] = []
#             b.append(cur_tree_choice[-1] + TOPK * (i + bias) + 1)
#         start += depth_counts[i]
    
#     p_indices = [-1] + p_indices
#     tree_position_ids = torch.zeros(tree_len, dtype=torch.long)
#     start = 0
#     for i in range(len(depth_counts)):
#         tree_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
#         start += depth_counts[i]
    
#     retrieve_indices_nest = []
#     retrieve_paths = []
#     for i in range(len(sorted_tree_choices)):
#         cur_tree_choice = sorted_tree_choices[-i - 1]
#         retrieve_indice = []
#         if cur_tree_choice in retrieve_paths:
#             continue
#         else:
#             for c in range(len(cur_tree_choice)):
#                 retrieve_indice.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]))
#                 retrieve_paths.append(cur_tree_choice[:c + 1])
#         retrieve_indices_nest.append(retrieve_indice)

#     max_length = max([len(x) for x in retrieve_indices_nest])
#     retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
#     retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
#     retrieve_indices = retrieve_indices + 1
#     retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices],
#                                  dim=1)

#     maxitem = retrieve_indices.max().item() + 5

#     def custom_sort(lst):
#         # sort_keys=[len(list)]
#         sort_keys = []
#         for i in range(len(lst)):
#             sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
#         return sort_keys

#     retrieve_indices = retrieve_indices.tolist()
#     retrieve_indices = sorted(retrieve_indices, key=custom_sort)
#     retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)

#     p_indices = torch.tensor(p_indices)
#     p_indices_new = p_indices[retrieve_indices]
#     p_indices_new = p_indices_new.tolist()

#     b_indices = [[]] + b_indices
#     b_indices_new = []
#     for ib in range(retrieve_indices.shape[0]):
#         iblist = []
#         for jb in range(retrieve_indices.shape[1]):
#             index = retrieve_indices[ib, jb]
#             if index == -1:
#                 iblist.append([])
#             else:
#                 b = b_indices[index]
#                 if len(b) > 0:
#                     bt = []
#                     for bi in b:
#                         bt.append(torch.where(tree_indices == bi)[0].item())
#                     iblist.append(torch.tensor(bt))
#                 else:
#                     iblist.append(b)
#         b_indices_new.append(iblist)
    
#     tree_buffers = {
#         "eagle_mask": tree_attn_mask,
#         "eagle_tree_ids": tree_indices,
#         "eagle_position_offsets": tree_position_ids,
#         "eagle_paths": retrieve_indices,
#         "num_eagle_heads": num_eagle_heads,
#         "eagle_topks": torch.tensor([TOPK])
#     }

#     return Namespace(**tree_buffers)

import copy
from argparse import Namespace
from functools import cmp_to_key
from typing import List

import numpy as np
import torch

from tensorrt_llm.logger import logger


def path_sorter(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return -1 if a[i] < b[i] else 1
    return 0  # shouldn't reach


path_sorting_key = cmp_to_key(path_sorter)


def expand_choices_if_needed(medusa_choices: List[List[int]]):
    """
    Do a simple check to see if the given choices are path-only or vanilla.
    """
    assert len(medusa_choices) > 0
    for c in medusa_choices:
        if len(c) > 1:
            try:
                _ = medusa_choices.index(
                    [c[0]])  # find the first parent of current path
                logger.debug(
                    "Detected vanilla-style of Medusa choices. No need to expand."
                )
                return medusa_choices  # if found, just return assuming it is already expanded
            except ValueError:
                logger.debug(
                    "Detected path-only style of Medusa choices. Expanding ...")
                break
    expanded_choices = set()
    for c in medusa_choices:
        cur = ()
        for n in c:
            cur = (*cur, n)
            expanded_choices.add(cur)
    expanded_choices = [list(c) for c in expanded_choices]
    return expanded_choices


def get_packed_mask(num_medusa_tokens, medusa_mask):
    num_packed_masks = (num_medusa_tokens + 1 + 32 - 1) // 32
    medusa_packed_mask = torch.zeros((num_medusa_tokens + 1, num_packed_masks),
                                     dtype=torch.int32)
    for token_idx in range(num_medusa_tokens + 1):
        if token_idx == 0:
            medusa_packed_mask[0, 0] = 1
        else:
            mask_list = medusa_mask[token_idx - 1, :].tolist()
            # insert 1 as there is one extra new token from the original lm head.
            mask_list.insert(0, True)
            # convert binary bits into 4 int32_t
            mask_str_list = [str(int(val)) for val in mask_list]
            mask_str_list.reverse()

            for mask_idx in range(num_packed_masks):
                if mask_idx * 32 >= len(mask_str_list):
                    break
                mask_32bits_str = ''.join(mask_str_list[-(mask_idx + 1) * 32:
                                                        (-mask_idx * 32 - 1)] +
                                          [mask_str_list[(-mask_idx * 32 - 1)]])
                valid_num_bits = len(mask_32bits_str)
                first_bit1 = mask_32bits_str[0] == '1'
                mask_31bits_str = mask_32bits_str[1:]
                mask_31bits = int(mask_31bits_str, 2)
                if valid_num_bits == 32:
                    mask_32bits = mask_31bits - first_bit1 * (2**(
                        valid_num_bits - 1))
                else:
                    mask_32bits = mask_31bits + first_bit1 * (2**(
                        valid_num_bits - 1))
                medusa_packed_mask[token_idx, mask_idx] = mask_32bits
    return medusa_packed_mask


def choices_2_paths(num_medusa_heads, choices):
    paths = {}
    all_paths = {}
    level_counts = [0] * num_medusa_heads
    choices.sort(key=len, reverse=True)
    for c in choices:
        k = ":".join([str(ci) for ci in c])
        if k not in all_paths:
            paths[k] = c
        for i in range(len(c)):
            k = ":".join([str(ci) for ci in c[:i + 1]])
            if k not in all_paths:
                all_paths[k] = c[:i + 1]
                level_counts[i] += 1
    return list(paths.values()), level_counts, paths, all_paths


def get_medusa_topks(num_medusa_heads, paths):
    medusa_topks = [0] * num_medusa_heads
    for p in paths:
        for i, k in enumerate(p):
            medusa_topks[i] = max(medusa_topks[i], k + 1)
    return medusa_topks


def get_medusa_tree(num_medusa_heads, medusa_topks, level_counts, paths):
    cum_topks = np.cumsum([0] + medusa_topks)
    cum_level_counts = np.cumsum([0] + level_counts)
    tree_paths = copy.deepcopy(paths)
    medusa_tree_ids = list(np.arange(medusa_topks[0]))
    medusa_position_offsets = [0] * medusa_topks[0]
    for i in range(1, num_medusa_heads):
        last_prefix = "-1"
        last = -1
        c = -1
        for pi, p in enumerate(paths):
            if i < len(p):
                prefix_str = ":".join([str(k) for k in p[:i]])
                if last_prefix != prefix_str or last != p[i]:
                    # new path
                    medusa_position_offsets.append(i)
                    medusa_tree_ids.append(p[i] + cum_topks[i])
                    last_prefix = prefix_str
                    last = p[i]
                    c += 1
                tree_paths[pi][i] = cum_level_counts[i] + c
    return medusa_tree_ids, medusa_position_offsets, tree_paths


def get_medusa_mask(medusa_tree_ids, medusa_paths):
    medusa_mask = torch.zeros((len(medusa_tree_ids), len(medusa_tree_ids)))
    medusa_mask[:, 0] = 1
    for p in medusa_paths:
        for i, idx in enumerate(p):
            if idx < 0:
                continue
            for j in range(i + 1):
                medusa_mask[idx, p[j]] = 1
    return medusa_mask


def _eagle_setup(choices_or_paths, num_medusa_heads=None):
    choices = copy.deepcopy(choices_or_paths)
    sorted_choices = sorted(choices, key=path_sorting_key)
    if num_medusa_heads is None:
        num_medusa_heads = max([len(c) for c in sorted_choices])
    paths, level_counts, _, _ = choices_2_paths(num_medusa_heads,
                                                sorted_choices)
    paths = sorted(paths, key=path_sorting_key)
    medusa_topks = get_medusa_topks(num_medusa_heads, paths)
    medusa_tree_ids, medusa_position_offsets, tree_paths = get_medusa_tree(
        num_medusa_heads, medusa_topks, level_counts, paths)

    num_medusa_tokens = len(medusa_tree_ids)
    # now do the padding before converting to torch.Tensor
    medusa_paths = []
    for p in tree_paths:
        medusa_paths.append(
            torch.tensor([-1] + p + ([-2] * (num_medusa_heads - len(p)))))
    medusa_topks = torch.tensor(medusa_topks)
    medusa_paths = torch.stack(medusa_paths) + 1
    medusa_tree_ids = torch.tensor([-1] + medusa_tree_ids) + 1
    medusa_position_offsets = torch.tensor([-1] + medusa_position_offsets) + 1
    medusa_mask = get_medusa_mask(medusa_tree_ids, medusa_paths)
    medusa_packed_mask = get_packed_mask(num_medusa_tokens, medusa_mask[1:, 1:])

    return Namespace(
        eagle_mask=medusa_mask.cuda(),
        eagle_packed_mask=medusa_packed_mask.cuda(),
        eagle_topks=medusa_topks.cuda(),
        eagle_paths=medusa_paths.cuda(),
        eagle_tree_ids=medusa_tree_ids.cuda(),
        eagle_position_offsets=medusa_position_offsets.cuda(),
        num_eagle_heads=num_medusa_heads
    )
