from argparse import Namespace
import copy
import torch

TOPK = 10
def pad_path(path, length, pad_value=-2):
    return path + [pad_value] * (length - len(path))

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

def _eagle_setup(choices_or_paths):

    tree_choices = copy.deepcopy(choices_or_paths)
    sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
    tree_len = len(sorted_tree_choices) + 1

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
                    iblist.append(torch.tensor(bt))
                else:
                    iblist.append(b)
        b_indices_new.append(iblist)
    
    initial_buffer = generate_tree_buffers_for_eagle(choices_or_paths)
    tree_buffers = {
        "eagle_mask": tree_attn_mask.cuda(),
        "eagle_tree_ids": tree_indices.cuda(),
        "eagle_position_offsets": tree_position_ids.cuda(),
        "eagle_paths": retrieve_indices.cuda(),
        "eagle_topks": TOPK,
        "initial_buffer": initial_buffer

    }

    return Namespace(**tree_buffers)