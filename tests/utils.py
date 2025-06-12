from collections import OrderedDict
from itertools import combinations
import torch

def rename_state_dict_keys(state_dict, device='cuda:0'):
    """Change name of model parameter  tensors, e.g., `model.fc1.bias` --> 
    `fc1.bias` to use with pure pytorch instead of lightning."""
    new_dict = OrderedDict()
    for k in state_dict:
        new_key = '.'.join(k.split('.')[1:])
        new_dict.update({new_key: state_dict[k]})
    # if device == 'cpu':
    if True:
        newer_dict = OrderedDict()
        for k in new_dict:
            if 'lin_dst' in k:
                pass
            elif 'lin_src' in k:
                newer_dict.update({k.replace('lin_src', 'lin'): new_dict[k]})
            else:
                newer_dict.update({k: new_dict[k]})
        new_dict = newer_dict

    return new_dict

def create_complete_graph_edges(num_nodes: int) -> torch.tensor:
    # num_of_edges = n(n-1)/2 ; *2 for pyg
    edge_index = [edge for edge_to in list(combinations(range(num_nodes), 2)) for edge in (edge_to, (edge_to[1], edge_to[0]))]
    edge_index = torch.tensor(edge_index).t().contiguous()
    return edge_index