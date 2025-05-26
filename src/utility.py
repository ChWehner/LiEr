# supposed to be messy

import torch
import numpy as np
from prettytable import PrettyTable
import gc
from typing import List

def pretty_print_relation_types(relation_types): 
    num_relation_types = len(relation_types)
    pretty_print = "" 
    for count, relation_type in enumerate(relation_types):
        if num_relation_types==1:
            pretty_print = f'{relation_type.split("__")[-1]}'
        elif count == num_relation_types-1:
            pretty_print+= f', and {relation_type.split("__")[-1]}'   
        else:
            pretty_print += f'{relation_type.split("__")[-1]}, '
    return pretty_print   

# from https://discuss.pytorch.org/t/how-to-sort-tensor-by-given-order/61625/2
def smart_sort(x, permutation):                     
    d1, d2 = x.size()
    ret = x[
        torch.arange(d1).unsqueeze(1).repeat((1, d2)).flatten(),
        permutation.flatten()
    ].view(d1, d2)
    return ret

def pretty_print_believes(self, environment, ranking, logits, mrr, num_prints=1):
    """ TODO: _summary_
    Args:
        ranking (_type_): _description_
        logits (_type_): _description_
        mrr (_type_): _description_
        num_prints (int, optional): _description_. Defaults to 3.

     Returns:
        _type_: _description_
    """
    # pretty print top believes 
    # bring previous_action, current_state, missing_relation, and start_state in rollout view
    # create paths views [batch_dim, rollout_dim, step_dim, 4=(previous_action, current_state, missing_relation, start_state)] (and logits?)
    states =environment.states[1:]                  # shape [steps, batch_size x rollouts, numbers(=3)]
    actions = environment.actions[1:]                   # shape [steps, batch_size x rollouts, numbers(=2)]
    #missing_relations = torch.repeat_interleave(self.trainer.model.env.missing_relations.unspeeze(dim=0), repeats=actions.size(), axis=0)  # shape [steps, batch_size x rollouts, number(=1)]
    #starts = np.repeat(np.expand_dims(self.trainer.model.env.states[0].view(), axis=0), repeats=actions.shape[0], axis=0)  # shape [steps, batch_size x rollouts, number(=1)]
     
    #paths = np.concatenate((actions, states, missing_relations, starts), axis=2)
    paths = torch.concatenate((actions, states), dim = 2)
    # bring paths in rollout view 
    batch_size = int(actions.size(1)/self.rollouts_test)
    paths = torch.reshape(paths, (actions.size(0), batch_size, self.rollouts_test , 5))  # shape [steps, batch_size, rollouts, numbers(=5)]
    # and transpose to make batch dim to the first dim
    paths = torch.permute(paths, (1,2,0,3)) # shape [batch_size, rollouts, step_dim, number(=1)]
 
    batch_idx_memory = np.array([])
    print_paths =  np.array([])
    print_logits = np.array([])
    for i in range(num_prints):
        if i < batch_size: 
            batch_idx = np.random.randint(low=0, high=batch_size, size=1)
            while batch_idx in batch_idx_memory : 
                batch_idx = np.random.randint(low=0, high=batch_size, size=1)
            top_rollout_idx = ranking[batch_idx, -1]
            path = paths[batch_idx, top_rollout_idx].detach().cpu().numpy()
            logit = np.exp(logits[batch_idx, top_rollout_idx].detach().cpu().numpy())
            if len(print_paths) == 0:
                print_paths = path
                print_logits = logit
            else:
                print_paths = np.concatenate((print_paths,path), axis=0)
                print_logits = np.concatenate((print_logits,logit), axis=0)
    # get the all node uris form the environment
    #all_nodes = np.unique(np.concatenate((print_paths[:, :, 1], print_paths[:, :, 3]), axis = 0)).astype(str).tolist()
    node_uri_dict = environment.id_to_entity
    # get the all relations types form the environment
    #all_relations = np.unique(np.concatenate((print_paths[:, :, 0], print_paths[:, :, 2]), axis = 0)).astype(str).tolist()
    relation_type_dict = environment.id_to_relation

    def pretty_string_node(node):     # TODO: put this into utility script
        return node.split("#")[-1]
        
    def pretty_string_relation(relation):     # TODO: put this into utility script
        return relation.split("__")[-1]

    def pretty_string_query(path, node_uri_dict, relation_type_dict):            # TODO: put this into utility script     
        head_entity = node_uri_dict[path[0, 3]]
        query_symbol = relation_type_dict[path[0,4]]

        head_entity = pretty_string_node(head_entity)
        query_symbol = pretty_string_relation(query_symbol)
        return f'({head_entity} {query_symbol} ???)'
        
    def pretty_string_reasoning(path, node_uri_dict, relation_type_dict):
        # add start
        reasoning = pretty_string_node(node_uri_dict[path[0, 3]])
        for step in path:
            # try/expect "filters" the self loop relation
            try:
                relation = pretty_string_relation(relation_type_dict[step[0]])
                current_node = pretty_string_node(node_uri_dict[step[2]])                                 
                reasoning += f'--{relation}-->{current_node}'  
            except KeyError:
                continue
        return reasoning
 
    print("---------------------------------------------------------------------------------") 
    print("Here are a few examples of what I currently believe:")
    tab = PrettyTable()
    tab.add_column("", ["Query", "Answer", "Reasoning", "Probability"], align='l')
    for idx, path in enumerate(print_paths):
        path = path.squeeze()
        # construct querys
        query = pretty_string_query(path, node_uri_dict, relation_type_dict)
        # construct answers
        answer = pretty_string_node(node_uri_dict[path[-1, 2]])
        # construct reasonings
        reasoning = pretty_string_reasoning(path, node_uri_dict, relation_type_dict)
        print(print_logits.shape)
        print(idx)
        probability = "{:.2f} %".format((print_logits[idx]*100).flatten()[0])

        # pretty print
        tab.add_column(f"{idx+1}", [query, answer, reasoning, probability], )
    print(tab)
    print(f"Overall, I have an MRR of {mrr} on the test knowledge graph.")
    print("Let me learn a bit more.")
                

    return

def unique(x, dim=0):
    """from : https://github.com/pytorch/pytorch/issues/36748
    """
    unique, inverse, counts = torch.unique(x, dim=dim, 
        sorted=True, return_inverse=True, return_counts=True)
    inv_sorted = inverse.argsort(stable=True)
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]
    return unique, inverse, counts, index

def make_path_dict(path:list)-> dict:
    """ creates paths dict for mongodb. 
    Args:
        path (list): shape = [step_dim, 4=(previous_action, current_state, start_state, missing_relation)]
    """
    walk = [{"head": path[idx][2] if idx==0 else path[idx-1][1], "relation":path[idx][0], "tail": path[idx][1]} for idx in range(len(path))]
    query = {"head": path[0][2], "missing_relation":path[0][3]}
    path = {"query": query, "answer": path[-1][1] ,"walk": walk}
    return path

def edge_index_select(t: torch.sparse_coo, query_row: torch.Tensor, query_col: torch.Tensor) -> torch.Tensor:
    """from https://github.com/rusty1s/pytorch_sparse/issues/214
    """
    row, col, val = t.coo()
    row_mask = row == query_row.view(-1, 1)
    col_mask = col == query_col.view(-1, 1)
    mask = torch.max(torch.logical_and(row_mask, col_mask), dim=0).values
    return val[mask]

def discount_cumsum(vector: torch.tensor, discount:float, device:str='cpu'): 
    wts = discount**torch.arange(vector.size(1), dtype = torch.float64, device=device) # pre-compute all discounts
    x = wts*vector # weighted vector
    cum_sum = torch.cumsum(x, dim=1) #forward cumsum
    re_cum_sum = x - cum_sum + cum_sum[:, -1:] #reversed cumsum
    return re_cum_sum/wts

class MappingError(Exception):
    "Raised when the environment misses a mapping dictionary."
    def __init__(self, msg="The environment is missing an id_to_* mapping."):
        super().__init__(msg)


def process_file(input_file, output_file):
    """
    Utility function to process existing localization files to adequate json format for further processing.
    """
    with open(input_file, 'r') as f:
        data = f.read()

    formatted_paths = data.split("{")[1:]

    result = {}
    for i, block in enumerate(formatted_paths):
        try:
            block = "{" + block.split("}", 1)[0] + "}"
            parsed_block = json.loads(block)
            result[str(i)] = parsed_block["formattedPath"]
        except Exception as e:
            print(f"failed {i} in file {input_file}: {e}")

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)


def get_memory_size_of_tensor(tensor: torch.Tensor, return_as: str = 'GiB', get_nnz: bool = False):
    """
    Returns memory occupied by a tensor, including handling sparse tensors.
    Ignores Python object overhead.

    Args:
        tensor (torch.Tensor): The tensor whose memory size is to be calculated.
        return_as (str): The unit for the returned memory size.
                         Accepted values: 'Bit', 'Byte', 'KiB', 'MiB', 'GiB'
        get_nnz (bool): If true, prints memory allocated for non-zero elements of dense tensor.

    Returns:
        float: Memory occupied by the tensor in the specified unit.
    """
    unit_factors = {
        'Bit': 8,
        'Byte': 1,
        'KiB': 1 / 1024,
        'MiB': 1 / (1024**2),
        'GiB': 1 / (1024**3),
    }

    if return_as not in unit_factors:
        raise ValueError(f"Unsupported unit '{return_as}'. Supported units are: {list(unit_factors.keys())}")

    if tensor.is_sparse:
        indices_memory = tensor.indices().numel() * tensor.indices().element_size()
        values_memory = tensor.values().numel() * tensor.values().element_size()
        total_memory = indices_memory + values_memory
    else:
        total_memory = tensor.numel() * tensor.element_size()
        if get_nnz:
            nonzero = torch.nonzero(tensor)
            nonzero_memory = nonzero.numel() * nonzero.element_size()
            fraction_nonzero = nonzero.size(0) / tensor.numel()
            print(f'Memory allocated for non-zero elements: {nonzero_memory}. '
                  f'Fraction of non-zero elements: {fraction_nonzero}')


    return total_memory * unit_factors[return_as]

def delete_tensors(tensors: list[torch.tensor]):
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            del tensor
    gc.collect()
    torch.cuda.empty_cache()


def compare_paths(ground_truth: List[str], candidate: List[str]):
    """
    Compares the candidate path with the ground_truth path step by step.

    Args:
        ground_truth (list): The reference path.
        candidate (list): The path to compare.

    Returns:
        tuple: (count, valid) where count is the number of consecutive matching
               elements from the beginning, and valid is True if the entire candidate
               matches the corresponding portion of ground_truth.
    """
    count = 0
    for i, gt in enumerate(ground_truth):
        if i < len(candidate) and candidate[i] == gt:
            count += 1
        else:
            break
    valid = count == len(candidate)
    return count, valid
