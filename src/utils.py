import os
import regex as re
import logging
import torch.nn as nn
import json
import torch 
import random
import numpy as np
import inspect
from typing import Union, Iterable, Optional
import sys
sys.path.append('..')

logger = logging.getLogger(__name__)

def count_trainable_parameters(model):
    """[minwoo] source from : https://github.com/ZhengxiangShi/PowerfulPromptFT/blob/main/src/model.py#L243"""
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_trainable_params}/{num_params}")
    return num_trainable_params

def random_mask_input_ids(input_ids, mask_token_id, exceptions, prob=0.15):
    # generate randomly masked input_ids for MLM task
    # 현재 task 에선 사용할 필요 없을 듯.
    # modified from https://towardsdatascience.com/masked-language-modelling-with-bert-7d49793e5d2c
    """
    [minwoo] source from : https://github.com/salesforce/Overture/blob/main/utils.py
    exceptions: list, token ids that should not be masked
    """
    probs = torch.rand(input_ids.shape)
    mask = probs < prob
    for ex_id in exceptions:
        mask = mask * (input_ids != ex_id)
    selection = []
    for i in range(input_ids.shape[0]):
        selection.append(torch.flatten(mask[i].nonzero()).tolist())
    for i in range(input_ids.shape[0]):
        input_ids[i, selection[i]] = mask_token_id
    return input_ids

def fix_seed(seed):
    """[minwoo] source from https://github.com/MANGA-UOFA/PTfer/blob/main/get_emb.py"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    
def wrap(v_or_vs: Union[str, Iterable[str]]) -> Optional[frozenset[str]]:
    """[minwoo] wrap the input into frozenset if it is not None."""
    if v_or_vs is None:
        return None
    if isinstance(v_or_vs, str):
        return frozenset({v_or_vs})
    else:
        return frozenset(v_or_vs)

def print_params_only_requires_grad_true(model: torch.nn.Module):
    """[minwoo] print all params with requires_grad=True"""
    for n,p in model.named_parameters():
        if p.requires_grad:
            print(n, p.requires_grad)

def print_params_with_requires_grad(model: torch.nn.Module):
    """[minwoo] print all params"""
    for n,p in model.named_parameters():
        print(n, p.requires_grad)

def get_params_dict(model: torch.nn.Module):
    """[minwoo] return params from torch model"""
    return {n:p for n,p in model.named_parameters()}

def print_object_methods(obj):
    """[minwoo] print all methods of the object"""
    methods = inspect.getmembers(obj, predicate=inspect.ismethod)
    for name, method in methods:
        print(name)

def print_object_fields(obj):
    """[minwoo] print all fields of the object"""
    fields = vars(obj)
    for field_name, field_value in fields.items():
        print(f"{field_name}: {field_value}")

def create_dir(output_dir):
    """
    Checks whether to the output_dir already exists and creates it if not.
    Args:
      output_dir: path to the output_dir
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for p in model.parameters():
        p.requires_grad = False

def load_soft_prompt_from(filepath: str):
    """[minwoo] return pretrained soft prompt"""
    if filepath.endswith('.bin'):
        weight_dict = torch.load(filepath)
        return weight_dict['soft_prompt']     
    
    elif filepath.endswith('.npy'):
        return np.load(filepath)
    
    elif filepath.endswith('.pt'):
        return torch.load(filepath)

def tensor_to_array(tensor: torch.Tensor):
    """[minwoo] convert tensor to numpy"""
    return tensor.detach().cpu().numpy()
        
def array_to_tensor(array: np.ndarray):
    """[minwoo] convert numpy to tensor"""
    return torch.from_numpy(array)

if __name__ == '__main__':
    
    pass