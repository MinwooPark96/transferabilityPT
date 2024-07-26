import numpy as np
import torch
import random
import argparse
import os
import logging

from src.rel2abs_utils import get_absolute_anchors
from src.utils import fix_seed, load_soft_prompt_from, tensor_to_array
from src.rel2abs import Rel2abs_Decoder
    
def save_soft_prompt(args, soft_prompt):
    save_path = os.path.join('./output',args.transfered_prompt_filename)
    torch.save(soft_prompt, save_path)
        
def transfer(args):
    source_soft_prompt = load_soft_prompt_from(args.prompt_filename)
    
    source_anchor_embedding, target_anchor_embedding, target_statistic = get_absolute_anchors(
        source_model_name = args.source_model_name, 
        target_model_name = args.target_model_name, 
        num_anchor = args.num_anchor, 
        all_anchors = args.all_anchors,
        common_vocab = args.common_vocab
        )

    # [minwoo] decoder inititalization
    decoder = Rel2abs_Decoder(args, 
                              source_soft_prompt, 
                              source_anchor_embedding, 
                              target_anchor_embedding, 
                              target_statistic)
    
    x = decoder.search()

    # save_soft_prompt(args, x)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_model_name', type=str, default='bert-base-cased')
    parser.add_argument('--target_model_name', type=str, default='roberta-large')
    
    # parser.add_argument('--prompt_filename', type=str, default='./prompts/100_bert-base-uncased_sst2_5.bin')
    parser.add_argument('--prompt_filename', type=str, default='./prompts/100_bert-base-cased_sst2_5.bin')
    parser.add_argument('--transfered_prompt_filename', type=str, default='transfered_prompt.pt')
    
    parser.add_argument('--num_anchor', type=int, default=8192)
    parser.add_argument('--all_anchors', action='store_true')
    parser.add_argument('--common_vocab', type=str, default=None)
    
    parser.add_argument('--seed', type=int, default=42)
    
    # args for validation
    parser.add_argument('--dataset_name', type=str, default='sst2')
    parser.add_argument('--budget', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-2)
    
    args = parser.parse_args()

    fix_seed(args.seed)
    
    transfer(args)