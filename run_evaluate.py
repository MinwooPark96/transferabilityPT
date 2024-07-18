import os
import torch
import argparse
from tqdm import tqdm
from typing import Optional

from src.utils import load_soft_prompt_from, fix_seed
from src.glue_dataset import GlueDatasetForEval
from src.model import MLMPromptForEval
from src.eval import Evaluator

# [minwoo] source from https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue_no_trainer.py#L507C4-L507C5

def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--prompt_filename', type=str, default='./output/bert-base-uncased/sst2/epoch_100_prompt.bin')
    parser.add_argument('--dataset_name', type=str, default='sst2')
    parser.add_argument('--pad_to_max_length', type=bool, default=True)
    parser.add_argument('--do_eval', type=bool, default=True)
    parser.add_argument('--do_predict', type=bool, default=False)
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--max_eval_samples', type=int, default=None)
    parser.add_argument('--max_predict_samples', type=int, default=None)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=16)
    parser.add_argument('--per_device_predict_batch_size', type=int, default=16)
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--overwrite_cache', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    return args
    
if __name__ == '__main__':
    args = parse_args()
    evaluator = Evaluator(**vars(args),device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(evaluator.eval())
    