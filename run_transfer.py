# REL=P17
# src_model=bert-base-cased
# tgt_model=bert-large-cased
# src_prompt_path=/OptiPrompt/optiprompt-outputs/${src_model}
# log_path=/output/bert_base2bert_large_log
# mkdir ${log_path}

# python get_emb.py \
#     --log_path ${log_path} \
#     --relation ${REL} \
#     --src_model ${src_model} \
#     --tgt_model ${tgt_model} \
#     --src_prompt_path ${src_prompt_path} \
#     --prompt_filename prompt_vec.npy \
#     --num_vectors 5 \
#     --all_anchors \
#     --topk 8192 

import numpy as np
import torch
import random
import argparse
import os
import logging

from src.transfer_utils import get_absolute_anchors
from src.utils import fix_seed, load_soft_prompt_from, tensor_to_array
from src.rel2abs import Rel2abs_Decoder
    
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                datefmt='%m/%d/%Y %H:%M:%S',
                level=logging.INFO)

logger = logging.getLogger(__name__)


def save_embeddings(args, embeddings):
    save_path = os.path.join(args.tgt_prompt_path, args.relation, args.transferred_prompt_filename)
    with open(save_path, 'wb') as f:
        logger.info('Saving transferred prompt embeddings to %s' % save_path)
        np.save(f, embeddings)
        
def transfer(args):
    source_soft_prompt = load_soft_prompt_from(args.prompt_filename)
    
    source_anchor_embedding, target_anchor_embedding, target_statistic = get_absolute_anchors(
        source_model_name = args.source_model_name, 
        target_model_name = args.target_model_name, 
        num_anchor = args.num_anchor, 
        common_vocab = args.common_vocab, 
        all_anchors = args.all_anchors
        )

    # [minwoo] decoder inititalization
    decoder = Rel2abs_Decoder(args, 
                              logger, 
                              source_soft_prompt, 
                              source_anchor_embedding, 
                              target_anchor_embedding, 
                              target_statistic)
    
    x = decoder.search()

    # save_embeddings(args, x)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--target_model_name', type=str, default='bert-large-uncased')
    
    parser.add_argument('--prompt_filename', type=str, default='./output/bert-base-uncased/sst2/epoch_100_prompt.bin')
    parser.add_argument('--transferred_prompt_filename', type=str, default='transfered_prompt_vecs.npy')
    parser.add_argument('--num_anchor', type=int, default=8192)

    parser.add_argument('--all_anchors', action='store_true')
    parser.add_argument('--common_vocab', type=str, default = None)
    parser.add_argument('--seed', type=int, default=42)
    
    parser.add_argument('--log_path', type=str, default='./logs')
    
    # args for validation
    parser.add_argument('--dataset_name', type=str, default='sst2')
    # parser.add_argument('--relation_profile', type=str, default='OptiPrompt/relation_metainfo/LAMA_relations.jsonl')
    parser.add_argument('--absolute', action='store_true') # [minwoo] topk masking 시에 절대값을 기준으로 masking 할지, 아닐지 여부.
    parser.add_argument('--topk', type=int, default=8192)
    parser.add_argument('--budget', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=5e-3)
    
    args = parser.parse_args()

    logger.addHandler(logging.FileHandler(args.log_path))

    logger.info(args)

    fix_seed(args.seed)
    
    transfer(args)