import os
import torch
import argparse
from tqdm import tqdm
from typing import Optional, Union, Tuple
import random
import numpy as np
import logging

from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig
from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.models.roberta.modeling_roberta import RobertaForMaskedLM
from transformers import default_data_collator, DataCollatorWithPadding

from torch.utils.data import DataLoader
import torch.nn.functional as F

from datasets.load import load_dataset
import evaluate
from utils import fix_seed, get_params_dict, print_params_with_requires_grad

"""
[minwoo] source from AutoPrompt paper.
"""

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
class ModelForLabelSearch(torch.nn.Module):
    def __init__(self,
                 model,
                 config,
                 tokenizer,
                 pre_seq_len,
                 *args,
                **kwargs
                ):
    
        super(ModelForLabelSearch,self).__init__()
        self.model, self.config, self.tokenizer = model, config, tokenizer
        self.model.eval()
        self.pre_seq_len = pre_seq_len
        self.mask_ids = self.tokenizer.mask_token_id
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs): 
        
        batch_size = input_ids.shape[0]         
        mask_input_ids = torch.full((batch_size, 1 + self.pre_seq_len), self.mask_ids).to(input_ids.device)
        input_ids = torch.cat([mask_input_ids, input_ids], dim=1)
        
        mask_attention_mask = torch.ones(batch_size, 1 + self.pre_seq_len).to(input_ids.device)
        attention_mask = torch.cat([mask_attention_mask, attention_mask],dim=1)
        
        self.model(
            input_ids = input_ids,
            attention_mask = attention_mask            
        )
        return None
        
class GlueDataset():
    def __init__(self,
                    model_name: str,
                    dataset_name: Optional[str] = None,
                    pad_to_max_length: bool = True,
                    max_seq_length: int = 128,
                    max_train_samples: Optional[int] = None,
                    cache_dir: Optional[str] = None,
                    overwrite_cache: bool = False,
                    *args,
                    **kwargs
                 ) -> None:
        
        self.model_name = model_name
        
        raw_datasets = load_dataset("glue", dataset_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.is_regression = dataset_name == "stsb" # [minwoo] stsb 는 regression task
        
        if not self.is_regression:
            self.label_list = raw_datasets["train"].features["label"].names
            self.num_labels = len(self.label_list)
        
        else:
            self.num_labels = 1

        self.sentence1_key, self.sentence2_key = task_to_keys[dataset_name]

        if pad_to_max_length: #[minwoo] 이쪽을 사용하는 듯?
            self.padding = "max_length"
        else:
            self.padding = False

        if not self.is_regression:
            self.label2id = {l: i for i, l in enumerate(self.label_list)}
            self.id2label = {id: label for label, id in self.label2id.items()}

        if max_seq_length > self.tokenizer.model_max_length:
            print(
                f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the"
                f"model ({self.tokenizer.model_max_length}). Using max_seq_length={self.tokenizer.model_max_length}."
            )
        self.max_seq_length = min(max_seq_length, self.tokenizer.model_max_length)

        raw_datasets = raw_datasets.map(
            self.preprocess_function,
            batched=True,
            load_from_cache_file = not overwrite_cache,
            desc="Running tokenizer on dataset",
        )

        self.train_dataset = raw_datasets["train"]
        if max_train_samples is not None:
            self.train_dataset = self.train_dataset.select(range(args.max_train_samples))
            self.train_dataset = self.merge_source_target_datasets(self.source_train_dataset, self.target_train_dataset)
        
        # [minwoo] self.metric = load_metric("glue", args.dataset_name) -> warning 수정
        if dataset_name is not None:
            self.metric = evaluate.load("glue", 
                                        dataset_name, 
                                        cache_dir=cache_dir)
        elif self.is_regression:
            self.metric = evaluate.load("mse", 
                                        cache_dir=cache_dir)
        else:
            self.metric = evaluate.load("accuracy", 
                                        cache_dir = cache_dir)
        

        if pad_to_max_length:
            self.data_collator = default_data_collator
        
        else:
            self.data_collator = DataCollatorWithPadding(self.tokenizer)
            
    # [minwoo] init에서 자동으로 호출됨.
    def preprocess_function(self, examples):
        # Tokenize the texts
        args = (
            (examples[self.sentence1_key],) if self.sentence2_key is None else (examples[self.sentence1_key], examples[self.sentence2_key])
        )
        
        result = self.tokenizer(*args, padding=self.padding, max_length=self.max_seq_length, truncation=True)

        return result
    
    def compute_metrics(self, p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if self.is_regression else np.argmax(preds, axis=1)
        
        if self.args.dataset_name is not None:
            result = self.metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result

        elif self.is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

class Hooker:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained. 
    """
    def __init__ (self, module):
        self._stored_output = None
        module.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        self._stored_output = output
    
    def get(self):
        return self._stored_output

def get_final_embeddings(model):
    if isinstance(model, BertForMaskedLM):
        """
        [minwoo]
        BertPredictionHeadTransform(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (transform_act_fn): GELUActivation()
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            )
        """
        return model.cls.predictions.transform
    elif isinstance(model, RobertaForMaskedLM):
        return model.lm_head.layer_norm
    else:
        raise NotImplementedError(f'{model} not currently supported')

def get_word_embeddings(model):
    if isinstance(model, BertForMaskedLM):
        return model.cls.predictions.decoder.weight
    elif isinstance(model, RobertaForMaskedLM):
        return model.lm_head.decoder.weight
    else:
        raise NotImplementedError(f'{model} not currently supported')

def load_pretrained(model_name):
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name, config=config) # BertForMaskedLM
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # utils.add_task_specific_tokens(tokenizer)
    return config, model, tokenizer

def main(args):
    fix_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading model, tokenizer, etc.")
    config, model, tokenizer = load_pretrained(args.model_name)
    
    search_model = ModelForLabelSearch(
        model = model, 
        config = config, 
        tokenizer = tokenizer, 
        pre_seq_len = 5).to(device)
    
    final_embeddings = get_final_embeddings(model)
    embedding_storage = Hooker(final_embeddings) # [minwoo] Hook the final_embeddings
    word_embeddings = get_word_embeddings(model)
    
    dataset = GlueDataset(args.model_name, args.dataset_name)
    """
    [minwoo] 
        It is a linear layer hidden 
        hidden dim -> label dim
    """
    projection = torch.nn.Linear(config.hidden_size, len(dataset.label_list)).to(device)
    
    train_dataset = dataset.train_dataset
    train_dataloader = DataLoader(train_dataset, 
                                  shuffle=True, 
                                  collate_fn=dataset.data_collator, 
                                  batch_size=args.per_device_train_batch_size)

    optimizer = torch.optim.Adam(projection.parameters(), lr=args.lr)

    scores = torch.matmul(projection.weight, word_embeddings.transpose(0, 1)) 
    scores = F.softmax(scores, dim=0)
    
    print('Training')
    for i in range(args.iters):
        pbar = tqdm(train_dataloader)
        for step, batch in enumerate(pbar):
            optimizer.zero_grad()
            model_inputs = {'input_ids' : batch['input_ids'].to(device),
                            'attention_mask' : batch['attention_mask'].to(device),
                            'token_type_ids' : batch.get('token_type_ids').to(device)}
            
            labels = batch['labels'].to(device)
            
            with torch.no_grad():
                search_model(**model_inputs)
            
            # [minwoo] hooking embeddings (b, seq_len, hidden_size)
            embeddings = embedding_storage.get() 
            # 각 sequence 의 prediction mask token 을 hooking 해옴.
            # TODO 생각해보니 현재 dataset 은 masking 전임.. ㅋㅋ
            # TODO 당연하지만, masking token 들 (trigger) 도 attention 함.
             
            predict_embeddings = embeddings[:,0,:] # [minwoo] It's shape is (b, hidden_size)
            logits = projection(predict_embeddings) # logits = (b, label_dim)
            loss = F.cross_entropy(logits, labels.squeeze(-1))
            loss.backward()
            optimizer.step()
            pbar.set_description(f'loss: {loss : 0.4f}')

        scores = torch.matmul(projection.weight, word_embeddings.transpose(0, 1))
        scores = F.softmax(scores, dim=0)
        
        for i, row in enumerate(scores):
            _, top = row.topk(args.k)
            decoded = tokenizer.convert_ids_to_tokens(top)
            print(f"Top k for class {dataset.id2label[i]}: {', '.join(decoded)}")
            
            # [minwoo] Save the words to a txt file
            with open(f'./top_k_words_{dataset.id2label[i]}.txt', 'w') as f:
                f.write('\n'.join(decoded))
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='bert-base-cased')
    parser.add_argument('--dataset_name', type=str, default='sst2')
    parser.add_argument('--per_device_train_batch_size', type=int, default=32)
    parser.add_argument('--pad_to_max_length', type=bool, default=True)
    parser.add_argument('--max_seq_length', type=int, default=30)
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--overwrite_cache', type=bool, default=False)
    
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--iters', type=int, default=100,
                        help='Number of iterations to run trigger search algorithm')
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    main(args)
    