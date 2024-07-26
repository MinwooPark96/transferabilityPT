import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os

from transformers import AutoModel
from typing import Optional
from tqdm import tqdm
from eval import Evaluator
from utils import load_soft_prompt_from
from glue_dataset import GlueDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

class DynamicRel2abs:
    
    def __init__(
        self,
        source_soft_prompt: torch.Tensor,
        source_model_name: str,
        target_model_name: str,
        dataset_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        pad_to_max_length: bool = True,
        do_train: bool = True,
        max_train_samples: Optional[int] = None,
        per_device_train_batch_size: int = 16,
        do_eval: bool = True,
        max_seq_length: int = 128,
        max_eval_samples: Optional[int] = None,
        per_device_eval_batch_size: int = 16,
        cache_dir: Optional[str] = None,
        overwrite_cache: bool = False,
        ):

        self.device = device
        self.evaluator = Evaluator(
            model_name = target_model_name,
            dataset_name = dataset_name,
            device = self.device)
        
        self.source_model_name = source_model_name
        self.target_model_name = target_model_name
        
        self.source_model = AutoModel.from_pretrained(source_model_name)
        self.target_model = AutoModel.from_pretrained(target_model_name)
        
        self.source_embedding_layer = self.source_model.get_input_embeddings().eval().to(self.device)
        self.target_embedding_layer = self.target_model.get_input_embeddings().eval().to(self.device)

        self.mean = torch.mean(self.target_embedding_layer.weight.data.reshape(-1)) 
        self.std = torch.std(self.target_embedding_layer.weight.data.reshape(-1))
        
        self.source_soft_prompt = source_soft_prompt.to(self.device)
        self.pre_seq_len = self.source_soft_prompt.shape[0]
        self.target_soft_prompt = torch.empty((self.source_soft_prompt.shape[0], self.target_model.config.hidden_size)).to(self.device) 
        self.target_soft_prompt.requires_grad = True
        
        torch.nn.init.xavier_normal_(self.target_soft_prompt)
        
        self.dataset = GlueDataset(
            source_model_name = source_model_name,
            target_model_name = target_model_name,
            dataset_name = dataset_name,
            pad_to_max_length = pad_to_max_length,
            max_seq_length = max_seq_length,
            do_train = do_train,
            max_train_samples = max_train_samples,
            do_eval = do_eval,
            max_eval_samples = max_eval_samples,
            cache_dir = cache_dir,
            overwrite_cache = overwrite_cache
            )
        
        self.max_seq_length = max_seq_length
        
        self.source_tokenizer = self.dataset.source_tokenizer
        self.target_tokenizer = self.dataset.target_tokenizer
        
        self.train_dataset, self.eval_dataset = self.dataset.train_dataset, self.dataset.eval_dataset
        self.per_device_train_batch_size, self.per_device_eval_batch_size = per_device_train_batch_size, per_device_eval_batch_size
        
        self.train_dataloader = DataLoader(self.train_dataset, 
                                           batch_size = self.per_device_train_batch_size, 
                                           shuffle = True,
                                           collate_fn=self.dataset.data_collator)
        
        self.train_dataloader = DataLoader(self.eval_dataset, 
                                           batch_size = self.per_device_eval_batch_size, 
                                           shuffle = True,
                                           collate_fn=self.dataset.data_collator)
        
        self.cos_loss = nn.CosineEmbeddingLoss()
        
        print('DynamicRel2abs initialized!')
    
    def regularize_tensor(self, tensor: torch.Tensor):
        
        current_mean = torch.mean(tensor).to(self.device)
        current_std = torch.std(tensor).to(self.device)
        
        normalized_tensor = (tensor - current_mean) / current_std
        regularized_tensor = normalized_tensor * self.std + self.mean

        return regularized_tensor
    
    def proj2rel(self, 
                tensor_absolute: torch.Tensor, 
                anchor_embedding: torch.Tensor) -> torch.Tensor:
    
        A = F.normalize(tensor_absolute, dim=-1) # (pre_seq_len, hidden_size)
        B = F.normalize(anchor_embedding, dim=-1) # (batch, max_seq_len, hidden_size)
        
        return torch.matmul(A, B.transpose(-1,-2)) # (batch, pre_seq_len, max_seq_len)
    
    def train(self):
        optimizer = optim.Adam([self.target_soft_prompt], lr=0.001)
        # y = torch.ones(self.per_device_train_batch_size).to(self.device)
        
        pbar = tqdm(range(1000))
        for i in pbar:
            for _, batch in enumerate(self.train_dataloader):
                
                with torch.no_grad():
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                batch_size = batch['source_input_ids'].shape[0]
                y = torch.ones(batch_size * self.pre_seq_len).to(self.device)
        
                source_tokens = batch['source_input_ids'] # (batch, max_seq_len)
                target_tokens = batch['target_input_ids'] # (batch, max_seq_len)
                
                source_anchor = self.source_embedding_layer(source_tokens) # (batch, max_seq_len, source_hidden_size)
                target_anchor = self.target_embedding_layer(target_tokens) # (batch, max_seq_len, target_hidden_size)
                
                source_relative = self.proj2rel(self.source_soft_prompt, source_anchor) # (batch, pre_seq_len, max_seq_len)
                target_relative = self.proj2rel(self.target_soft_prompt, target_anchor) # (batch, pre_seq_len, max_seq_len)
                
                
                loss = self.cos_loss(\
                    target_relative.view(-1,self.max_seq_length),\
                    source_relative.view(-1,self.max_seq_length),\
                        y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                pbar.set_description(f'loss: {loss : 0.4f}')
                
            if (i+1) % 100 == 0 :
                with torch.no_grad():
                    # regularized_target_soft_prompt = self.regularize_tensor(self.target_soft_prompt) # 민우
                    
                    # current_target_soft_prompt = regularized_target_soft_prompt.detach()#논문
                    
                    # precision = self.evaluator.eval(soft_prompt = regularized_target_soft_prompt)['accuracy']
                    
                    precision = self.evaluator.eval(soft_prompt = self.target_soft_prompt.detach())['accuracy']
                
                    print('Get best precision: %.4f at step %d! loss: %.8f' % (precision, i+1, loss.item()))
            
    
if __name__ == '__main__':
    soft_prompt = load_soft_prompt_from("../prompts/100_bert-base-cased_sst2_5.bin")
    dynamic = DynamicRel2abs(
        source_soft_prompt=soft_prompt,
        source_model_name='bert-base-cased',
        target_model_name='roberta-large',
        dataset_name='sst2',
    )
    
    dynamic.train()
