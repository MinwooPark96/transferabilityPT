import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel
from typing import Optional
from tqdm import tqdm
from eval import Evaluator
from glue_dataset import GlueDataset

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
        
        self.source_soft_prompt = source_soft_prompt.to(self.device)
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
    
    def train(self):
        optimizer = optim.Adam([self.target_soft_prompt], lr=0.01)
        for _ in range(1):
            pbar = tqdm(self.train_dataloader)    
            for step, batch in enumerate(pbar):
                # print(batch)
                print(self.source_tokenizer)
                print(self.target_tokenizer)
                print(self.source_tokenizer.decode(batch['source_input_ids'][0]))
                print(self.target_tokenizer.decode(batch['target_input_ids'][0]))
                exit()
                
                
                
                # loss = outputs.loss
                # loss.backward()
                # optimizer.step()
                # optimizer.zero_grad()
                
    
    
if __name__ == '__main__':
    
    dynamic = DynamicRel2abs(
        source_soft_prompt=torch.randn(10, 768),
        source_model_name='bert-base-cased',
        target_model_name='roberta-large',
        dataset_name='sst2',
        overwrite_cache=False
    )
    
    dynamic.train()
