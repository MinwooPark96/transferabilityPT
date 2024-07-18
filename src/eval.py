import os
import torch
import argparse
from tqdm import tqdm
from typing import Optional

from src.utils import load_soft_prompt_from, fix_seed
from src.glue_dataset import GlueDatasetForEval
from src.model import MLMPromptForEval

from torch.utils.data import DataLoader

class Evaluator:
    def __init__(self, 
                model_name: str,
                dataset_name: str,
                device: str = "cuda" if torch.cuda.is_available() else "cpu",
                prompt_filename: Optional[str] = None,
                pad_to_max_length: bool = True,
                do_eval: bool = True,
                max_seq_length: int = 128,
                max_eval_samples: Optional[int] = None,
                per_device_eval_batch_size: int = 16,
                cache_dir: Optional[str] = None,
                overwrite_cache: bool = False,
                seed: int = 42,
                *args,
                **kwargs
                ):
        
        self.device = device
        self.model = MLMPromptForEval(model_name=model_name,
                                dataset_name=dataset_name).to(self.device)
        
        self.dataset = GlueDatasetForEval(model_name=model_name,
                                    dataset_name=dataset_name,
                                    pad_to_max_length=pad_to_max_length,
                                    do_eval=do_eval,
                                    max_seq_length=max_seq_length,
                                    max_eval_samples=max_eval_samples,
                                    per_device_eval_batch_size=per_device_eval_batch_size,
                                    cache_dir=cache_dir,
                                    overwrite_cache=overwrite_cache)
        
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.seed = seed
        self.prompt_filename = prompt_filename

    def eval(self, 
             soft_prompt: Optional[torch.Tensor] = None):

        fix_seed(self.seed)
        
        if soft_prompt is not None:
            self.model.set_soft_prompt(soft_prompt.to(self.device)) 
            
        elif self.prompt_filename is not None:
            self.model.set_soft_prompt(load_soft_prompt_from(self.prompt_filename).to(self.device))
            print(f'loaded soft prompt from {self.prompt_filename}')
        
        else :
            raise ValueError("soft_prompt or prompt_filename should be given.")
        
        eval_dataset = self.dataset.eval_dataset
        eval_dataloader = DataLoader(eval_dataset, 
                                     collate_fn=self.dataset.data_collator, 
                                     batch_size = self.per_device_eval_batch_size)
        metric = self.dataset.metric

        self.model.eval()
        
        for step, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            with torch.no_grad():
                # 데이터를 GPU로 이동
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
            
            predictions = torch.argmax(outputs, dim=1)
            predictions = predictions
            references = batch["labels"]
            
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()

        return eval_metric