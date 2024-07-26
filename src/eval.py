import os
import torch
import argparse
from tqdm import tqdm
from typing import Optional
import numpy as np
import logging
import evaluate

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
)
from datasets.load import load_dataset
from torch.utils.data import DataLoader

from utils import load_soft_prompt_from, fix_seed
from model import MLMPromptForEval

logger = logging.getLogger(__name__)

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

class GlueDatasetForEval():
    def __init__(self,
                    model_name: str,
                    dataset_name: Optional[str] = None,
                    pad_to_max_length: bool = True,
                    max_seq_length: int = 128,
                    do_eval: bool = True,
                    max_eval_samples: Optional[int] = None,
                    do_predict: bool = False,
                    max_predict_samples: Optional[int] = None,
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
            logger.warning(
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

        if do_eval:
            self.eval_dataset = raw_datasets["validation_matched" if dataset_name == "mnli" else "validation"]
            if max_eval_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(max_eval_samples))

        if do_predict:
            
            self.predict_dataset = raw_datasets["test_matched" if dataset_name == "mnli" else "test"]
            
            if max_predict_samples is not None:
                self.predict_dataset = self.predict_dataset.select(range(max_predict_samples))

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
    
    """
    [minwoo] 
    의 Line 510 과 거의 유사 
    """
    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if self.is_regression else np.argmax(preds, axis=1)
        
        if self.args.dataset_name is not None:
            result = self.metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        # [minwoo] glue task 에서 이 아래로 내려올 일 없을 듯?
        elif self.is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


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
            print('Set soft prompt from input!')
            
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
            # print(predictions)
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()

        return eval_metric