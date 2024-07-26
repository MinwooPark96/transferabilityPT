from datasets.load import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
)
import numpy as np
import logging
import evaluate
from typing import Optional
"""
[minwoo] source from : https://github.com/WHU-ZQH/PANDA/blob/main/p-tuning-v2/tasks/glue/dataset.py
huggigface run_glue.py : https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py#L71
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

logger = logging.getLogger(__name__)

class GlueDataset():
    def __init__(self,
                source_model_name: str,
                target_model_name: str,
                dataset_name: Optional[str] = None,
                pad_to_max_length: bool = True,
                max_seq_length: int = 128,
                do_train: bool = True,
                max_train_samples: Optional[int] = None,
                do_eval: bool = True,
                max_eval_samples: Optional[int] = None,
                do_predict: bool = False,
                max_predict_samples: Optional[int] = None,
                cache_dir: Optional[str] = None,
                overwrite_cache: bool = False,
                 ) -> None:
        """
        [minwoo] This class is used to load glue dataset with dual tokenizer (source & target)
        
        args :
            .source_model_name = 'bert-base-uncased'
            .target_model_name = 'bert-large-uncased'
            .dataset_name = 'sst2'
            .pad_to_max_length = 'max_length'
            .max_seq_length = 128
            .do_train = True
            .max_train_samples = None
            .do_eval = True
            .max_eval_samples = None
            .do_predict = False
            .max_predict_samples = None
            .cache_dir = None
            .overwrite_cache = False
        """

        self.source_model_name = source_model_name
        self.source_tokenizer = AutoTokenizer.from_pretrained(self.source_model_name)
        
        self.target_model_name = target_model_name
        self.target_tokenizer = AutoTokenizer.from_pretrained(self.target_model_name)
        
        raw_datasets = load_dataset("glue", dataset_name, trust_remote_code=True)
            
        #labels
        self.is_regression = dataset_name == "stsb" # [minwoo] stsb 는 regression task
        
        if not self.is_regression:
            self.label_list = raw_datasets["train"].features["label"].names
            self.num_labels = len(self.label_list)
        
        else:
            self.num_labels = 1

        # Preprocessing the raw_datasets
        self.sentence1_key, self.sentence2_key = task_to_keys[dataset_name]

        # Padding strategy
        if pad_to_max_length: #[minwoo] 이쪽을 사용하는 듯?
            self.padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            self.padding = False

        # Some models have set the order of the labels to use, so let's make sure we do use it.
        if not self.is_regression:
            self.label2id = {l: i for i, l in enumerate(self.label_list)}
            self.id2label = {id: label for label, id in self.label2id.items()}
    
        model_max_seq_length = min(self.source_tokenizer.model_max_length, self.target_tokenizer.model_max_length)
            
        if max_seq_length > model_max_seq_length:
            
            logger.warning(
                f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the"
                f"model ({model_max_seq_length}). Using max_seq_length={model_max_seq_length}."
            )
        
        self.max_seq_length = min(max_seq_length, model_max_seq_length)

        source_raw_datasets = raw_datasets.map(
            self.source_preprocess_function,
            batched=True,
            load_from_cache_file = not overwrite_cache,
            desc="Running tokenizer on dataset",
        )

        target_raw_datasets = raw_datasets.map(
            self.target_preprocess_function,
            batched=True,
            load_from_cache_file = not overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        
        if do_train:
            self.source_train_dataset = source_raw_datasets["train"]
            self.target_train_dataset = target_raw_datasets["train"]
            
            if max_train_samples is not None:
                self.source_train_dataset = self.source_train_dataset.select(range(max_train_samples))
                self.target_train_dataset = self.target_train_dataset.select(range(max_train_samples))

            self.train_dataset = self.merge_source_target_datasets(self.source_train_dataset, self.target_train_dataset)
            
        if do_eval:
            self.source_eval_dataset = source_raw_datasets["validation_matched" if dataset_name == "mnli" else "validation"]
            self.target_eval_dataset = target_raw_datasets["validation_matched" if dataset_name == "mnli" else "validation"]
            
            if max_eval_samples is not None:
                self.source_eval_dataset = self.source_eval_dataset.select(range(max_eval_samples))
                self.target_eval_dataset = self.target_eval_dataset.select(range(max_eval_samples))

            self.eval_dataset = self.merge_source_target_datasets(self.source_eval_dataset, self.target_eval_dataset)
            
        if do_predict:
            
            self.source_predict_dataset = source_raw_datasets["test_matched" if dataset_name == "mnli" else "test"]
            self.target_predict_dataset = target_raw_datasets["test_matched" if dataset_name == "mnli" else "test"]
            
            if max_predict_samples is not None:
                self.source_predict_dataset = self.source_predict_dataset.select(range(max_predict_samples))
                self.target_predict_dataset = self.target_predict_dataset.select(range(max_predict_samples))
            
            self.predict_dataset = self.merge_source_target_datasets(self.source_predict_dataset, self.target_predict_dataset)
        
        # [minwoo] self.metric = load_metric("glue", dataset_name) -> warning 수정
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
            """
            data_colloar(
                dataset : Dataset(features : list[list[Any]) 
            ) 
                -> Dict[str,torch.Tensor]
            """
            self.data_collator = default_data_collator
        
        else : 
            NotImplementedError("[minwoo] pad_to_max_length = False is not implemented yet")

    def source_preprocess_function(self, examples):
        args = (
            (examples[self.sentence1_key],) if self.sentence2_key is None else (examples[self.sentence1_key], examples[self.sentence2_key])
        )
        
        result = self.source_tokenizer(*args, padding=self.padding, max_length=self.max_seq_length, truncation=True)

        return result
    
    def target_preprocess_function(self, examples):
        args = (
            (examples[self.sentence1_key],) if self.sentence2_key is None else (examples[self.sentence1_key], examples[self.sentence2_key])
        )
        
        result = self.target_tokenizer(*args, padding=self.padding, max_length=self.max_seq_length, truncation=True)

        return result
    
    def merge_source_target_datasets(self, source_dataset, target_dataset):
        if len(source_dataset) != len(target_dataset):
            raise ValueError("Source and target datasets must have the same length.")
        
        merged_dataset = source_dataset.map(lambda example, idx: {
            'source_input_ids': example['input_ids'],
            'target_input_ids': target_dataset[idx]['input_ids'],
            'source_attention_mask': example['attention_mask'],
            'target_attention_mask': target_dataset[idx]['attention_mask'],
        }, with_indices=True)
        columns_to_remove = ['input_ids', 'attention_mask']
        
        if 'token_type_ids' in source_dataset.column_names:
            columns_to_remove.append('token_type_ids')
        
        merged_dataset = merged_dataset.remove_columns(columns_to_remove)
        
        return merged_dataset

    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if self.is_regression else np.argmax(preds, axis=1)
        
        if self.dataset_name is not None:
            result = self.metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        # [minwoo] glue task 에서 이 아래로 내려올 일 없을 듯?
        elif self.is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
