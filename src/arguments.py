from dataclasses import dataclass, field
from typing import Optional

GLUE_DATASETS = [
    "cola",
    "mnli",
    "mrpc",
    "qnli",
    "qqp",
    "rte",
    "sst2",
    "stsb",
    "wnli"
]

TASKS = ["glue"]

DATASETS = GLUE_DATASETS

@dataclass
class Arguments:
    model_name: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    
    