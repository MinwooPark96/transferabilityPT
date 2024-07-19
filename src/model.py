from typing import Optional, Union, Tuple
import logging

from transformers.modeling_outputs import MaskedLMOutput
from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM

from utils import freeze_params

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

logger = logging.getLogger(__name__)

class MLMPromptForEval(nn.Module):
    """
    [minwoo] It is for evaluating the transferability of the prompt.
    """
    def __init__(self,
                 model_name: str,
                 dataset_name: str,
                *args,
                **kwargs
                ):
        
        super(MLMPromptForEval,self).__init__()
        
        self.model = AutoModelForMaskedLM.from_pretrained(model_name) # Pretrained MLM model
        self.config = self.model.config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config._name_or_path)
        self.finetuning_task = dataset_name
        
        self.normal_embedding_layer = self.model.get_input_embeddings() # word embedding layer of the pretrained model
        self.hidden_size = self.model.config.hidden_size
        
        self.mask_ids = torch.tensor([self.tokenizer.mask_token_id])
        
        if self.finetuning_task == 'stsb':
            NotImplementedError("[minwoo] STSB task is not supported yet... It is regression task.")
        
        freeze_params(self.model)
        
        bert_map = {'positive' : 3893,'negative' : 4997,'yes' : 2748,'neutral' : 8699,'no' : 2053,'true' : 2995,'false' : 6270}
        roberta_map = {'positive' : 22173,'negative' : 2430,'yes' : 4420,'neutral' : 7974,'no' : 117,'true' : 1528,'false' : 3950}
        
        self.map = bert_map if 'bert' in model_name else roberta_map
        
    def get_soft_prompt(self):
        """Return the soft prompt."""
        return self.soft_prompt

    def get_word_embedding_layer(self):
        """Return the word embedding layer."""
        return self.normal_embedding_layer

    def set_soft_prompt(self, soft_prompt: torch.Tensor):
        """Set the soft prompt."""
        self.soft_prompt = soft_prompt
        self.pre_seq_len = soft_prompt.shape[0]
        
        # [minwoo] freeze the soft prompt
        self.soft_prompt.requires_grad = False
        
        assert self.hidden_size == self.soft_prompt.shape[1]
        
        # logger.info(f"Set the soft prompt with shape {self.soft_prompt.shape}")
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        *args,
        **kwargs) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        
        batch_size = input_ids.shape[0] #[minwoo] batch_size 가 edge 부분에서 달라질 수 있으므로, 여기서 계산하는 것이 맞음.
        mask_ids = torch.stack([self.mask_ids for _ in range(batch_size)]).to(input_ids.device)

        # [minwoo] each_embeddings.shape = (batch_size, each , hidden_size)
        mask_embeddings = self.normal_embedding_layer(mask_ids)
        soft_embeddings = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1) 
        text_embeddings = self.normal_embedding_layer(input_ids)
        
        mask_attention_mask = torch.ones(batch_size, 1).to(input_ids.device)
        soft_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(input_ids.device)
        
        input_embeddings = torch.cat([mask_embeddings, soft_embeddings,text_embeddings], dim = 1)
        attention_mask = torch.cat([mask_attention_mask, soft_attention_mask, attention_mask],dim=1)
        
        assert attention_mask.shape[1] == input_embeddings.shape[1] # == 1 + pre_seq_len + text_seq_len
        
        model_outputs = self.model( # AutoModelForMaskedLM
            attention_mask = attention_mask,
            inputs_embeds = input_embeddings,
        ) # -> MaskedLMOutput(loss = None, logits = tensor(batch, total_length, vacab_size))
        
        # [minwoo] logits.shape = (batch, total_length, vocab_size)
        logits = model_outputs.logits
        mask_logits = logits[:,0]

            
        """
        [minwoo] setting of https://aclanthology.org/2022.naacl-main.290.pdf
            TODO : 
                I think it is better to utilize 
                    1. Manual Template? or Pattern?. 
                    2. Verbalizer?
                        see PET : https://arxiv.org/abs/2001.07676
                        
        SA (Sentiment Analysis)
            IMDB: positive, negative 
            SST-2: positive, negative
                Bert : 3893, 4997
                Roberta : 22173, 2430

        NLI (Natural Language Inference)**: 
            MNLI: yes, neutral, no
                Bert : 2748, 8699, 2053
                Roberta : 4420, 7974, 117
    
    
        PI (Paraphrase Identification)**:
            QQP: true, false
            MRPC: true, false
                Bert : 2995, 6270
                Roberta : 1528, 3950
        
        """
        
        if self.finetuning_task in ['sst2', 'imdb']:
            #score = torch.cat([mask_logits[:,3893].unsqueeze(1), mask_logits[:,4997].unsqueeze(1)],dim = 1)
            score = torch.cat([mask_logits[:,self.map['positive']].unsqueeze(1), mask_logits[:,self.map['negative']].unsqueeze(1)],dim = 1)
        elif self.finetuning_task in ['mnli']:
            #score = torch.cat([mask_logits[:,2748].unsqueeze(1), mask_logits[:,8699].unsqueeze(1), mask_logits[:,2053].unsqueeze(1)],dim = 1)
            score = torch.cat([mask_logits[:,self.map['yes']].unsqueeze(1), mask_logits[:,self.map['neutral']].unsqueeze(1), mask_logits[:,self.map['no']].unsqueeze(1)],dim = 1)
        elif self.finetuning_task in ['qqp', 'mrpc']:
            # score = torch.cat([mask_logits[:,2995].unsqueeze(1), mask_logits[:,6270].unsqueeze(1)], dim = 1)
            score = torch.cat([mask_logits[:,self.map['true']].unsqueeze(1), mask_logits[:,self.map['false']].unsqueeze(1)], dim = 1)
        else :
            NotImplementedError(f"[minwoo] {self.finetuning_task} is not supported yet...")
        
        return score

if __name__ == '__main__':
    
    import argparse
    # python src/models/modeling_bert.py --task_name glue --dataset_name sst2 --source_model_name bert-base-uncased --target_model_name bert-large-uncased  --output_dir ./output --prompt_path ./output/bert-base-uncased/sst2/epoch_100_prompt.bin
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    args.dataset_name = 'sst2'
    
    model = MLMPromptForEval(args = args)
