import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .eval import Evaluator

class Rel2abs_Decoder:
    def __init__(self, 
                 args, 
                 source_soft_prompt, 
                 source_anchor_embedding, 
                 target_anchor_embedding, 
                 target_statistic):
        
        """
        [minwoo]
        source_soft_prompt : torch.Tensor (pre_seq_len, hidden_size) 
        source_anchor_embedding : torch.Tensor (num_anchor, hidden_size)
        """
        self.args = args
        
        self.absolute = args.absolute
        self.topk = args.topk
        self.budget = args.budget
        self.learning_rate = args.lr
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.evaluator = Evaluator(
            model_name = args.target_model_name,
            dataset_name = args.dataset_name,
            device = self.device,
            prompt_filename = args.prompt_filename)
        
        self.source_soft_prompt = source_soft_prompt.to(self.device)
        self.source_anchor_embedding = source_anchor_embedding.to(self.device)
        self.target_anchor_embedding = target_anchor_embedding.to(self.device)
        
        self.mean = target_statistic[0].to(self.device)
        self.std = target_statistic[1].to(self.device)

        self.source_relative = self.proj2rel(self.source_soft_prompt, self.source_anchor_embedding)

        if self.topk > 0:
            self.source_relative, self.mask = self.zero_except_topk(self.source_relative)

        self.target_soft_prompt = torch.empty((self.source_relative.shape[0], target_anchor_embedding.shape[1])).to(self.device) # (pre_seq_len, target_hidden_size)
        
        self.target_soft_prompt.requires_grad = True
        torch.nn.init.xavier_normal_(self.target_soft_prompt)
        
        self.cos_loss = nn.CosineEmbeddingLoss()
        self.y = torch.ones(self.source_relative.shape[0]).to(self.device) 
        # [minwoo] anchor 의 서로 다른 embedding 에 대하여 비교하므로, y=1 (두 벡터가 유사함) 을 정답으로 설정함.

        # self.logger = logger

        # self.target_abss = None

        # non_zero_indices = self.source_relative.nonzero(as_tuple=True)
        # non_zero_values = self.source_relative[non_zero_indices]
        
        # mean = non_zero_values.mean().item()
        # std = non_zero_values.std().item()
        
        # self.logger.info('Relative representations stat: mean %.4f, std %.4f' % (mean, std))

    def zero_except_topk(self, tensor_relative: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        if self.absolute:
            _, topk_indices = torch.topk(torch.abs(tensor_relative), self.topk)
        else:
            _, topk_indices = torch.topk(tensor_relative, self.topk)

        mask = torch.zeros_like(tensor_relative).to(self.device)
        mask.scatter_(-1, topk_indices, 1)
        masked_tensor = tensor_relative * mask
        return masked_tensor, mask
    
    def regularize_tensor(self, tensor):
        
        current_mean = torch.mean(tensor).to(self.device)
        current_std = torch.std(tensor).to(self.device)
        
        normalized_tensor = (tensor - current_mean) / current_std
        regularized_tensor = normalized_tensor * self.std + self.mean

        return regularized_tensor
    
    def proj2rel(self, 
                 tensor_absolute: torch.Tensor, 
                 anchor_embedding: torch.Tensor) -> torch.Tensor:
        
        A = F.normalize(tensor_absolute, dim=-1) # (pre_seq_len, hidden_size)
        B = F.normalize(anchor_embedding, dim=-1) # (anchor_num, hidden_size)
        
        return torch.matmul(A, B.T) # (pre_seq_len, anchor_num)
    
    # def set_target_abs(self, y):
    #     self.target_abss = torch.tensor(y).type(torch.float32)
    #     self.target_abss.requires_grad = False

    # def eval(self, x):
    #     if self.target_abss == None:
    #         raise AssertionError('No source_soft_prompt abs embedding defined?')
    #     cosine = nn.functional.cosine_similarity(x, self.target_abss, dim=-1)
    #     return torch.mean(cosine).item()
    
    def search(self) -> torch.Tensor:
        optimizer = optim.Adam([self.target_soft_prompt], lr=self.learning_rate)

        best_precision = -1
        best_target_soft_prompt = None

        pbar = tqdm(range(self.budget))
        for i in pbar:
            
            regularized_target_soft_prompt = self.regularize_tensor(self.target_soft_prompt)
            target_relative = self.proj2rel(regularized_target_soft_prompt, self.target_anchor_embedding)
            
            if self.topk > 0:
                target_relative = target_relative * self.mask

            loss = self.cos_loss(target_relative, self.source_relative, self.y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if (i+1) % 1000 == 0 :
                with torch.no_grad():
                    current_target_soft_prompt = regularized_target_soft_prompt.detach()
                    precision = self.evaluator.eval(soft_prompt = current_target_soft_prompt)['accuracy']
                
                if precision > best_precision:
                    best_target_soft_prompt = current_target_soft_prompt
                    best_precision = precision
                    print('Get best precision: %.4f at step %d! loss: %.4f' % (best_precision, i+1, loss.item()))
            
            pbar.set_description('best precision: %.4f, loss: %.4f' % (best_precision, loss.item()))

        with torch.no_grad():
            if best_target_soft_prompt is None:
                best_target_soft_prompt = regularized_target_soft_prompt.detach()
            
            test_precision = self.evaluator.eval(soft_prompt = best_target_soft_prompt)['accuracy']

        print('Test precision: %.4f' % test_precision)

        return best_target_soft_prompt