from transformers import AutoModel, AutoTokenizer
import numpy as np
import torch

def build_vocab(tokenizer) -> tuple[list,dict[int,str]]:
    vocab = list(tokenizer.get_vocab())
    inverse_vocab = {w: i for i, w in enumerate(vocab)}
    return vocab, inverse_vocab

def vocab_cleaning(vocab):
    vocab = [token.replace("Ġ", "") for token in vocab] # for roberta
    vocab = [token.replace("▁", "") for token in vocab] # for albert
    return vocab

def select_random_samples(matrix: torch.Tensor,
                          indices: np.ndarray):
    """
    [minwoo]
    args:
        matrix : embedding weight array
        indices : array([int])
    """
    
    k = matrix.shape[1] # hidden_dim (e.g. 768 for BERT)
    m = indices.shape[0] # 0 ~ vocab size
    
    selected_embeddings = torch.empty((m, k)) # [minwoo] m x k matrix (vocab_size, hidden_dim)
    
    for i, idx in enumerate(indices):
        selected_embeddings[i] = matrix[idx]
    
    return selected_embeddings

def get_absolute_anchors(source_model_name: str,
                         target_model_name: str,
                         num_anchor = 1024,
                         all_anchors = False):
    """
    [minwoo]
        source_model_name:
            Name or path of the source model.
        target_model_name:
            Name or path of the target model.
        num_anchor:
            Number of anchor points to select.
        common_vocab: [minwoo] remove.
            Path to a file containing common vocabulary (optional).
        all_anchors:
            Boolean to decide whether to use all anchor points or a subset.
    """
    
    source_model = AutoModel.from_pretrained(source_model_name)
    target_model = AutoModel.from_pretrained(target_model_name)
    
    source_tokenizer = AutoTokenizer.from_pretrained(source_model_name)
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    
    source_unknwon_token_id = source_tokenizer.unk_token_id
    target_unknwon_token_id = target_tokenizer.unk_token_id
    
    source_embedding_layer = source_model.get_input_embeddings().weight.data.detach().clone()#.cpu()
    target_embedding_layer = target_model.get_input_embeddings().weight.data.detach().clone()#.cpu()
    
    
    # if common_vocab != None:
    #     with open(common_vocab, 'r') as f:
    if False:
        lines = f.readlines()
        filtered_vocab = [x.strip() for x in lines]

    else:
        source_vocab, _ = build_vocab(source_tokenizer)
        target_vocab, _ = build_vocab(target_tokenizer)

        print('model 1 and 2 vocabs:', len(source_vocab), len(target_vocab))
        source_vocab = vocab_cleaning(source_vocab)
        target_vocab = vocab_cleaning(target_vocab)

        filtered_vocab = list(set(source_vocab) & set(target_vocab))

    source_filtered_indices = list()
    target_filtered_indices = list()
    
    for word in filtered_vocab:
        # [minwoo] 일부 tokenizer 의 경우 앞에 공백이 없으면 다르게 해석할 수 있다고 함.
        source_tokens = source_tokenizer.tokenize(' '+word)
        
        if (len(source_tokens) == 1) and (source_tokens[0] != source_unknwon_token_id):
            source_index = source_tokenizer.convert_tokens_to_ids(source_tokens)[0]
        else:
            continue
        
        target_tokens = target_tokenizer.tokenize(' ' + word)
        if (len(target_tokens) == 1) and (target_tokens[0] != target_unknwon_token_id):
            target_index = target_tokenizer.convert_tokens_to_ids(target_tokens)[0]
        else:
            continue
        
        # [minwoo] 같은 word(filtered) 에 대하여 순차적으로 해당 word 의 id를 저장 -> 순서중요.
        source_filtered_indices.append(source_index)
        target_filtered_indices.append(target_index)

    assert len(source_filtered_indices) == len(target_filtered_indices)
    
    print('Number of filtered shared tokens %d' % len(target_filtered_indices))
    
    # [minwoo]TODO 원본 layer 에서의 mean,std 를 구하네? 왜지?
    mean = torch.mean(target_embedding_layer.reshape(-1)) 
    std = torch.std(target_embedding_layer.reshape(-1))
    
    if all_anchors:
        source_anchor_embedding = select_random_samples(source_embedding_layer, np.array(source_filtered_indices))
        target_anchor_embedding = select_random_samples(target_embedding_layer, np.array(target_filtered_indices))
        
    else:
        # source_filtered_indices 에서 num_anchor 만큼 랜덤하게 선택
        selected_indices = np.random.choice(list(range(0, len(source_filtered_indices))), num_anchor, replace=False)
        source_anchor_embedding = select_random_samples(source_embedding_layer, np.array([source_filtered_indices[i] for i in selected_indices]))
        target_anchor_embedding = select_random_samples(target_embedding_layer, np.array([target_filtered_indices[i] for i in selected_indices]))
        
    return source_anchor_embedding, target_anchor_embedding, (mean, std), 


if __name__ == '__main__':
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # vocab, inverse_vocab = build_vocab(tokenizer)
    # print(vocab)
    # print(inverse_vocab)
    
    # source_embedding_layer, target_embedding_layer, (mean, std) = get_absolute_anchors(
    #     source_model_name = "bert-base-uncased",
    #     target_model_name = "bert-large-uncased",
    #     num_anchor = 1024,
    #     common_vocab = None,
    #     seed = 42,
    #     all_anchors = False
    # )
    
    # print(source_embedding_layer.shape)
    # print(target_embedding_layer.shape)
    # print(mean, std)
    
    pass