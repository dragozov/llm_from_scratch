import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDatasetV1(Dataset):
    def __init__(self, txt, max_length, stride, tokenizer = None):
        if tokenizer is None:
            self.tokenizer = default_tokenizer()
        else:
            self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []
        
        token_ids = self.tokenizer.encode(txt)
        
        for i in range(0, len(token_ids) - max_length - 1, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i+1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
            

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
def default_tokenizer():
    return tiktoken.get_encoding('gpt2')

def create_data_loader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, tokenizer = None):
    dataset = GPTDatasetV1(txt, max_length, stride, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader

def create_embedding_layer(vocab_size, dim):
    return torch.nn.Embedding(vocab_size, dim)
    
def main():
    with open('the-verdict.txt', 'r') as f:
        data = f.read()
        
    tokenizer = default_tokenizer()
    vocab_size = tokenizer.max_token_value + 1
    output_dim = 256
    emb_layer = create_embedding_layer(vocab_size, output_dim)
    max_length = 4
    dataloader = create_data_loader(data, batch_size=8, max_length=max_length, stride=max_length, shuffle=False, tokenizer=tokenizer)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs: %s" % inputs)
    
    token_embeddings = emb_layer(inputs)
    print("Embedding", token_embeddings.shape)
    
    ctx_len = max_length
    positional_emb_layer = torch.nn.Embedding(ctx_len, output_dim)
    positional_embeddings = positional_emb_layer(torch.arange(ctx_len))
    print("Positional Embeddings", positional_embeddings.shape)
    
    input_embeddings = token_embeddings + positional_embeddings
    print("Input Embeddings", input_embeddings.shape)
    
if __name__ == '__main__':
    main()