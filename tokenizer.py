import transformers
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader


class Tokenizer_RU_EN():
    def __init__(self, tokenizer = "ru_gpt3_tokenizer"):
        """
        Args:
            tokenizer (str): путь к локальному токинизатору или huggingface. Defaults to "ru_gpt3_tokenizer".
        """
        
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained(tokenizer)
        
    def pad_token_id(self, pytorch = True):
        if pytorch:
            return torch.tensor(self.tokenizer.pad_token_id)
        else:
            return self.tokenizer.pad_token_id
    
    def vocab_size(self, pytorch = True):
        if pytorch:
            return torch.tensor(self.tokenizer.vocab_size)
        else:
            return self.tokenizer.vocab_size

    def tokenize_text(self, text, pytorch = True):
        if pytorch:
            return torch.tensor(self.tokenizer.tokenize(text))
        else:
            return self.tokenizer.tokenize(text)
    
    def encode_text(self, text, pytorch = True):
        if pytorch:
            return torch.tensor(self.tokenizer.encode(text))
        else:
            return self.tokenizer.encode(text)

    def decode_text(self, token_ids, pytorch = True):
        if pytorch:
            token_ids = torch(self.tokenizer.decode(token_ids))
        else:
            return self.tokenizer.decode(token_ids, )
        

class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_length = 256):
        self.texts_df = df
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.tokenizer.pad_token_id
        self.max_length = max_length

    def __len__(self):
        return len(self.texts_df)

    def __getitem__(self, idx):
        ru_text, en_text = self.texts_df.iloc[idx]
        
        ru_text = self.tokenizer.encode_text(ru_text)
        en_text = self.tokenizer.encode_text(en_text)
        
        assert len(ru_text) < self.max_length and len(en_text) < self.max_length, "encoded text is longer than max_length"
        
        if len(ru_text) < self.max_length:
            ru_text = torch.cat([ru_text, torch.ones(self.max_length - len(ru_text)) * self.pad_token_id], dim=0)
            
        if len(en_text) < self.max_length:
            en_text = torch.cat([en_text, torch.ones(self.max_length - len(en_text)) * self.pad_token_id], dim=0)
        
        return ru_text, en_text