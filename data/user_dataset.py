import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from nltk import ngrams
import torch
from DS300.utils import get_users

class userDataset(Dataset):
    def __init__(self, news, users):
        self.users = get_users(news, users)
        self.tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx]

    def collate_fn(self, batch):
        batch_dict = {
            'Ids': [item['Id'] for item in batch],
            'news_ids': [item['news_id'] for item in batch],
            'comments': [item['comments'] for item in batch],
            'categories': [item['categories'] for item in batch],
            'interactec_rate': [item['interactec_rate'] for item in batch]
        }

        trigrams = []
        for comments in batch_dict['comments']:
            lst = []
            for c_ in comments:
                sixgrams = ngrams(c_.split(), 3)
                for grams in sixgrams:
                    lst.append(' '.join(grams))
            trigrams.append(lst)

        batch_dict['trigrams'] = []
        for tri_ in trigrams:
            tokenize_trigrams = self.tokenizer(tri_, padding="max_length", max_length=50, truncation=True, return_tensors='pt').input_ids
            batch_dict['trigrams'].append(tokenize_trigrams)
        batch_dict['trigrams'] = torch.stack(batch_dict['trigrams'])
        return batch_dict
        
