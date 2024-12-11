import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from nltk import ngrams
from utils import get_users
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
            'articles_ids': [item['article_id'] for item in batch],
            'comments': [item['comments'] for item in batch],
            'categories': [item['categories'] for item in batch],
            'interacted_categories': [item['interacted_categories'] for item in batch],
            'interacted_rate': [item['interacted_rate'] for item in batch]
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
        batch_dict['interacted_categories'] = torch.stack(batch_dict['interacted_categories'])
        batch_dict['interacted_rate'] = torch.stack(batch_dict['interacted_rate'])
        return batch_dict
        
