import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import get_articles

class NewsDataset(Dataset):
    def __init__(self, news, users):
        self.news = get_articles(news, users)
        # self.users = get_users(news, users)
        self.tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")

        self.samples = []
        for news_ in self.news:
            for i in range(len(news_['comments'])):
                self.samples.append({
                    'title': news_['title'],
                    'article_id': news_['article_id'],
                    'author_name': news_['author_name'],
                    'description': news_['description'],
                    'content': news_['content'],
                    'reader': news_['reader'][i],
                    'category': news_['category'],
                    'comment': news_['comments'][i]
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def collate_fn(self, batch):
        # Combine values for each key
        batch_dict = {
            "article_ids": [item["article_id"] for item in batch],
            "descriptions": [item["description"] for item in batch],
            "readers": [item["reader"] for item in batch],
            "categories": [item["category"] for item in batch],
            "comments": [item["comment"] for item in batch],
        }

        batch_dict['descriptions'] = self.tokenizer(batch_dict['descriptions'], padding="max_length", max_length=150, truncation=True, return_tensors='pt').input_ids
        batch_dict['comments'] = self.tokenizer(batch_dict['comments'], padding="max_length", max_length=150, truncation=True, return_tensors='pt').input_ids
        return batch_dict

