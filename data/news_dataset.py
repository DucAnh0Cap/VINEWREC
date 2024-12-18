import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from utils import get_articles, get_users
from torch.nn.utils.rnn import pad_sequence
from nltk import ngrams
import torch


class NewsDataset(Dataset):
    def __init__(self, config, df: pd.DataFrame):
        self.data = df
        self.news = get_articles(self.data)
        self.tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
        self.trigram_dim = config['DATA']['TRIGRAM_DIM']
        self.samples = []

        for news_ in self.news:
            for i in range(len(news_['comments'])):
                self.samples.append({
                    'title': news_['Title'],
                    'article_id': news_['article_id'],
                    'description': news_['description'],
                    'usr_ids': news_['usr_ids'][i],
                    'category': news_['category'],
                    'comment': news_['comments'][i],
                    'label': news_['labels'][i]
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def collate_fn(self, batch):
        batch_dict = {
            "article_ids": [item["article_id"] for item in batch],
            "descriptions": [item["description"] for item in batch],
            "usr_ids": [item["usr_ids"] for item in batch],
            "usr_categories": [item["category"] for item in batch],
            "usr_comments": [item["comment"] for item in batch],
            "labels": [item["label"] for item in batch]
        }

        # Get users with corresponding IDs
        df = self.data.loc[self.data.usr_id.isin(batch_dict['usr_ids'])]
        users = get_users(df)

        # Map user data for efficient access
        user_interacted_rates = {usr['usr_id']: (torch.tensor(usr['interacted_rate'], dtype=torch.float32),
                                                 torch.tensor(usr['interacted_categories'], dtype=torch.int64),
                                                  ' '.join(usr['tags']))
                                 for usr in users}

        # Prepare NLI scores and user related data
        # labels = []
        interacted_rates = []
        interacted_categories = []
        usr_tags = []

        for article_id, usr_id in zip(batch_dict['article_ids'], batch_dict['usr_ids']):
            # label = df.loc[(df.article_id == article_id) & (df.usr_id == usr_id), 'label']
            # labels.append(label.iloc[0] if not label.empty else 0)  # Handle case if score is empty

            if usr_id in user_interacted_rates:
                rate, categories, usr_tag = user_interacted_rates[usr_id]
                interacted_rates.append(rate)
                interacted_categories.append(categories)
                usr_tags.append(usr_tag)
            else:
                interacted_rates.append(torch.zeros(len(batch_dict['usr_categories'][0]), dtype=torch.float32))
                interacted_categories.append(torch.zeros(len(batch_dict['usr_categories'][0]), dtype=torch.int64))
                usr_tags.append('')

        # Get trigrams for user comments
        usr_hist_lst = [df.loc[df.usr_id == id, 'user_comment'].tolist() for id in batch_dict['usr_ids']]
        
        trigrams = []
        for comments in usr_hist_lst:
            lst = [' '.join(grams) for c_ in comments for grams in ngrams(c_.split(), 3)]
            trigrams.append(lst)

        # Tokenize trigrams
        tokenized_trigrams = []
        for tri_ in trigrams:
            if tri_:
                tokenized = self.tokenizer(tri_, padding="max_length", max_length=self.trigram_dim,
                                           truncation=True, return_tensors='pt').input_ids
                tokenized_trigrams.append(tokenized)
            else:
                tokenized_trigrams.append(torch.empty(0, self.trigram_dim, dtype=torch.long))

        batch_dict['usr_trigram'] = pad_sequence(tokenized_trigrams, batch_first=True, padding_value=0)

        # Fill batch_dict
        batch_dict['labels'] = torch.tensor(batch_dict['labels'])
        batch_dict['usr_interacted_categories'] = torch.stack(interacted_categories).long()
        batch_dict['usr_interacted_rates'] = torch.stack(interacted_rates).float()
        batch_dict['descriptions'] = self.tokenizer(batch_dict['descriptions'], padding="max_length", max_length=150,
                                                    truncation=True, return_tensors='pt').input_ids
        batch_dict['usr_comments'] = self.tokenizer(batch_dict['usr_comments'], padding="max_length", max_length=150,
                                                    truncation=True, return_tensors='pt').input_ids
        batch_dict['usr_tags'] = self.tokenizer(usr_tags, padding="max_length", max_length=self.trigram_dim,
                                                truncation=True, return_tensors='pt').input_ids
        return batch_dict