import os
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import get_articles, get_users
from torch.nn.utils.rnn import pad_sequence
import torch

class NewsDataset(Dataset):
    def __init__(self, news: pd.DataFrame, users: pd.DataFame):
        self.news_df = news
        self.user_df = users
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

        usr_hist_lst = []
        df = self.user_df.loc[self.user_df.Id.isin(batch_dict['readers'])]

        # Get interacted_rate and interacted_categories
        batch_dict['interacted_categories'] = []
        batch_dict['interacted_rates'] = []
        users = get_users(self.news_df, df)
        
        # Create a dictionary to store interacted_rates for each user
        user_interacted_rates = {} 

        for usr in users:
            user_interacted_rates[usr['Id']] = (torch.tensor(usr['interacted_rate'], dtype=torch.float32),
                                                torch.tensor(usr['interacted_categories'], dtype=torch.int64))

        # Now iterate through batch_dict['readers'] and retrieve the rate from the dictionary
        for id in batch_dict['readers']:
            if id in user_interacted_rates:
                rate, categories = user_interacted_rates[id]
                batch_dict['interacted_rates'].append(rate)
                batch_dict['interacted_categories'].append(categories)


        # Get trigrams
        for id in batch_dict['readers']:
            comments = df.loc[df.Id == id]
            usr_hist_lst.append(comments[comments.Id == id].user_comment.to_list()) 
        
        trigrams = []
        for comments in usr_hist_lst:
            lst = []
            for c_ in comments:
                sixgrams = ngrams(c_.split(), 3)
                for grams in sixgrams:
                    lst.append(' '.join(grams))
            trigrams.append(lst)
        
        # Tokenize
        batch_dict['trigram_ids'] = []
        for tri_ in trigrams:
            tokenize_trigrams = self.tokenizer(tri_, padding="max_length", max_length=100, truncation=True, return_tensors='pt').input_ids
            batch_dict['trigram_ids'].append(tokenize_trigrams)
        batch_dict['trigram_ids'] = pad_sequence(batch_dict['trigram_ids'], batch_first=True, padding_value=0)

        batch_dict['interacted_categories'] = torch.stack(batch_dict['interacted_categories'])
        batch_dict['interacted_rates'] = torch.stack(batch_dict['interacted_rates'])
        batch_dict['descriptions'] = self.tokenizer(batch_dict['descriptions'], padding="max_length", max_length=150, truncation=True, return_tensors='pt').input_ids
        batch_dict['comments'] = self.tokenizer(batch_dict['comments'], padding="max_length", max_length=150, truncation=True, return_tensors='pt').input_ids
        return batch_dict

