import os
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from utils import get_articles, get_users
from torch.nn.utils.rnn import pad_sequence
from nltk import ngrams
import torch


class NewsDataset(Dataset):
    def __init__(self, config, df: pd.DataFrame):
        self.data = df
        self.news = get_articles(self.data)
        # self.users = get_users(news, users)
        self.tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
        self.trigram_dim = config['DATA']['TRIGRAM_DIM']
        self.samples = []
        for news_ in self.news:
            for i in range(len(news_['comments'])):
                self.samples.append({
                    'title': news_['Title'],
                    'article_id': news_['article_id'],
                    'author_name': news_['author_name'],
                    'description': news_['description'],
                    'usr_ids': news_['usr_ids'][i],
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
            "usr_ids": [item["usr_ids"] for item in batch],
            "usr_categories": [item["category"] for item in batch],
            "usr_comments": [item["comment"] for item in batch],
        }

        usr_hist_lst = []
        
        # Get NLI Scores
        batch_dict['nli_scores'] = self.data.loc[(self.data.article_id.isin(batch_dict['article_ids']) & 
                                                  self.data.usr_id.isin(batch_dict['usr_ids']))].nli_score.to_list()

        # Get interacted_rate and interacted_categories
        batch_dict['usr_interacted_categories'] = []
        batch_dict['usr_interacted_rates'] = []
        batch_dict['usr_tags'] = []
        batch_dict['nli_scores'] = []
        # Get users with coresponding Ids
        df = self.data.loc[self.data.usr_id.isin(batch_dict['usr_ids'])]
        users = get_users(df)
        
        # Store interacted_rates, interacted_categories and usr_tags for each user
        user_interacted_rates = {} 
        user_tags = {}

        for usr in users:
            user_interacted_rates[usr['usr_id']] = (torch.tensor(usr['interacted_rate'], dtype=torch.float32),
                                                    torch.tensor(usr['interacted_categories'], dtype=torch.int64))
            user_tags[usr['usr_id']] = ' '.join(usr['tags'])
            # batch_dict['nli_scores'].append(usr['nli_scores'])
        
        for i in range(len(batch_dict['article_ids'])):
            batch_dict['nli_scores'].append(df.loc[(df.article_id==batch_dict['article_ids'][i]) & (df.usr_id==batch_dict['usr_ids'][i])].nli_score.iloc[0])

        # Retrieve the rate from the dictionary
        for id in batch_dict['usr_ids']:
            if id in user_interacted_rates:
                rate, categories = user_interacted_rates[id]
                usr_tag = user_tags[id]
                batch_dict['usr_interacted_rates'].append(rate)
                batch_dict['usr_interacted_categories'].append(categories)
                batch_dict['usr_tags'].append(usr_tag)
                
        # Get trigrams
        for id in batch_dict['usr_ids']:
            comments = df.loc[df.usr_id == id]
            usr_hist_lst.append(comments[comments.usr_id == id].user_comment.to_list()) 
        
        trigrams = []
        for comments in usr_hist_lst:
            lst = []
            for c_ in comments:
                sixgrams = ngrams(c_.split(), 3)
                for grams in sixgrams:
                    lst.append(' '.join(grams))
            trigrams.append(lst)
        
        # Tokenize
        batch_dict['usr_trigram'] = []
        for tri_ in trigrams:
            if tri_:  
                tokenize_trigrams = self.tokenizer(tri_, padding="max_length", max_length=self.trigram_dim, truncation=True, return_tensors='pt').input_ids
                batch_dict['usr_trigram'].append(tokenize_trigrams)
            else:
                # If tri_ is empty, append an empty tensor with the correct shape
                batch_dict['usr_trigram'].append(torch.empty(0, self.trigram_dim, dtype=torch.long)) 
            
        batch_dict['usr_trigram'] = pad_sequence(batch_dict['usr_trigram'], batch_first=True, padding_value=0)
        
        batch_dict['nli_scores'] = torch.tensor(batch_dict['nli_scores'])
        batch_dict['usr_interacted_categories'] = torch.stack(batch_dict['usr_interacted_categories'])
        batch_dict['usr_interacted_rates'] = torch.stack(batch_dict['usr_interacted_rates'])
        batch_dict['descriptions'] = self.tokenizer(batch_dict['descriptions'], padding="max_length", max_length=150, truncation=True, return_tensors='pt').input_ids
        batch_dict['usr_comments'] = self.tokenizer(batch_dict['usr_comments'], padding="max_length", max_length=150, truncation=True, return_tensors='pt').input_ids
        batch_dict['usr_tags'] = self.tokenizer(batch_dict['usr_tags'], padding="max_length", max_length=self.trigram_dim, truncation=True, return_tensors='pt').input_ids
        return batch_dict

