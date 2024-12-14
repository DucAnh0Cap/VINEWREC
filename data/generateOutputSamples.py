from torch.utils.data import Dataset
from transformers import AutoTokenizer
from utils import get_users
import torch


class TestSamples(Dataset):
    def __init__(self, news, users):
        self.users = get_users(news, users)
        self.tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")

        self.samples = []
        for user in self.users:
            if 'article_ids' not in user:
                user['article_ids'] = []  
            
            remain_ids = 50 - len(user['article_ids'])
            comments = ['. '.join(user['comments']) for i in range(50)]
            
            # Filter news articles not in user's history and sample
            df = news[~news.article_id.isin(user['article_ids'])].sample(remain_ids)
            
            # Combine user's article ids with sampled ids
            article_ids = user['article_ids']
            
            # Generate labels
            labels = torch.zeros(50)
            labels[:len(user['article_ids'])] = 1

            article_ids.extend(df.article_id.to_list())

            # Get article descriptions, handling potential missing descriptions
            article_desc = []
            for id in article_ids:
                desc = news.loc[news.article_id == id, 'description']
                # Check if description exists and is not empty
                article_desc.append(desc.iloc[0] if not desc.empty else '') 


            self.samples.append({
                'id': user['Id'],
                'comments': comments,
                'article_ids': article_ids,
                'descriptions': article_desc,
                'labels': labels
            })

            # Tokenize article descriptions and comments within the loop, after appending to self.samples
            self.samples[-1]['descriptions'] = self.tokenizer(self.samples[-1]['descriptions'], padding="max_length", max_length=150, truncation=True, return_tensors='pt').input_ids
            self.samples[-1]['comments'] = self.tokenizer(self.samples[-1]['comments'], padding="max_length", max_length=150, truncation=True, return_tensors='pt').input_ids
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]