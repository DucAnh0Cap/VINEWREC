from torch.utils.data import Dataset
from transformers import AutoTokenizer
from utils import get_users
import torch
from tqdm.auto import tqdm
from nltk import ngrams

class TestSamples(Dataset):
    '''
    This class is used for generating test sample for evaluation
    '''
    def __init__(self, config, data, full_data):
        self.data = data
        self.users = get_users(data)
        self.tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
        self.num_items = 50
        self.trigram_dim = config['DATA']['TRIGRAM_DIM']
        self.samples = []

        # Create a dict template
        full_data.sort_values(by='article_id', inplace=True) 

        unique_ids = full_data.article_id.unique()
        ids_dict = {}
        for id in unique_ids:
            ids_dict[id] = 0
        self.num_items = len(unique_ids)

         # Get all article description
        articles = full_data[full_data.article_id.isin(unique_ids)][['article_id', 'description']]
        articles = articles.drop_duplicates()
        article_desc = articles.description.to_list()

        # Loop through users
        for _, user in tqdm(enumerate(self.users)):
            if 'articles_id' not in user:
                user['article_ids'] = []  

            comments = ['. '.join(user['comments']) for i in range(self.num_items)]

            # Generate labels
            label_map = ids_dict.copy()
            for id in user['articles_id']:
                label_map[id] += 1
            labels = list(label_map.values())
            
            # Get trigrams
            trigrams = []
            comments_ = data.loc[data.usr_id == user['usr_id']].user_comment.to_list()
            for c_ in comments_:
                sixgrams = ngrams(c_.split(), 3)
                for grams in sixgrams:
                    trigrams.append(' '.join(grams))

            self.samples.append({
                'id': user['usr_id'],
                'usr_comments': comments,
                'article_ids': unique_ids,
                'descriptions': article_desc,
                'labels': labels,
                'usr_interacted_rates': torch.stack([torch.tensor(user['interacted_rate']) for i in range(self.num_items)]),
                'usr_trigram': trigrams
            })
            # Tokenize article descriptions and comments within the loop, after appending to self.samples
            self.samples[-1]['descriptions'] = self.tokenizer(self.samples[-1]['descriptions'], padding="max_length", max_length=150, truncation=True, return_tensors='pt').input_ids
            self.samples[-1]['usr_comments'] = self.tokenizer(self.samples[-1]['usr_comments'], padding="max_length", max_length=150, truncation=True, return_tensors='pt').input_ids
            # Check if trigrams is not empty before tokenizing
            if self.samples[-1]['usr_trigram']:
                self.samples[-1]['usr_trigram'] = self.tokenizer(self.samples[-1]['usr_trigram'], padding="max_length", max_length=self.trigram_dim, truncation=True, return_tensors='pt').input_ids
            else:
                self.samples[-1]['usr_trigram'] = torch.empty(0, 150, dtype=torch.long) # Assign an empty tensor if trigrams is empty
            trigrams = self.samples[-1]['usr_trigram']
            self.samples[-1]['usr_trigram'] = torch.stack([trigrams for i in range(self.num_items)])

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]