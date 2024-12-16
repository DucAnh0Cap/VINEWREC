from torch.utils.data import Dataset
from transformers import AutoTokenizer
from utils import get_users
import torch
from tqdm.auto import tqdm
from nltk import ngrams

class TestSamples(Dataset):
    '''
    This class is used for generating test samples for evaluation
    '''
    def __init__(self, config, data, full_data):
        self.data = data
        self.users = get_users(data)
        self.tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
        self.trigram_dim = config['DATA']['TRIGRAM_DIM']
        self.num_items = 200
        self.samples = []

        # Prepare article descriptions and IDs
        full_data.sort_values(by='article_id', inplace=True)
        unique_ids = list(full_data.article_id.unique())
        ids_dict = {id: 0 for id in unique_ids}
        self.num_items = len(unique_ids)

        articles = full_data[full_data.article_id.isin(unique_ids)][['article_id', 'description']]
        articles = articles.drop_duplicates()
        self.article_desc = articles.description.to_list()
        self.article_ids = unique_ids
        self.ids_dict_template = ids_dict

        # Precompute user samples without tokenization or tensor conversions
        for _, user in tqdm(enumerate(self.users)):
            if 'articles_id' not in user:
                user['article_ids'] = []

            label_map = ids_dict.copy()
            for id in user['articles_id']:
                label_map[id] += 1

            trigrams = []
            comments_ = data.loc[data.usr_id == user['usr_id']].user_comment.to_list()
            for c_ in comments_:
                sixgrams = ngrams(c_.split(), 3)
                for grams in sixgrams:
                    trigrams.append(' '.join(grams))

            self.samples.append({
                'id': user['usr_id'],
                'comments': ['. '.join(user['comments']) for _ in range(self.num_items)],
                'labels': list(label_map.values()),
                'usr_interacted_rates': [user['interacted_rate'] for _ in range(self.num_items)],
                'usr_trigram': trigrams
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def collate_fn(self, batch):
        # Extract data from the batch
        ids = [item['id'] for item in batch]
        comments = [item['comments'] for item in batch]
        labels = [item['labels'] for item in batch]
        usr_interacted_rates = [torch.tensor(item['usr_interacted_rates']) for item in batch]
        usr_trigrams = [item['usr_trigram'] for item in batch]

        # Tokenize and pad article descriptions and user comments
        article_desc_tokenized = self.tokenizer(self.article_desc, padding="max_length", max_length=150, truncation=True, return_tensors='pt').input_ids
        comments_tokenized = [
            self.tokenizer(comments[i], padding="max_length", max_length=150, truncation=True, return_tensors='pt').input_ids
            for i in range(len(batch))
        ]

        # Process trigrams
        usr_trigrams_tokenized = []
        for i, trigrams in enumerate(usr_trigrams):
            if trigrams:
                trigrams_tokenized = self.tokenizer(trigrams, padding="max_length", max_length=self.trigram_dim, truncation=True, return_tensors='pt').input_ids
            else:
                trigrams_tokenized = torch.empty(0, 150, dtype=torch.long)  # Empty tensor if no trigrams
            usr_trigrams_tokenized.append(torch.stack([trigrams_tokenized for _ in range(self.num_items)]))

        return {
            'id': ids,
            'usr_comments': torch.stack(comments_tokenized),
            'article_ids': torch.tensor(self.article_ids),
            'descriptions': article_desc_tokenized,
            'labels': torch.tensor(labels),
            'usr_interacted_rates': torch.stack(usr_interacted_rates),
            'usr_trigram': torch.stack(usr_trigrams_tokenized)
        }
