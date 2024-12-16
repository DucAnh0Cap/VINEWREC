from torch.utils.data import Dataset
from transformers import AutoTokenizer
from utils import get_users
import torch
from tqdm.auto import tqdm
from nltk import ngrams
import random

class TestSamples(Dataset):
    '''
    This class is used for generating test samples for evaluation
    '''
    def __init__(self, config, data, full_data):
        self.data = data
        self.users = get_users(data)
        self.tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
        self.trigram_dim = config['DATA']['TRIGRAM_DIM']
        self.num_items = config['DATA']['NUM_ITEMS']  # Customizable number of articles
        self.samples = []

        # Prepare article descriptions and IDs
        full_data.sort_values(by='article_id', inplace=True)
        unique_ids = list(full_data.article_id.unique())
        self.article_desc = dict(zip(unique_ids, full_data.description))

        # Precompute user samples without tokenization or tensor conversions
        for _, user in tqdm(enumerate(self.users)):
            if 'articles_id' not in user:
                user['article_ids'] = []

            interacted_ids = set(user['articles_id'])
            non_interacted_ids = list(set(unique_ids) - interacted_ids)

            # Ensure `self.num_items` includes all interacted articles
            sampled_non_interacted = random.sample(
                non_interacted_ids, max(0, self.num_items - len(interacted_ids))
            )
            selected_ids = list(interacted_ids) + sampled_non_interacted
            random.shuffle(selected_ids)  # Shuffle to mix interacted and non-interacted

            # Generate labels (1 for interacted, 0 for non-interacted)
            labels = [1 if article_id in interacted_ids else 0 for article_id in selected_ids]

            # Generate trigrams
            trigrams = []
            user_comments = data.loc[data.usr_id == user['usr_id']].user_comment.to_list()
            for comment in user_comments:
                for grams in ngrams(comment.split(), 3):
                    trigrams.append(' '.join(grams))

            self.samples.append({
                'id': user['usr_id'],
                'selected_ids': selected_ids,
                'comments': ['. '.join(user['comments']) for _ in range(len(selected_ids))],
                'labels': labels,
                'usr_interacted_rates': [user['interacted_rate'] for _ in range(len(selected_ids))],
                'usr_trigram': trigrams
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def collate_fn(self, batch):
        # Extract data from the batch
        ids = [item['id'] for item in batch]
        selected_ids = [item['selected_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        comments = [item['comments'] for item in batch]
        usr_interacted_rates = [torch.tensor(item['usr_interacted_rates']) for item in batch]
        usr_trigrams = [item['usr_trigram'] for item in batch]

        # Tokenize and pad article descriptions
        article_descs = [
            [self.article_desc[article_id] for article_id in batch_item['selected_ids']]
            for batch_item in batch
        ]
        article_desc_tokenized = [
            self.tokenizer(descs, padding="max_length", max_length=150, truncation=True, return_tensors='pt').input_ids
            for descs in article_descs
        ]

        # Tokenize user comments
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
                trigrams_tokenized = torch.empty(0, self.trigram_dim, dtype=torch.long)  # Empty tensor if no trigrams
            usr_trigrams_tokenized.append(torch.stack([trigrams_tokenized for _ in range(len(selected_ids[i]))]))

        return {
            'id': ids,
            'article_ids': selected_ids,
            'usr_comments': torch.stack(comments_tokenized),
            'descriptions': torch.stack(article_desc_tokenized),
            'labels': torch.tensor(labels, dtype=torch.float),
            'usr_interacted_rates': torch.stack(usr_interacted_rates),
            'usr_trigram': torch.stack(usr_trigrams_tokenized)
        }
