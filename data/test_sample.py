from torch.utils.data import Dataset
from transformers import AutoTokenizer
from utils import get_users
import torch
from tqdm.auto import tqdm
from nltk import ngrams
import random


class TestSamples(Dataset):
    '''
    This class is used for generating test samples for evaluation.
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
        self.article_desc = dict(zip(full_data.article_id, full_data.description))

        # Precompute user samples without tokenization or tensor conversions
        for user in tqdm(self.users):
            interacted_ids = set(user.get('articles_id', []))
            non_interacted_ids = list(set(self.article_desc.keys()) - interacted_ids)

            # Sample non-interacted articles if necessary
            sampled_non_interacted = random.sample(
                non_interacted_ids, max(0, self.num_items - len(interacted_ids))
            )
            selected_ids = list(interacted_ids) + sampled_non_interacted
            random.shuffle(selected_ids)  # Shuffle to mix interacted and non-interacted

            # Generate labels (1 for interacted, 0 for non-interacted)
            labels = [1 if article_id in interacted_ids else 0 for article_id in selected_ids]

            # Generate trigrams from user comments
            user_comments = data.loc[data.usr_id == user['usr_id'], 'user_comment'].tolist()
            trigrams = [' '.join(grams) for comment in user_comments for grams in ngrams(comment.split(), 3)]

            # Append sample data
            self.samples.append({
                'id': user['usr_id'],
                'selected_ids': selected_ids,
                'comments': ['. '.join(user.get('comments', []))] * len(selected_ids),
                'labels': labels,
                'usr_interacted_rates': [user.get('interacted_rate')] * len(selected_ids),
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
        article_desc_tokenized = self.tokenizer(
            [desc for descs in article_descs for desc in descs],
            padding="max_length", max_length=150, truncation=True, return_tensors='pt'
        ).input_ids.view(len(batch), -1, 150)  # Reshape to [batch_size, num_items, max_length]

        # Tokenize user comments
        comments_tokenized = self.tokenizer(
            [comment for sublist in comments for comment in sublist],
            padding="max_length", max_length=150, truncation=True, return_tensors='pt'
        ).input_ids.view(len(batch), -1, 150)  # Reshape as above

        # Process trigrams
        usr_trigrams_tokenized = []
        for trigrams in usr_trigrams:
            if trigrams:
                trigrams_tokenized = self.tokenizer(
                    trigrams, padding="max_length", max_length=self.trigram_dim,
                    truncation=True, return_tensors='pt'
                ).input_ids
                usr_trigrams_tokenized.append(trigrams_tokenized.unsqueeze(0).repeat(len(selected_ids), 1, 1))
            else:
                usr_trigrams_tokenized.append(torch.zeros(len(selected_ids), self.trigram_dim, dtype=torch.long))

        return {
            'id': ids,
            'article_ids': selected_ids,
            'usr_comments': comments_tokenized,
            'descriptions': article_desc_tokenized,
            'labels': torch.tensor(labels, dtype=torch.float),
            'usr_interacted_rates': torch.stack(usr_interacted_rates),
            'usr_trigram': torch.stack(usr_trigrams_tokenized)
        }