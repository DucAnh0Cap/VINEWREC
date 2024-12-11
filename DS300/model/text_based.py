import torch
from torch import nn
import torch.nn.functional as F

# Model definition with trigram features
class TrigramTextScoreModel(nn.Module):
    def __init__(self, config):
        super(TrigramTextScoreModel, self).__init__()
        self.batch_size = config['DATA']['batch_size']
        self.seq_len = config['DATA']['seq_len']

        self.trigram_embedding = nn.Embedding(config['DATA']['vocab_size'], config['TEXT_BASED']['embedding_dim'])
        self.subreddit_embedding = nn.Embedding(config['DATA']['vocab_size'], config['TEXT_BASED']['embedding_dim'])

        self.fc1 = nn.Linear(config['DATA']['trigram_dim'] * config['TEXT_BASED']['embedding_dim'], config['DATA']['trigram_dim'])
        self.fc2 = nn.Linear(config['DATA']['trigram_dim'] + config['TEXT_BASED']['hidden_dim'], config['TEXT_BASED']['hidden_dim'])

        self.fc3 = nn.Linear(config['TEXT_BASED']['hidden_dim'], config['DATA']['num_classes'])


    def forward(self, subreddit_ids, trigram_ids):
        # Subreddit embeddings
        trigram_embeds = self.trigram_embedding(trigram_ids) # Shape: (batch_size, seq_len, embedding_dim)
        subreddit_embeds = self.subreddit_embedding(subreddit_ids)  # Shape: (batch_size, seq_len, embedding_dim)
        subreddit_features = subreddit_embeds.mean(dim=1)  # Aggregate features, Shape: (batch_size, embedding_dim)
        # print(trigram_embeds.shape)
        trigrams_features = trigram_embeds.view([self.batch_size, self.seq_len, -1]).mean(dim=1) # Aggregate features, Shape: (batch_size, trigram_dim * embedding_dim )
        # print(trigrams_features.shape)
        # Fully connected layers
        trigrams_features = F.relu(self.fc1(trigrams_features))

        # Concatenate subreddit features with trigram vectors
        combined_features = torch.cat([subreddit_features, trigrams_features], dim=-1)  # Shape: (batch_size, trigram_dim + hidden_dim)
        # print(combined_features.shape)
        hidden = F.relu(self.fc2(combined_features))
        output = self.fc3(hidden)  # Shape: (batch_size, num_classes)
        return output