import torch
from torch import nn
import torch.nn.functional as F

# Model definition with trigram features
class TrigramTextScoreModel(nn.Module):
    def __init__(self, config):
        super(TrigramTextScoreModel, self).__init__()

        self.trigram_embedding = nn.Embedding(config['DATA']['VOCAB_SIZE'], config['TEXT_BASED']['EMBEDDING_DIM'])
        self.subreddit_embedding = nn.Embedding(config['DATA']['VOCAB_SIZE'], config['TEXT_BASED']['EMBEDDING_DIM'])

        self.fc1 = nn.Linear(config['DATA']['TRIGRAM_DIM'] * config['TEXT_BASED']['EMBEDDING_DIM'], config['DATA']['TRIGRAM_DIM'])
        self.fc2 = nn.Linear(config['DATA']['TRIGRAM_DIM'] + config['TEXT_BASED']['HIDDEN_DIM'], config['TEXT_BASED']['HIDDEN_DIM'])

        self.fc3 = nn.Linear(config['TEXT_BASED']['HIDDEN_DIM'], config['DATA']['NUM_CLASSES'])


    def forward(self, items):
        # trigram and interacted_rate embeddings
        trigram_embeds = self.trigram_embedding(items['trigram_ids']) # Shape: (batch_size, seq_len, embedding_dim)
        interacted_rate_embeds = self.subreddit_embedding(items['interacted_rate'])  # Shape: (batch_size, seq_len, embedding_dim)
        interacted_rate_features = interacted_rate_embeds.mean(dim=1)  # Aggregate features, Shape: (batch_size, embedding_dim)
        
        batch_size, seq_len, _, _ = trigram_embeds.shape
        trigrams_features = trigram_embeds.view([batch_size, seq_len, -1]).mean(dim=1) # Aggregate features, Shape: (batch_size, trigram_dim * embedding_dim )
        # print(trigrams_features.shape)
        # Fully connected layers
        trigrams_features = F.relu(self.fc1(trigrams_features))

        # Concatenate subreddit features with trigram vectors
        combined_features = torch.cat([interacted_rate_features, trigrams_features], dim=-1)  # Shape: (batch_size, trigram_dim + hidden_dim)
        # print(combined_features.shape)
        hidden = F.relu(self.fc2(combined_features))
        output = self.fc3(hidden)  # Shape: (batch_size, num_classes)
        return output