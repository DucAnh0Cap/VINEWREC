import torch
from torch import nn
import torch.nn.functional as F
from text_based import text_based_clf


class NeuCF(nn.Module):
    def __init__(self, config):
        super(NeuCF, self).__init__()

        self.latent_dim_mlp = config['NCF']['latent_dim_mlp']
        self.latent_dim_gmf = config['NCF']['latent_dim_gmf']
        self.layers = config['NCF']['layers']
        self.weight_init_gaussian = config['NCF']['weight_init_gaussian']
        self.text_based_score = config['NCF']['text_based_score']

        # Use text-based score
        if self.text_based_score:
            self.text_based_clf = TrigramTextScoreModel(config)
            for param in self.text_based_clf.parameters():
                param.requires_grad = False
            self.tb_fc_1 = nn.Linear(config['DATA']['num_classes'], self.latent_dim_mlp)
            self.tb_fc_2 = nn.Linear(self.layers[0], self.layers[1])


        # Embed item and user features
        self.embed_user_mlp = nn.Embedding(num_embeddings=config['DATA']['vocab_size'], embedding_dim=self.latent_dim_mlp)
        self.embed_item_mlp = nn.Embedding(num_embeddings=config['DATA']['vocab_size'], embedding_dim=self.latent_dim_mlp)
        self.embed_user_gmf = nn.Embedding(num_embeddings=config['DATA']['vocab_size'], embedding_dim=self.latent_dim_gmf)
        self.embed_item_gmf = nn.Embedding(num_embeddings=config['DATA']['vocab_size'], embedding_dim=self.latent_dim_gmf)

        # Create mlp layers
        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            self.fc_layers.append(
                nn.Linear(in_features=in_size, out_features=out_size)
            )

        # Output layer
        self.affine_output = nn.Linear(in_features=self.layers[-1] + self.latent_dim_mlp, out_features=1)
        self.logic = nn.Sigmoid()

        # Init weight
        if config['NCF']['weight_init_gaussian']:
            self.init_weight()


    def init_weight(self):
        for sm in self.modules():
            if isinstance(sm, (nn.Embedding, nn.Linear)):
                torch.nn.init.normal_(sm.weight.data, 0.0, 0.01)


    def forward(self, user, item, trigram_ids=None, news_ids=None):
        user_embedding_mlp = self.embed_user_mlp(user).mean(dim=1) # Shape: (batch_size, seq_len, embedding_dim)
        item_embedding_mlp = self.embed_item_mlp(item).mean(dim=1)
        user_embedding_gmf = self.embed_user_gmf(user).mean(dim=1)
        item_embedding_gmf = self.embed_item_gmf(item).mean(dim=1)

        # Add text_based score
        if self.text_based_score:

            embed_scores = self.text_based_clf(news_ids, trigram_ids)
            text_based_feature = self.tb_fc_1(embed_scores)

            user_embedding_gmf = torch.cat([user_embedding_gmf, text_based_feature], dim=-1)
            user_embedding_mlp = torch.cat([user_embedding_mlp, text_based_feature], dim=-1)

            user_embedding_gmf = self.tb_fc_2(user_embedding_gmf)
            user_embedding_mlp = self.tb_fc_2(user_embedding_mlp)

        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        gmf_vector = torch.mul(user_embedding_gmf, item_embedding_gmf)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = nn.ReLU()(mlp_vector)

        vector = torch.cat([mlp_vector, gmf_vector], dim=-1)
        print(vector.shape)
        logits = self.affine_output(vector.flatten(1))
        ratings = self.logic(logits)

        return ratings