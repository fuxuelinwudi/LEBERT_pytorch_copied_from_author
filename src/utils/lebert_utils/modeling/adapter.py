# coding:utf-8

import torch
import torch.nn as nn


class LexiconAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.word_transform = nn.Sequential(
            nn.Linear(config.word_embed_dim, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )

        self.fuse_layer = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )

        self.attn_W = nn.Parameter(torch.zeros(config.hidden_size, config.hidden_size))
        self.attn_W.data.normal_(mean=0.0, std=config.initializer_range)

    def cul_alpha(self, layer_output, word_outputs, input_word_mask):

        alpha = torch.matmul(layer_output.unsqueeze(2), self.attn_W)
        alpha = torch.matmul(alpha, torch.transpose(word_outputs, 2, 3))
        alpha = alpha.squeeze()
        alpha = alpha + (1 - input_word_mask.float()) * (-10000.0)
        alpha = torch.nn.Softmax(dim=-1)(alpha)
        alpha = alpha.unsqueeze(-1)

        return alpha

    def forward(self, layer_output, input_word_embeddings, input_word_mask):

        word_outputs = self.word_transform(input_word_embeddings)
        alpha = self.cul_alpha(layer_output, word_outputs, input_word_mask)
        weighted_word_embedding = torch.sum(word_outputs * alpha, dim=2)
        layer_output = layer_output + weighted_word_embedding
        layer_output = self.fuse_layer(layer_output)

        return layer_output
