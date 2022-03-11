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

    def WordAttention(self, layer_output, word_outputs, input_word_mask):
        alpha = torch.matmul(layer_output.unsqueeze(2), self.attn_W)
        alpha = torch.matmul(alpha, torch.transpose(word_outputs, 2, 3))
        alpha = alpha.squeeze()
        alpha = alpha + (1 - input_word_mask.float()) * (-10000.0)
        alpha = torch.nn.Softmax(dim=-1)(alpha)
        alpha = alpha.unsqueeze(-1)
        return alpha

    def BiLinear_Attention(self, hidden_states, word_embedding, word_mask):
        word_embedding = self.word_transform(word_embedding)
        word_attention = self.WordAttention(hidden_states, word_embedding, word_mask)
        word_embedding = torch.sum(word_embedding * word_attention, dim=2)
        hidden_states = hidden_states + word_embedding
        return hidden_states

    def Add_Norm(self, hidden_states):
        hidden_states = self.fuse_layer(hidden_states)
        return hidden_states

    def forward(self, layer_output, word_embeddings, word_mask):
        layer_output = self.BiLinear_Attention(layer_output, word_embeddings, word_mask)
        layer_output = self.Add_Norm(layer_output)
        return layer_output

