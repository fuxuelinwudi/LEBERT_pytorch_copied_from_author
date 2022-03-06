# coding:utf-8

import torch
import torch.nn as nn


class WordEmbeddings(nn.Module):
    def __init__(self, pretrained_embeddings):
        super().__init__()
        self.word_embedding = nn.Embedding(pretrained_embeddings.shape[0], pretrained_embeddings.shape[1])
        self.word_embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def forward(self, matched_word_ids):
        word_embedding = self.word_embedding(matched_word_ids)
        return word_embedding
