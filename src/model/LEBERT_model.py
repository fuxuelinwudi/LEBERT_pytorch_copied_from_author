# coding:utf-8


import os
import random
import torch
import torch.nn as nn
import numpy as np
from src.utils.lebert_utils.modeling.crf import CRF
from src.utils.lebert_utils.modeling.embedding import WordEmbeddings
from src.utils.lebert_utils.modeling.modeling import BertPreTrainedModel, LEBertModel


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


seed_everything(2022)


class LEBertCRFForNer(BertPreTrainedModel):
    def __init__(self, config, pretrained_embeddings, num_labels):
        super().__init__(config)

        self.num_labels = num_labels
        self.word_embedding = WordEmbeddings(pretrained_embeddings)

        self.bert = LEBertModel(config)
        self.dropout = nn.Dropout(config.HP_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels + 2)
        self.crf = CRF(self.num_labels)

        self.init_weights()

    def forward(self,
                input_ids=None,
                inputs_embeds=None,
                token_type_ids=None,
                attention_mask=None,
                labels=None,
                boundary_ids=None,
                matched_word_ids=None,
                matched_word_mask=None
                ):

        matched_word_embeddings = self.word_embedding(matched_word_ids)

        sequence_output, pooled_output = self.bert(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            matched_word_embeddings=matched_word_embeddings,
            matched_word_mask=matched_word_mask,
            boundary_ids=boundary_ids,
            return_dict=False
        )

        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)

        if labels is not None:
            loss, predict = self.crf.neg_log_likelihood_loss(logits, attention_mask, labels), \
                            self.crf.decode(logits, attention_mask)

            return loss, predict

        else:
            predict = self.crf.decode(logits, attention_mask)

            return predict
