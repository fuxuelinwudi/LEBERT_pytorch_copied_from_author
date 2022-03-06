# coding:utf-8

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from src.utils.lebert_utils.modeling.layers import BertLayer


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.add_layers = config.add_layers

        total_layers = []
        for i in range(config.num_hidden_layers):
            if i in self.add_layers:
                total_layers.append(BertLayer(config, True))
            else:
                total_layers.append(BertLayer(config, False))

        self.layer = nn.ModuleList(total_layers)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            input_word_embeddings=None,
            input_word_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    input_word_embeddings,
                    input_word_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )

            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    input_word_embeddings,
                    input_word_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_attentions
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions
        )