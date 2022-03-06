# coding:utf-8


import torch.nn as nn
from transformers import apply_chunking_to_forward
from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput
from src.utils.lebert_utils.modeling.adapter import LexiconAdapter
from src.utils.lebert_utils.modeling.attention import BertAttention


class BertLayer(nn.Module):
    def __init__(self, config, word_attn=False):
        super().__init__()

        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = BertAttention(config)

        self.word_attn = word_attn
        self.adapter = LexiconAdapter(config)

        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            input_word_embeddings=None,
            input_word_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False
    ):

        # 1.character contextual representation
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]  # this is the contextual representation
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # decode need join attention from the outputs
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers " \
               f"by setting `config.add_cross_attention=True`"
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        # add word attention
        if self.word_attn and input_word_mask is not None:
            layer_output = self.adapter(layer_output, input_word_embeddings, input_word_mask)

        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
