from typing import Optional

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import PreTrainedModel, AutoModelForMaskedLM, AutoModelForQuestionAnswering, BartConfig
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput, MaskedLMOutput
from .configuration_lava import LavaConfig
from .modeling_lava import LavaModel

class LavaModelSmall(LavaModel):

    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def from_lava_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:

        encoder = AutoModelForMaskedLM.from_pretrained(encoder_pretrained_model_name_or_path)
        decoder_ = AutoModelForQuestionAnswering.from_pretrained(decoder_pretrained_model_name_or_path)
        decoder_state_dict = decoder_.state_dict()
        decoder = AutoModelForQuestionAnswering.from_config(
            BartConfig(encoder_layers = 1, decoder_layers = 1, d_model = 768, decoder_ffn_dim = 3072, encoder_ffn_dim = 3072)
            )


        
        config = LavaConfig.from_encoder_decoder_configs(encoder.config, decoder.config, **kwargs)
        inst = cls(config)
        inst.encoder = encoder
        inst.decoder = decoder

        inst.decoder.load_state_dict({k:decoder_state_dict[k] for k in decoder.state_dict()})
        inst.decoder.model.shared = encoder.roberta.embeddings.word_embeddings
        inst.decoder.model.encoder.embed_tokens = encoder.roberta.embeddings.word_embeddings
        inst.decoder.model.decoder.embed_tokens = encoder.roberta.embeddings.word_embeddings

        inst.decoder.qa_outputs.bias.requires_grad = False

        return inst
