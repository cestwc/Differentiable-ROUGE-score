from typing import Optional

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import PreTrainedModel, AutoModelForMaskedLM, AutoModelForQuestionAnswering
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput, MaskedLMOutput
from .configuration_lava import LavaConfig

class LavaModel(PreTrainedModel):

    config_class = LavaConfig
    base_model_prefix = "lava"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)

        self.decoder = AutoModelForQuestionAnswering.from_config(config.decoder)
        self.encoder = AutoModelForMaskedLM.from_config(config.encoder)

        self.decoder.qa_outputs.bias.requires_grad = False
        
        self.decoder.config = self.config.decoder
        self.encoder.config = self.config.encoder
        
    def _set_gradient_checkpointing(self, module, value=False):
        self.encoder._set_gradient_checkpointing(module, value=value)
        self.decoder._set_gradient_checkpointing(module, value=value)

    def get_input_embeddings(self):
        return self.decoder.get_input_embeddings()

    def get_output_embeddings(self):
        return self.encoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.encoder.set_output_embeddings(new_embeddings)

    @classmethod
    def from_lava_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:

        encoder = AutoModelForMaskedLM.from_pretrained(encoder_pretrained_model_name_or_path)
        decoder = AutoModelForQuestionAnswering.from_pretrained(decoder_pretrained_model_name_or_path)
        
        config = LavaConfig.from_encoder_decoder_configs(encoder.config, decoder.config, **kwargs)
        inst = cls(config)
        inst.encoder = encoder
        inst.decoder = decoder

        inst.decoder.qa_outputs.bias.requires_grad = False

        return inst

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> MaskedLMOutput:

        decoder_attention_mask = torch.ones_like(input_ids) if labels is None else 1 - (labels == -100).float()
        decoder_input_ids = torch.where(decoder_attention_mask.bool(), 50264, 1).long()

        input_ids_ = input_ids.clone()
        if self.training:
            masked_indices = torch.bernoulli(torch.ones_like(input_ids) * 0.15).bool()
            input_ids_[~masked_indices] = -100

            indices_replaced = torch.bernoulli(torch.ones_like(input_ids) * 0.8).bool() & masked_indices
            input_ids[indices_replaced] = 50264

            indices_random = torch.bernoulli(torch.ones_like(input_ids) * 0.5).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint_like(input_ids, high=50264)
            input_ids[indices_random] = random_words[indices_random]

        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids = decoder_input_ids,
            decoder_attention_mask = decoder_attention_mask,
            output_hidden_states=True
        )

        if labels is None:
            decoder_attention_mask = (decoder_outputs.end_logits < -0).float()
            first_zeros = (decoder_attention_mask == 0).float().argmax(dim=1)
            zero_mask = torch.arange(decoder_attention_mask.shape[1]).unsqueeze(0).to(first_zeros.device) > first_zeros.unsqueeze(1)
            decoder_attention_mask.masked_fill_(zero_mask, 0)
        else:
            labels_cat = torch.cat([input_ids_, labels], dim = 1)

        end_logits_cat = torch.cat([torch.zeros_like(attention_mask), decoder_outputs.end_logits], dim = 1)
        attention_mask_cat = torch.cat([attention_mask, decoder_attention_mask], dim = 1)
        inputs_embeds_cat = torch.cat([decoder_outputs.encoder_last_hidden_state, decoder_outputs.decoder_hidden_states[-1]], dim = 1)

        attention_sort = torch.arange(attention_mask_cat.size(1)).unsqueeze(0).repeat(attention_mask_cat.size(0), 1)
        attention_sort[attention_mask_cat == 0] = attention_mask_cat.size(1) + 1
        sorted_order = torch.argsort(attention_sort, dim=1)
        sorted_order = sorted_order[:, :attention_mask_cat.sum(dim = 1).max().long()].to(attention_mask_cat.device)
        
        attention_mask_cat = torch.gather(attention_mask_cat, 1, sorted_order)
        end_logits_cat = torch.gather(end_logits_cat, 1, sorted_order)
        inputs_embeds_cat = torch.gather(inputs_embeds_cat, 1, sorted_order.unsqueeze(2).expand(-1, -1, inputs_embeds_cat.size(2)))
        if labels is not None:
            labels_cat = torch.gather(labels_cat, 1, sorted_order)

        
        encoder_outputs = self.encoder(
            attention_mask=attention_mask_cat,
            inputs_embeds=inputs_embeds_cat,
            output_hidden_states=True,
            output_attentions=True
        )
        
        encoder_outputs.logits[:,:, self.encoder.config.eos_token_id] += end_logits_cat

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            return MaskedLMOutput(
                loss = loss_fct(encoder_outputs.logits.reshape(-1, self.encoder.config.vocab_size), labels_cat.view(-1)),
                **encoder_outputs
            )
        else:
            return encoder_outputs