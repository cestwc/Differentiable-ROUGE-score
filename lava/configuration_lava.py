import copy

from transformers import PretrainedConfig, EncoderDecoderConfig


class LavaConfig(EncoderDecoderConfig):

    model_type = "lava"
    is_composition = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_encoder_decoder = False