from image_captioning.modeling import registry

from .base_decoder import BaseDecoder
from .att_decoder import AttDecoder
from .decoder_core import build_decoder_core


@registry.DECODER_MODELS.register("Baseline")
class FCModel(BaseDecoder):
    def __init__(self, cfg):
        super(FCModel, self).__init__(cfg)
        self.core = build_decoder_core(cfg, len(self.vocab), "Baseline")


@registry.DECODER_MODELS.register("TopDown")
class TopDownModel(AttDecoder):
    def __init__(self, cfg):
        super(TopDownModel, self).__init__(cfg)
        self.core = build_decoder_core(cfg, len(self.vocab), "TopDownCore")


@registry.DECODER_MODELS.register("TopDownNoConv")
class TopDownModelNoConv(AttDecoder):
    def __init__(self, cfg):
        super(TopDownModelNoConv, self).__init__(cfg)
        self.core = build_decoder_core(cfg, len(self.vocab), "TopDownCoreNoConv")


@registry.DECODER_MODELS.register("ConvHidden")
class ConvHiddenModel(AttDecoder):
    def __init__(self, cfg):
        super(ConvHiddenModel, self).__init__(cfg)
        self.core = build_decoder_core(cfg, len(self.vocab), "DualCore")


@registry.DECODER_MODELS.register("DualNoConv")
class DualModelNoConv(AttDecoder):
    def __init__(self, cfg):
        super(DualModelNoConv, self).__init__(cfg)
        self.core = build_decoder_core(cfg, len(self.vocab), "DualCoreNoConv")


@registry.DECODER_MODELS.register("TopDownChannel")
class ChannelModel(AttDecoder):
    def __init__(self, cfg):
        super(ChannelModel, self).__init__(cfg)
        self.core = build_decoder_core(cfg, len(self.vocab), "ChannelCore")


@registry.DECODER_MODELS.register("TopDownChannelNoConv")
class ChannelModelNoConv(AttDecoder):
    def __init__(self, cfg):
        super(ChannelModelNoConv, self).__init__(cfg)
        self.core = build_decoder_core(cfg, len(self.vocab), "ChannelCoreNoConv")

@registry.DECODER_MODELS.register("SelfAtt")
class ChannelModelNoConv(AttDecoder):
    def __init__(self, cfg):
        super(ChannelModelNoConv, self).__init__(cfg)
        self.core = build_decoder_core(cfg, len(self.vocab), "SelfAttCore")
