from collections import OrderedDict

from torch import nn

from image_captioning.modeling import registry
from image_captioning.modeling.make_layers import conv_with_kaiming_uniform
from . import resnet
from . import fpn as fpn_module


@registry.ENCODERS.register("R-50-C4")
@registry.ENCODERS.register("R-50-C5")
@registry.ENCODERS.register("R-101-C4")
@registry.ENCODERS.register("R-101-C5")
@registry.ENCODERS.register("R-152-C4")
@registry.ENCODERS.register("R-152-C5")
def build_resnet(cfg):
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    return model


@registry.ENCODERS.register("R-50-FPN")
@registry.ENCODERS.register("R-101-FPN")
def build_resnet_fpn(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


def build_encoder(cfg):
    conv_body = cfg.MODEL.ENCODER.CONV_BODY
    assert conv_body in registry.ENCODERS,\
        "cfg.MODEL.ENCODER.CONV_BODY:{} are not registered in registry".format(
            conv_body
        )
    return registry.ENCODERS[conv_body](cfg)
