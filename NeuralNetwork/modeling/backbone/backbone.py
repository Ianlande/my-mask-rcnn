#-*- coding:utf-8 -*-

'''
    主要定义由各种 resnet 与 fpn 组合而成的各种 backbone 类型
    并通过调用 registry 注册器提供可供选择的多种 backbone 类型
    主要接口为 build_backbone(cfg)
'''

# 导入有序字典
from collections import OrderedDict

from torch import nn

# 注册器, 用于管理 module 的注册, 使得可以像使用字典一样使用 module
from NeuralNetwork.modeling.utils import registry
from NeuralNetwork.modeling.utils.make_layers import conv_with_kaiming_uniform
from . import fpn as fpn_module
from . import resnet


# 创建网络, 根据配置信息会被下面的 build_backbone 函数调用
@registry.BACKBONES.register("R-50-C4")
@registry.BACKBONES.register("R-50-C5")
@registry.BACKBONES.register("R-101-C4")
@registry.BACKBONES.register("R-101-C5")
def build_resnet_backbone(cfg):
    # 实例化 resnet 网络
    body = resnet.ResNet(cfg)
    
    # 创建 resnet 网络
    # nn.Sequential 与 nn.Module 功能差不多, 都是用于创建神经网络
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    return model


# 创建网络, 根据配置信息会被下面的 build_backbone 函数调用
@registry.BACKBONES.register("R-50-FPN")
@registry.BACKBONES.register("R-101-FPN")
@registry.BACKBONES.register("R-152-FPN")
def build_resnet_fpn_backbone(cfg):
    # 先实例化 resnet 网络
    body = resnet.ResNet(cfg)
    
    # 通过 cfg 获取 fpn 所需的 channels 参数
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    
    # 实例化 fpn 网络
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    
    # 创建网络, 将 ResNet 和 FPN 组合在一起形成新的神经网络
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


# 创建网络, 根据配置信息会被下面的 build_backbone 函数调用
@registry.BACKBONES.register("R-50-FPN-RETINANET")
@registry.BACKBONES.register("R-101-FPN-RETINANET")
def build_resnet_fpn_p3p7_backbone(cfg):
    # 先实例化 resnet 网络
    body = resnet.ResNet(cfg)
    
    # 通过 cfg 获取所需参数
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 else out_channels
        
    # 实例化 fpn 网络
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    
    # 创建网络
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


# 最终搭建 backbone 网络
def build_backbone(cfg):
    # if no cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES 时触发, 即 cfg 中没有这个选择
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(cfg.MODEL.BACKBONE.CONV_BODY)
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
