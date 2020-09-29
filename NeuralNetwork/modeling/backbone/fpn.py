#-*- coding:utf-8 -*-

'''
    主要定义 FPN 类
'''

import torch
import torch.nn.functional as F
from torch import nn


class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    翻译: 
    在一系列的 feature map (实际上就是stage2~5的最后一层输出)后添加 FPN
    这些 feature maps 的 depth 假定是不断递增的, 并且是连续的
    """

    def __init__(self, in_channels_list, out_channels, conv_block, top_blocks=None):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that will be fed
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
                如果提供有类似结构，输出层特征将会被这些结构进行额外的处理
        """
        super(FPN, self).__init__()
        
        # inner_blocks 指将 Resnet 或其他 backbone 上特征图通过一个 1x1 卷积核卷积得到 fpn 网络上对应的层
        self.inner_blocks = []
        # layer_blocks 指将从 backbone 上特征图获得的融合特征层或直接获得的特征图上进行卷积操作以得到输出的特征图
        self.layer_blocks = []
        
        # 假设使用 ResNet-50-FPN 和配置, 则 in_channels_list 的值为: [256, 512, 1024, 2048]
        for idx, in_channels in enumerate(in_channels_list, 1):
            # 用下表起名: fpn_inner1, fpn_inner2, fpn_inner3, fpn_inner4
            inner_block = "fpn_inner{}".format(idx)
            # 用下表起名: fpn_layer1, fpn_layer2, fpn_layer3, fpn_layer4
            layer_block = "fpn_layer{}".format(idx)

            if in_channels == 0:
                continue
            
            # 构造两种卷积结构
            inner_block_module = conv_block(in_channels, out_channels, 1)
            layer_block_module = conv_block(out_channels, out_channels, 3, 1)
            
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        self.top_blocks = top_blocks

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
         # 将 backbone 上最后的输出层通过 1x1 卷积核卷积得到 fpn 网络上对应的最后一层
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        results = []
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))
        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            if not inner_block:
                continue
            inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")
            inner_lateral = getattr(self, inner_block)(feature)
            # TODO use size instead of scale to make it robust to different sizes
            # inner_top_down = F.upsample(last_inner, size=inner_lateral.shape[-2:],
            # mode='bilinear', align_corners=False)
            last_inner = inner_lateral + inner_top_down
            results.insert(0, getattr(self, layer_block)(last_inner))

        if isinstance(self.top_blocks, LastLevelP6P7):
            last_results = self.top_blocks(x[-1], results[-1])
            results.extend(last_results)
        elif isinstance(self.top_blocks, LastLevelMaxPool):
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)

        return tuple(results)


# 最后一级的 max pool 层, 被 FPN 调用
class LastLevelMaxPool(nn.Module):
    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]


# 在得到的 fpn 网络上再添加两层 P6P7 , 一般用于一阶段目标检测模型, 被 FPN 调用
class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """
    def __init__(self, in_channels, out_channels):
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]
