#-*- coding:utf-8 -*-

"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from .NeuralNetwork.structures.image_list import to_image_list

from ..backbone.backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    中文:
    GeneralizedRCNN 网络由三部分组成: backbone网络, rpn网络, roi_head
    backbone 是主干网络, 一般由残差网络 restnet 加上 小特征识别网络 fpn 组成
    rpn 跟在 backbone 后面, 是用来提取候选框(Region Proposal)的网络, 增加了网络效率与速度
    跟在 rpn 后面有两个 roi_head, 并联的互不影响, 分别用于 bounding box 和 mask, 我们也可以根据需要额外添加其它功能的 roi_head
    返回:
    train 状态下返回 loss
    test 状态下返回 model 的预测结果
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        # 根据配置信息创建 backbone 网络
        self.backbone = build_backbone(cfg)
        # 根据配置信息创建 rpn 网络
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        # 根据配置信息创建 roi_heads
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        
        # 当 training 设置为 True 时, 必须提供 targets, 即 targets 不为 None
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be provided")
            
        # 将图片的数据类型转换成 ImageList
        images = to_image_list(images)
        
        # 利用 backbone 网络获取图片的 features
        features = self.backbone(images.tensors)
        
        # 利用 rpn 网络获取 proposals 和相应的 loss
        proposals, proposal_losses = self.rpn(images, features, targets)
        
        # 如果 roi_heads 不为 None, 则计算其输出的结果
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}
            
        # training 模式下, 输出损失值
        if self.training:
            losses = {}
            # dict.update(dict2): 将dict2添加进dict中
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses
        
        # 如果不在训练模式下, 即测试时, 则输出模型的预测结果
        # 还有两个参数: x, detector_losses 可供输出
        return result
