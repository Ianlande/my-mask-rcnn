#-*- coding:utf-8 -*-

from generalized_rcnn import GeneralizedRCNN

_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN}

# 根据 cfg 实例化一个 class GeneralizedRCNN 类
def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
