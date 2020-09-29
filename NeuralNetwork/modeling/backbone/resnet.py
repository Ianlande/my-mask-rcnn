#-*- coding:utf-8 -*-

"""
Variant of the resnet module that takes cfg as an argument.
Example usage. Strings may be specified in the config file.
    model = ResNet(
        "StemWithFixedBatchNorm",
        "BottleneckWithFixedBatchNorm",
        "ResNet50StagesTo4",
    )
OR:
    model = ResNet(
        "StemWithGN",
        "BottleneckWithGN",
        "ResNet50StagesTo4",
    )
Custom implementations may be written in user code and hooked in via the
`register_*` functions.
"""

'''
    主要定义各种 resnet 类
'''

from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

from NeuralNetwork.layers.misc import Conv2d
from NeuralNetwork.layers.misc import DFConv2d
from NeuralNetwork.layers.batch_norm import FrozenBatchNorm2d
from NeuralNetwork.modeling.utils.make_layers import group_norm
from NeuralNetwork.utils.registry import Registry


# ResNet stage specification
# namedtuple 命名元组, 可以用名字访问元素
StageSpec = namedtuple(
    "StageSpec",
    [
        "index",  # stage 的下标, 如 1, 2, ..., 5
        "block_count",  # stage 当中的 block 的数量
        "return_features",  # 布尔值, 若为 True, 则返回当前 stage 的最后一层的 feature map
    ],
)


# -----------------------------------------------------------------------------
# Standard ResNet models
# 标准 ResNet 模块
# 定义各个 ResNet 模型的 stages 的卷积层数量
# -----------------------------------------------------------------------------
# ResNet-50 (including all stages)
# ResNet-50 full stages 的2~5阶段的卷积层数分别为:3,4,6,3(从0开始数)
# 元组内部的元素类型为 StageSpec
ResNet50StagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 6, False), (4, 3, True))
)
# ResNet-50 up to stage 4 (excludes stage 5)
# ResNet-50-C4, 只使用到第四阶段输出的特征
ResNet50StagesTo4 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 6, True))
)
# ResNet-101 (including all stages)
# ResNet-101 full stages 的2~5阶段的卷积层数分别为:3,4,23,3(从0开始数)
ResNet101StagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 23, False), (4, 3, True))
)
# ResNet-101 up to stage 4 (excludes stage 5)
# ResNet-101-C4, 只使用到第四阶段输出的特征
ResNet101StagesTo4 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 23, True))
)
# ResNet-50-FPN (including all stages)
# ResNet-50-FPN full stages, 由于 FPN 需要用到每一个阶段输出的特征图谱, 故 return_features 参数均为 True
ResNet50FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 6, True), (4, 3, True))
)
# ResNet-101-FPN (including all stages)
# ResNet-101-FPN full stages, 2~5阶段的卷积层数分别为:3,4,23,3
ResNet101FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 23, True), (4, 3, True))
)
# ResNet-152-FPN (including all stages)
# ResNet-152-FPN full stages, 2~5阶段的卷积层数分别为:3,8,36,3
ResNet152FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 8, True), (3, 36, True), (4, 3, True))
)


class ResNet(nn.Module):
    def __init__(self, cfg):
        super(ResNet, self).__init__()

        # If we want to use the cfg in forward(), then we should make a copy
        # of it and store it for later use:
        # self.cfg = cfg.clone()
        
        # Translate string names to implementations
        # 将配置文件中的字符串转化成具体的实现, 下面三个分别使用了对应的注册模块, 定义在文件的最后
        # 这里是 stem 的实现, 也就是 resnet 的第一阶段 conv1
        stem_module = _STEM_MODULES[cfg.MODEL.RESNETS.STEM_FUNC]
        
        # resnet conv2_x~conv5_x 的实现
        # eg: cfg.MODEL.CONV_BODY="R-50-FPN"
        stage_specs = _STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY]
        
        # 剩余的转换函数
        transformation_module = _TRANSFORMATION_MODULES[cfg.MODEL.RESNETS.TRANS_FUNC]
        # 获取上面各个组成部分的实现以后, 就可以利用这些实现来构建模型
        
        # Construct the stem module
        # 构建 stem module, 即 conv1
        self.stem = stem_module(cfg)
        
        # Constuct the specified ResNet stages
        # 当 num_groups=1 时为 ResNet, >1 时 为 ResNeXt
        num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
        
        width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        
        # in_channels 指的是向后面的第二阶段输入时特征图谱的通道数, 也就是 stem 的输出通道数, 默认为 64
        in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
        
        # 第二阶段输入的特别图谱的通道数
        stage2_bottleneck_channels = num_groups * width_per_group
        
        # 第二阶段的输出, resnet 系列标准模型可从 resnet 第二阶段的输出通道数判断后续的通道数
        # 默认为256, 则后续分别为512, 1024, 2048, 若为64, 则后续分别为128, 256, 512
        stage2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        
        # 创建一个空的 stages 列表和对应的特征字典
        self.stages = []
        self.return_features = {}
        
        # resnet conv2_x~conv5_x
        for stage_spec in stage_specs:
            name = "layer" + str(stage_spec.index)
            
            # 计算每个stage的输出通道数, 每经过一个stage, 通道数都会加倍
            stage2_relative_factor = 2 ** (stage_spec.index - 1)
            
            # 计算输入的通道数
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
            
            # 计算输出的通道数
            out_channels = stage2_out_channels * stage2_relative_factor
            
            stage_with_dcn = cfg.MODEL.RESNETS.STAGE_WITH_DCN[stage_spec.index -1]
            
            # 当获取到所有需要的参数以后, 调用本文件的 `_make_stage` 函数, 该函数在下面被定义了
            # 该函数可以根据传入的参数创建对应 stage 的模块(注意是module而不是model)
            module = _make_stage(
                transformation_module,
                in_channels, # 输入的通道数
                bottleneck_channels, # 压缩后的通道数
                out_channels, # 输出的通道数
                stage_spec.block_count, #当前stage的卷积层数量
                num_groups, # ResNet时为1, ResNeXt时>1
                cfg.MODEL.RESNETS.STRIDE_IN_1X1,
                # 当处于 stage3~5时, 需要在开始的时候使用 stride=2 来downsize
                first_stride=int(stage_spec.index > 1) + 1,
                dcn_config={
                    "stage_with_dcn": stage_with_dcn,
                    "with_modulated_dcn": cfg.MODEL.RESNETS.WITH_MODULATED_DCN,
                    "deformable_groups": cfg.MODEL.RESNETS.DEFORMABLE_GROUPS,
                }
            )
            
            # 下一个 stage 的输入通道数即为当前 stage 的输出通道数
            in_channels = out_channels
            
            # 将当前stage模块添加到模型中
            self.add_module(name, module)
            
            # 将stage的名称添加到列表中
            self.stages.append(name)
            
            # 将stage的布尔值添加到字典中
            self.return_features[name] = stage_spec.return_features
            
        # Optionally freeze (requires_grad=False) parts of the backbone
        # 根据配置文件的参数选择性的冻结某些层(requires_grad=False)
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

    # 根据给定的参数冻结某层的参数更新
    def _freeze_backbone(self, freeze_at):
        if freeze_at < 0:
            return
        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem  # stage 0 is the stem, resnet 的第一阶段, 即为 stem
            else:
                m = getattr(self, "layer" + str(stage_index))
            # 将 m 中的所有参数置为不更新状态
            for p in m.parameters():
                p.requires_grad = False

    # ResNet 的前行传播
    def forward(self, x):
        outputs = []
        # stem(stage 1)
        x = self.stem(x)
        # 依次经过 stage2~5
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
            if self.return_features[stage_name]:
                 # 将stage2~5的结果(也就是特征)以list形式保存
                outputs.append(x)
        return outputs


# Resnet head 是指使用 Bottleneck 模块堆叠成的用于构成 Resnet 的功能结构
class ResNetHead(nn.Module):
    def __init__(
        self,
        block_module,
        stages,
        num_groups=1,
        width_per_group=64,
        stride_in_1x1=True,
        stride_init=None,
        res2_out_channels=256,
        dilation=1,
        dcn_config={}
    ):
        super(ResNetHead, self).__init__()

        stage2_relative_factor = 2 ** (stages[0].index - 1)
        stage2_bottleneck_channels = num_groups * width_per_group
        out_channels = res2_out_channels * stage2_relative_factor
        in_channels = out_channels // 2
        bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor

        block_module = _TRANSFORMATION_MODULES[block_module]

        self.stages = []
        stride = stride_init
        for stage in stages:
            name = "layer" + str(stage.index)
            if not stride:
                stride = int(stage.index > 1) + 1
            module = _make_stage(
                block_module,
                in_channels,
                bottleneck_channels,
                out_channels,
                stage.block_count,
                num_groups,
                stride_in_1x1,
                first_stride=stride,
                dilation=dilation,
                dcn_config=dcn_config
            )
            stride = None
            self.add_module(name, module)
            self.stages.append(name)
        self.out_channels = out_channels

    def forward(self, x):
        for stage in self.stages:
            x = getattr(self, stage)(x)
        return x


# 被 ResNet 调用
def _make_stage(
    transformation_module,
    in_channels,
    bottleneck_channels,
    out_channels,
    block_count,
    num_groups,
    stride_in_1x1,
    first_stride,
    dilation=1,
    dcn_config={}
):
    blocks = []
    stride = first_stride
    for _ in range(block_count):
        blocks.append(
            transformation_module(
                in_channels,
                bottleneck_channels,
                out_channels,
                num_groups,
                stride_in_1x1,
                stride,
                dilation=dilation,
                dcn_config=dcn_config
            )
        )
        stride = 1
        in_channels = out_channels
    return nn.Sequential(*blocks)


# Bottleneck 模块, 被 BottleneckWithFixedBatchNorm 与 BottleneckWithGN 调用
class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups,
        stride_in_1x1,
        stride,
        dilation,
        norm_func,
        dcn_config
    ):
        super(Bottleneck, self).__init__()

        self.downsample = None
        if in_channels != out_channels:
            down_stride = stride if dilation == 1 else 1
            self.downsample = nn.Sequential(
                Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=down_stride, bias=False
                ),
                norm_func(out_channels),
            )
            for modules in [self.downsample,]:
                for l in modules.modules():
                    if isinstance(l, Conv2d):
                        nn.init.kaiming_uniform_(l.weight, a=1)

        if dilation > 1:
            stride = 1 # reset to be 1

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
        )
        self.bn1 = norm_func(bottleneck_channels)
        # TODO: specify init for the above
        with_dcn = dcn_config.get("stage_with_dcn", False)
        if with_dcn:
            deformable_groups = dcn_config.get("deformable_groups", 1)
            with_modulated_dcn = dcn_config.get("with_modulated_dcn", False)
            self.conv2 = DFConv2d(
                bottleneck_channels,
                bottleneck_channels,
                with_modulated_dcn=with_modulated_dcn,
                kernel_size=3,
                stride=stride_3x3,
                groups=num_groups,
                dilation=dilation,
                deformable_groups=deformable_groups,
                bias=False
            )
        else:
            self.conv2 = Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=3,
                stride=stride_3x3,
                padding=dilation,
                bias=False,
                groups=num_groups,
                dilation=dilation
            )
            nn.init.kaiming_uniform_(self.conv2.weight, a=1)

        self.bn2 = norm_func(bottleneck_channels)

        self.conv3 = Conv2d(
            bottleneck_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn3 = norm_func(out_channels)

        for l in [self.conv1, self.conv3,]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu_(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu_(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu_(out)

        return out


# stem 模块, 被 StemWithFixedBatchNorm 与 StemWithGN 调用
class BaseStem(nn.Module):
    def __init__(self, cfg, norm_func):
        super(BaseStem, self).__init__()

        out_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS

        self.conv1 = Conv2d(
            3, out_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_func(out_channels)

        for l in [self.conv1,]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


'''
Bottleneck, BaseStem 结构的衍生与封装: 
Bottleneck 与 Stem 模块分别衍生出 Batch Normalization 和 Group Normalizetion 类;
具体为: BottleneckWithFixedBatchNorm, StemWithFixedBatchNorm, BottleneckWithGN, StemWithGN
'''
# 该类负责构建 ResNet 的 resnet2~5, resnet2~5 主要由 Bottleneck 组成
class BottleneckWithFixedBatchNorm(Bottleneck):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups=1,
        stride_in_1x1=True,
        stride=1,
        dilation=1,
        dcn_config={}
    ):
        super(BottleneckWithFixedBatchNorm, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=FrozenBatchNorm2d,
            dcn_config=dcn_config
        )


# 该类负责构建 ResNet 的 stem, resnet1 主要由 stem 组成
class StemWithFixedBatchNorm(BaseStem):
    def __init__(self, cfg):
        super(StemWithFixedBatchNorm, self).__init__(
            cfg, norm_func=FrozenBatchNorm2d
        )


class BottleneckWithGN(Bottleneck):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups=1,
        stride_in_1x1=True,
        stride=1,
        dilation=1,
        dcn_config={}
    ):
        super(BottleneckWithGN, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=group_norm,
            dcn_config=dcn_config
        )


class StemWithGN(BaseStem):
    def __init__(self, cfg):
        super(StemWithGN, self).__init__(cfg, norm_func=group_norm)


'''
以上四种衍生类被封装起来, 以便调用, 如下, 封装为 _TRANSFORMATION_MODULES 与 _STEM_MODULES
'''
_TRANSFORMATION_MODULES = Registry({
    "BottleneckWithFixedBatchNorm": BottleneckWithFixedBatchNorm,
    "BottleneckWithGN": BottleneckWithGN,
})

_STEM_MODULES = Registry({
    "StemWithFixedBatchNorm": StemWithFixedBatchNorm,
    "StemWithGN": StemWithGN,
})

# 注册的各个模块, 这些模块会通过配置文件 cfg 中的字符串信息来决定调用哪一个类或者参数
_STAGE_SPECS = Registry({
    "R-50-C4": ResNet50StagesTo4,
    "R-50-C5": ResNet50StagesTo5,
    "R-101-C4": ResNet101StagesTo4,
    "R-101-C5": ResNet101StagesTo5,
    "R-50-FPN": ResNet50FPNStagesTo5,
    "R-50-FPN-RETINANET": ResNet50FPNStagesTo5,
    "R-101-FPN": ResNet101FPNStagesTo5,
    "R-101-FPN-RETINANET": ResNet101FPNStagesTo5,
    "R-152-FPN": ResNet152FPNStagesTo5,
})
