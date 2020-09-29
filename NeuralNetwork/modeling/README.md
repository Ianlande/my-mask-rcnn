# modeling 模块: 定义网络的模型, 由 detector， backbone， rpn， roi_heads， utils 组成

## detector 模块: 调用 modeling 模块的接口
detector 模块负责 detect 与 mask  
主要调用另外三个模块: backbone 模块, rpn 模块, roi_heads 模块  
detector 模块包括 detectors.py, generalized_rcnn.py  
> * `detectors.py`: 根据 cfg 实例化 class GeneralizedRCNN 类
> * `generalized_rcnn.py`: 定义 GeneralizedRCNN 类; 该类调用 backbone , rpn , roi_heads 三个模块

## backbone 模块: 定义神经网络骨架 backbone 
backbone 模块包括 backbone.py, fpn.py, resnet.py  
> * `backbone.py`: 通过调用并组合 fpn, resnet 网络; 进而定义各种类型 backbone 网络; 并实例化 backbone 网络
> * `fpn.py`: 定义 fpn 网络
> * `resnet`: 定义 resnet 网络, 定义了众多 restnet 模块

## rpn 模块: 计算有可能的备选区域
根据生成的锚点 anchors 加上与基准边框 ground truth box 之间的偏差生成候选边框 proposals , 并根据loss进行优化  
> * `anchor_generator.py`: 生成 anchors
> * `inference.py`: 主要是 post processor
> * `loss.py`: 主要是 loss evaluator
> * `rpn.py`: 调用上面三个模块, 生成 rpn 网络
> * `utils.py`: 工具包, 主要是被上面几个程序调用

## roi_heads 模块: 计算具体区域的 feature map , 对比备选区域和 groundtruth
backbone 和 rpn 构建后, 就需要在 feature map 上划分相应的 RoI, 接口: `roi_heads.py`
### 组成
> * `roi_heads.py`: roi heads 接口, 创建 roi heads 模块, 选择性调用下面的 `box_head` `mask_head` 模块
> * box_head : 创建 bounding box, 内部接口为 `box_head.py`
> * mask_head : 创建 mask , 内部接口为 `mask_head.py`
### `box_head`, `mask_head` 模块解释
三个模块的原理和结构是一样的, 所以这里解释一次即可:
> * `roi_*_feature_extractors.py`: 提取各个 RoI 特征
> * `roi_*_predictors.py`: 对各个 RoI 进行预测
> * `inference.py`: 主要是 post processor
> * `loss.py`: 主要是 loss evaluator , 和上面的 post processor 计算分类器和回归器的损失
> * `*_head.py`: 调用上面四个程序, 建立 roi head, 最后选择性被 `roi_heads.py` 调用
### `*_head` : 网络预测的最后一层
网络的最后一层是 `roi_head`  
`roi_heads.py` 使用 `build_roi_heads`, 该接口使用 `CombinedROIHeads` 将各个 `roi_head` 组合起来  
网络的最终输出是各个 `*_roi` 的输出, 它们的输出都是 x, proposals, dict; 其中 proposals 便是我们想要的结果  
上述这些被封装进 `model` 接口中, 运行 `model`, 即返回 model 预测的结果  

## utils: 工具包
解释: 略
