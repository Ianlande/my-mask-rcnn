# structures 模块: 定义检测模式使用的数据结构

## bounding_box.py
主要定义了 class BoxList(object) 类, 该类用于表示一系列的 bounding boxes. 
这些 boxes 会以 N*4 大小的 Tensor 来表示. 为了唯一的确定 boxes 在图片中的准确位置, 该类还保存了图片的维度. 
另外, 也可以添加额外的信息到特定的 bounding box 中, 如标签信息.

## boxlist_ops.py
主要定义了一些与 BoxList 类型数据有关的操作

## image_list.py
图像 image 与张量 tensor 之间的一些转换与操作

## segmentation_mask.py
用于轮廓分割 mask segmentation
