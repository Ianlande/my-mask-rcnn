# mask rcnn (已经弃用)

## 项目功能
> * 基于神经网络的目标识别, 分类, 轮廓分割

## 文件说明
> * NeuralNetwork : 神经网络模块
> * tools : 训练网络, 测试网络
> * datasets : 存放训练集和验证集的地方
> * configs : 配置文件, 运行程序前会覆盖神经网络默认的配置; 以及预训练的权重

## 训练集制作 : `maskrcnn_demo` 文件
### 文件地址


### `maskrcnn_demo` 文件构成
> * data_origin: Images, labels
> * train: Images, labels
> * val: Images, labels
> * script: data_distribution.py, labelme2coco.py

### 使用方法
`data_origin/Images` 中储存原始图片, 使用 labelme 打好标签后, 标签(.json格式)放入 `data_origin/labels`  
运行 `script/data_distribution.py`, 该程序按照一定比例分配数据集和验证集, 分别自动存入 train 和 val  
运行 `script/labelme2coco.py`, 生成 `instances_train2014.json` 和 `instances_val2014.json`, 放入 `datasets/demo/annotation`  
`train/Images` 中的图片放入 `datasets/demo/train2014`  
`val/Images` 中的图片放入 `datasets/demo/val2014`  

## 如何训练
修改三个文件: 
> * configs/*.yaml
> * NeuralNetwork/config/defaults.py
> * NeuralNetwork/config/paths_catalog.py
> * tools/train_net.py

### 修改 configs/*.yaml
修改 `WEIGHT` 为预训练权重的地址
原先程序默认8GPU运行, 现已修改为1GPU情况  
> * 学习率(learning rate)除以8倍: SOLVER.BASE_LR
> * 最大迭代次数乘以8倍: SOLVER.MAX_ITER
> * 学习率调度乘以8倍: SOLVER.STEPS

### 修改 NeuralNetwork/config/defaults.py
> * 原先的程序默认是8GPU运行, 现在已经修改为1GPU运行, 具体修改`_C.SOLVER.IMS_PER_BATCH`和`_C.TEST.IMS_PER_BATCH`, 前者是训练时, 后者是测试时
> * `_C.MODEL.ROI_BOX_HEAD.NUM_CLASSES=识别的种类数+1(背景)`, 其实NUM_CLASSES有三个, 但是另外两个是`keypoint_head`和`retinanet`, 不用的话就不用管

### 修改 NeuralNetwork/config/paths_catalog.py
> * `DatasetCatalog/DATA_DIR` 中的 datasets 文件夹路径修改
> * `DatasetCatalog/DATASETS` 中的图片路径和标注路径的修改  

### 修改 tools/train_net.py
`main()` 函数中的 `parser.add_argument("--config-file"...`  

### 运行
python tools/train_net.py

## 如何测试
