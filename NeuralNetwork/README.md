# NeuralNetwork 模块: 神经网络, 项目中最重要的模块
以下按照调用顺序进行说明

## engine 模块: 训练与测试接口, 因此在通常情况下被外部函数首先调用
`trainer.py` : 训练网络, train
`inference.py` : 测试网络, test

## modeling 模块: 定义神经网络模型
神经网络的程序被封装在 modeling 模块中  
最外层接口在 `modeling/detector/detectors.py/build_detection_model`, 该接口将实例化一个神经网络  
外层相对最底层接口在 `modeling/roi_heads/*_head/*_head.py`, 这是神经网络的最后一层所在处  

## config 模块: 配置文件及默认配置
网络在运行前, configs 的配置会覆盖更新默认配置 `NeuralNetwork/config`  

## data 模块: 与数据有关的功能模块
接口: `data/build.py/make_data_loader`, 功能是导入数据, 一般被神经网络载入数据时调用

## solver 模块: 用于优化学习率的功能函数

## layers 模块: 与层相关的功能函数

## structures 模块: 底层数据结构的功能函数

## utils: 工具包
