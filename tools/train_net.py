#-*- coding:utf-8 -*-

import sys
import argparse
import os
import torch


# 设置一下当前地址, 否则无法查找到 NeuralNetwork 模块
this_dir = os.path.dirname(os.path.abspath(__file__)) # 获得此程序地址, 即 tools 地址
this_dir = os.path.dirname(this_dir) # 获取项目地址
sys.path.append(this_dir)

from NeuralNetwork.utils.env import setup_environment

from NeuralNetwork.config import cfg # config 导入默认配置信息, 接口在 config/__init__.py
from NeuralNetwork.data.build import make_data_loader # 数据集载入, 接口在 data/build.py
from NeuralNetwork.solver import make_lr_scheduler # 学习率更新策略, 接口在 solver/__init__.py, 即 solver/build.py
from NeuralNetwork.solver import make_optimizer # 设置优化器, 封装PyTorch的SGD类, 接口在 solver/__init__.py, 即 solver/build.py
from NeuralNetwork.engine.inference import inference # 推演代码
from NeuralNetwork.engine.trainer import do_train # 模型训练
from NeuralNetwork.modeling.detector.detectors import build_detection_model # 创建目标检测模型

from NeuralNetwork.utils.checkpoint import DetectronCheckpointer
from NeuralNetwork.utils.collect_env import collect_env_info
from NeuralNetwork.utils.comm import synchronize, get_rank # 分布式训练, gpu 个数为1则 get_rank()=0
from NeuralNetwork.utils.imports import import_file
from NeuralNetwork.utils.logger import setup_logger # 封装了 logging 模块
from NeuralNetwork.utils.miscellaneous import mkdir, save_config # 封装 os.mkdirs 函数等

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')


# 训练
def train(cfg, local_rank, distributed):

    # 根据 cfg 创建模型, 接口 NeuralNetwork/modeling/detector/detectors.py
    model = build_detection_model(cfg)
    
    # 使用cuda或者cpu
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)# 将模型存入GPU或者cpu
    
    # 优化, 封装了 torch.optiom.SGD() 函数
    optimizer = make_optimizer(cfg, model)
    # 根据配置信息设置 optimizer 的学习率更新策略
    scheduler = make_lr_scheduler(cfg, optimizer)

    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)
    
    # 多GPU时使用分布式训练
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )
        
    # 字典参数 arguments, 设置迭代次数=0
    arguments = {}
    arguments["iteration"] = 0
    
    # 输出的文件夹路径, 如果不设置则为当前目录
    output_dir = cfg.OUTPUT_DIR
    
    # 如果只有一个GPU则save_to_disk=true
    save_to_disk = get_rank() == 0
    
    # DetectronCheckpointer 对象, do_train() 会用
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT) # 加载指定权重文件
    arguments.update(extra_checkpoint_data) # 字典的update方法, 对字典的键值进行更新

    # 数据加载
    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )
    
    # 每迭代数次测试一下
    test_period = cfg.SOLVER.TEST_PERIOD
    if test_period > 0:
        data_loader_val = make_data_loader(cfg, is_train=False, is_distributed=distributed, is_for_period=True)
    else:
        data_loader_val = None
        
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    
    # 训练, 在 NeuralNetwork/engine/trainer.py 中
    do_train(
        cfg,
        model,
        data_loader,
        data_loader_val,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        test_period,
        arguments,
    )

    return model


# 测试
def run_test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST) # 根据标签文件数确定输出文件夹数
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder) # 创建输出文件夹
            output_folders[idx] = output_folder # 将文件夹的路径名放入列表
            
    # 根据配置文件信息创建数据集
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    
    # 以上都是与创建输出文件夹有关的, 下面才是主要过程
    # 遍历每个标签文件, 执行 inference 过程
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            bbox_aug=cfg.TEST.BBOX_AUG.ENABLED,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize() # 多GPU推演时的同步函数


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument("--config_file",default=r"D:\MyNetWork\Single_channel_weak_feature_object_detection_and_ pose_estimation\configs\e2e_mask_rcnn_R_50_FPN_1gpu_1x_demo.yaml",metavar="FILE",help="path to config file",type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--skip-test",dest="skip_test",help="Do not test the final model",action="store_true")
    parser.add_argument("opts",help="Modify config options using the command-line",default=None,nargs=argparse.REMAINDER)
    args = parser.parse_args()
    
    # 分布式计算, 如果 GPU 数量 =1, num_gpus=1, 不进行多 GPU 分布式训练
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
        
    # 用实际的 config_file 覆盖默认的 config
    # NeuralNetwork/config/default 是默认的 config, 训练前被实际的 config 覆盖, 使用修改后的 config 训练
    # 实际的 config 在 configs 下
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze() # 冻结所有的配置项, 防止修改
    
    # 封装 os.mkdir, 当 output_dir 存在时, 会略过, 不会返回错误
    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)
        
    # 保存在 output_dir 下的 log.txt 文件
    logger = setup_logger("NeuralNetwork", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)
    
    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())
    
     # 打开指定配置文件, 读取其中信息, 将值存储在 config_str 中, 然后输出到屏幕上
    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    
    # save overloaded model config in the output directory
    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    save_config(cfg, output_config_path)
    
    # 训练
    model = train(cfg, args.local_rank, args.distributed)
    
    # 是否跳过测试
    if not args.skip_test:
        run_test(cfg, model, args.distributed)


if __name__ == "__main__":
    main()
