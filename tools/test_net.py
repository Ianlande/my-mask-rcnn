#-*- coding:utf-8 -*-

import sys
import argparse
import os
import torch


# 设置一下当前地址, 否则无法查找到 NeuralNetwork 模块
this_dir = os.path.dirname(os.path.abspath(__file__))
this_dir = os.path.dirname(this_dir)
sys.path.append(this_dir)

from NeuralNetwork.utils.env import setup_environment  # noqa F401 isort:skip

from NeuralNetwork.config import cfg
from NeuralNetwork.data.build import make_data_loader
from NeuralNetwork.engine.inference import inference
from NeuralNetwork.modeling.detector.detectors import build_detection_model

from NeuralNetwork.utils.checkpoint import DetectronCheckpointer
from NeuralNetwork.utils.collect_env import collect_env_info
from NeuralNetwork.utils.comm import synchronize, get_rank
from NeuralNetwork.utils.logger import setup_logger
from NeuralNetwork.utils.miscellaneous import mkdir

# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument("--config-file",default="",metavar="FILE",help="path to config file")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--ckpt",help="The path to the checkpoint for test, default is the latest checkpoint.",default=None)
    parser.add_argument("opts",help="Modify config options using the command-line",default=None,nargs=argparse.REMAINDER)
    args = parser.parse_args()
    
    # GPU 数量, 根据 GPU 数量判断是否使用分布式计算
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    
    # 分布式计算
    if distributed:
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
    
    # 保存在 output_dir 下的 log.txt 文件
    save_dir = ""
    logger = setup_logger("NeuralNetwork", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())
    
    # 加载模型
    model = build_detection_model(cfg)
    
    # 使用cuda或者cpu
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)# 将模型存入GPU或者cpu

    # Initialize mixed-precision if necessary
    use_mixed_precision = cfg.DTYPE == 'float16'
    amp_handle = amp.init(enabled=use_mixed_precision, verbose=cfg.AMP_VERBOSE)
    
    # 输出文件夹
    output_dir = cfg.OUTPUT_DIR
    
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)
    
    # 添加 bounding box 和 mask segmentation
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    
    # 根据数据集的数量定义输出文件夹
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    
    # 创建输出文件夹
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    
    # 加载测试数据集
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    
    # 对数据集中的数据按批次调用 inference 函数
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
        synchronize()


if __name__ == "__main__":
    main()
