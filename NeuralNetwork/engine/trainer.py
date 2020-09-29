#-*- coding:utf-8 -*-

import datetime
import logging # 日志模块, 和 print 差不多功能
import os
import time

import torch
import torch.distributed as dist # 多 GPU 分布式计算需要
from tqdm import tqdm # 进度条配置, 用于显示进度条

from NeuralNetwork.data.build import make_data_loader
from NeuralNetwork.utils.comm import get_world_size, synchronize
from NeuralNetwork.utils.metric_logger import MetricLogger
from NeuralNetwork.engine.inference import inference

from apex import amp


# 对 loss 进行 reduce, 用于处理多 GPU 计算的问题
def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    
    # 单 GPU, 直接返回, 无需reduce
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
        
    # 多 GPU 时
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


'''
    神经网络训练的原理:
        1. 迭代, 遍历每一个数据
        2. 模型(model)做出选择
        2. 计算模型(model)返回的 loss
        3. 优化(Optimize), 这一步主要是通过 loss 修正模型(model)参数
'''
# 模型训练核心代码
def do_train(
    cfg, # 配置文件 cfg
    model, # 从 NeuralNetwork/modeling/detector/detectors.py 的 build_detection_model 函数得到的模型对象
    data_loader, # PyTorch 的 DataLoader 对象, 对应数据集
    data_loader_val, # 对应测试集
    optimizer, # torch.optim.sgd.SGD 对象
    scheduler, # 学习率的更新策略, 封装在 solver/lr_scheduler.py 文件中
    checkpointer, # DetectronCheckpointer, 用于自动转换 Caffe2 Detectron 的模型文件
    device, # 训练于 cpu or cuda(GPU)
    checkpoint_period, # int, 指定模型的保存迭代间隔, 默认为 2500
    test_period, # 测试的保存间隔
    arguments, # 额外的其他参数
):
    # 记录日志信息
    logger = logging.getLogger("NeuralNetwork.trainer")
    logger.info("Start training")
    
    # 用于记录一些变量的滑动平均值和全局平均值
    # delimiter 为定界符, 这里用两个空格作为定界符
    meters = MetricLogger(delimiter="  ")
    
    # 数据载入器重写了 len 函数, 使其返回载入器需要提供 batch 的次数, 即 cfg.SOLVER.MAX_ITER
    max_iter = len(data_loader)
    
    # 开始训练的迭代数, 默认为0, 但是会根据载入的权重文件, 变成其他值
    start_iter = arguments["iteration"]
    
    # 将 model 的模式置为 train
    model.train()
    
    # 计时
    start_training_time = time.time()
    end = time.time()

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    
    dataset_names = cfg.DATASETS.TEST
    
    # 遍历 data_loader, 第二个参数是设置序号的开始序号,
    # data_loader 的返回值为 (images, targets, shape)
    # ================= 遍历 =================
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        
        if any(len(target) < 1 for target in targets):
            logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
            continue
        
        # 获取一个 batch 所需的时间
        data_time = time.time() - end
        
        # 更新迭代次数 iteration
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device) # 移动到 device 上
        targets = [target.to(device) for target in targets] # 移动到 device 上
        
        # ================= Compute the loss =================
        # model 根据 images 和 targets 做出反应, 返回 loss
        # model 在 NeuralNetwork/modeling/detector/generalized_rcnn.py
        loss_dict = model(images, targets)
        # 将 boundingbox loss, mask loss 合并
        losses = sum(loss for loss in loss_dict.values())
        # ================= Compute the loss =================
        
        # 根据 GPUs 的数量对 loss 进行 reduce
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values()) # 合并loss
        meters.update(loss=losses_reduced, **loss_dict_reduced) # 更新滑动平均值
        
        # 清除梯度缓存
        optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward() # 计算梯度
        
        # ================= Optimize the network =================
        optimizer.step() # 更新参数
        scheduler.step()
        # ================= Optimize the network =================
        
        # 进行一次 batch 所需时间
        batch_time = time.time() - end
        end = time.time()
        
        meters.update(time=batch_time, data=data_time)
        
        # 根据时间的滑动平均值计算大约还剩多长时间结束训练
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        
        # 每经过20次迭代, 输出一次训练状态
        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
            
        # 每经过 checkpoint_period 次迭代后, 就将模型保存
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        
        if data_loader_val is not None and test_period > 0 and iteration % test_period == 0:
            meters_val = MetricLogger(delimiter="  ")
            synchronize()
            _ = inference(  # The result can be used for additional logging, e. g. for TensorBoard
                model,
                # The method changes the segmentation mask format in a data loader,
                # so every time a new data loader is created:
                make_data_loader(cfg, is_train=False, is_distributed=(get_world_size() > 1), is_for_period=True),
                dataset_name="[Validation]",
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=None,
            )
            synchronize()
            model.train()
            with torch.no_grad():
                # Should be one image for each GPU:
                for iteration_val, (images_val, targets_val, _) in enumerate(tqdm(data_loader_val)):
                    images_val = images_val.to(device)
                    targets_val = [target.to(device) for target in targets_val]
                    loss_dict = model(images_val, targets_val)
                    losses = sum(loss for loss in loss_dict.values())
                    loss_dict_reduced = reduce_loss_dict(loss_dict)
                    losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                    meters_val.update(loss=losses_reduced, **loss_dict_reduced)
            synchronize()
            logger.info(
                meters_val.delimiter.join(
                    [
                        "[Validation]: ",
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters_val),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        
        # 达到最大迭代次数后, 也进行保存
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)
            
    # 输出总的训练耗时
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
