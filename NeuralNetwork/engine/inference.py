#-*- coding:utf-8 -*-

import logging
import time
import os

import torch
from tqdm import tqdm # 显示进度条

from NeuralNetwork.data.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str


# 预测结果
def compute_on_dataset(model, data_loader, device, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        # 使用 model 运算, 不用计算梯度
        with torch.no_grad():
            if timer:
                timer.tic()
            else:
                # ============== 最重要的一行 ==============
                # model 输出预测结果
                output = model(images.to(device))
            if timer:
                if not device.type == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict


# 将所有GPU设备上的预测结果累加并返回
def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("NeuralNetwork.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


# 保存预测结果, 输出对结果的评价
def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None, # 自定义输出文件夹
):

    # convert to a torch.device for efficiency
    # 获取设备, 查看有多少 GPU
    device = torch.device(device)
    num_devices = get_world_size()
    
    # 日志信息
    logger = logging.getLogger("NeuralNetwork.inference")
    
    # 自定义数据集
    dataset = data_loader.dataset
    
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    
    # 预测结果
    predictions = compute_on_dataset(model, data_loader, device, inference_timer)
    # wait for all processes to complete before measuring the time
    
    # 等到所有的进程都结束以后再计算总耗时
    synchronize()
    
    # 计算总耗时记入log
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )
    
    # 将所有GPU设备上的预测结果累加并返回
    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return
        
    # 将结果保存
    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )
    
    # 调用评价函数, 返回预测结果的质量
    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
