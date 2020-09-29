#-*- coding:utf-8 -*-

"""Centralized catalog of paths."""

import os
from copy import deepcopy

class DatasetCatalog(object):
    DATA_DIR = r"datasets" # datasets 文件夹的路径
    DATASETS = {
        "coco_2014_train": {
            "img_dir": r"demo\train2014", # 图片路径, 根据实际情况修改
            "ann_file": r"demo\annotations\instances_train2014.json" # 标注文件路径, 根据实际情况修改
        },
        "coco_2014_val": {
            "img_dir": r"demo\val2014", # 图片路径, 根据实际情况修改
            "ann_file": r"demo\annotations\instances_val2014.json" # 标注文件路径, 根据实际情况修改
        },
    }

    @staticmethod
    def get(name):
        if "coco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="COCODataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))
