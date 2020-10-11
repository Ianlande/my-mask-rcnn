#-*- coding:utf-8 -*-

import torch
import numpy as np
from apex import amp


# nms: non_max_suppress
def nms(dets_cuda, scores_cuda, threshold):
    
    # 因为cuda上的数据不能使用numpy, 所以要储存在cpu才行
    dets = dets_cuda.cpu()
    scores = scores_cuda.cpu()
    
    x1 = dets[:,0]
    y1 = dets[:,1]
    x2 = dets[:,2]
    y2 = dets[:,3]
        
    areas = (y2-y1+1) * (x2-x1+1)
    
    # scores 从大到小排列的索引号, argsort(descending=True): 从大到小
    order = scores.argsort(descending=True)
    
    # 存放 NMS 后剩余的方框
    keep = []
    
    # 剔除遍历过的方框
    while order.size()[0] > 0:
        # 取出第一个方框进行和其他方框比对
        i = order[0]
        
        # 因为分数已经按从大到小排列了
        # 所以如果有合并存在, 也是保留分数最高的这个, 也就是保留第一个
        # 因为每次第一个都是分数最高的, 所以直接储存它
        # keep 存储的是索引值
        keep.append(i)
        
        # 计算分数最高的窗口(第i个)与其他所有窗口的交叠部分的面积的左上和右下坐标
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        # 如果两个方框相交, X22-X11 和 Y22-Y11 为正
        # 如果两个方框不相交, X22-X11 和 Y22-Y11 为负, 把不相交的W和H设为0
        w = np.maximum(0.0, xx2-xx1+1)
        h = np.maximum(0.0, yy2-yy1+1)
        
        # 求所有的交叠面积
        inter = w * h
        
        # IOU 公式求交并比
        # 得出来的 overlap 是一个列表, 里面拥有当前方框和其他所有方框的 IOU 结果
        overlap = inter / (areas[i] + areas[order[1:]] - inter)
        #print("overlap", overlap)
        # 大于阈值 threshold 的剔除, 小于阈值的才保留
        inds = np.where(overlap <= threshold)[0]
        #print("ind", inds)
        order = order[inds+1] # 下一轮迭代
        #print("order", order)
        
    return keep


# test
def main():
    dets=torch.tensor(
            [[218, 322, 385, 491, 0.98],[247, 312, 419, 461, 0.83],[237, 344, 407, 510, 0.92],
            [757, 218, 937, 394, 0.96],[768, 198, 962, 364, 0.85],[740, 240, 906, 414, 0.83],
            [1101, 84, 1302, 303, 0.82], [1110, 67, 1331, 260, 0.97], [1123, 42, 1362, 220, 0.85]]
            )
    dets = dets.cuda() # 放入GPU中运行
    scores = dets[:,4]
    
    keep = nms(dets, scores, threshold=0.5)
    print('keep:    ',keep)

if __name__ == "__main__":
    main()
