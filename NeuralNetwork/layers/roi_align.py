#-*- coding:utf-8 -*-

import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from NeuralNetwork.config import cfg

from apex import amp
import math


# 双线性插值算法
def bilinear_interpolate(bottom_data, height, width, y, x):
    
    # 防止超出 bin 的范围, 
    # 当下界低于 0 但是不低于 -1 时还可以补救, 将值变为 0 即可, 因为 f(0,0),f(0,1) 是向下取整
    if y < -1.0 or y > height or x < -1.0 or x > width :
        return 0
    if y <= 0 :
        y = 0
    if x <= 0 :
        x = 0
    
    # f(0,0),f(0,1) 分别是 x,y 向下取整
    y_low = int(y)
    x_low = int(x)
    
    # f(0,0),f(0,1) 分别是 x,y 向下取整; f(1,0),f(1,1) 理论上分别是 f(0,0)+1,f(0,1)+1 , 除非位置不够
    if y_low >= height - 1: # 位置不够
        y_high = y_low = height - 1
        y = y_low
    else:
        y_high = y_low + 1
    
    if x_low >= width - 1: # 位置不够
        x_high = x_low = width - 1
        x = x_low
    else:
        x_high = x_low + 1
    
    
    # 以下开始, 已经换坐标系了, 改为 xy 所在像素格的坐标系
    ly = y - y_low
    lx = x - x_low
    hy = 1.0 - ly
    hx = 1.0 - lx
    
    # f(0,0),f(1,0),f(0,1),f(1,1) 的值
    v1 = bottom_data[y_low][x_low]
    v2 = bottom_data[y_low][x_high]
    v3 = bottom_data[y_high][x_low]
    v4 = bottom_data[y_high][x_high]
    
    w1 = hy * hx
    w2 = hy * lx
    w3 = ly * hx
    w4 = ly * lx
    
    # 计算公式
    val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
    
    return val


'''
roi align 前向传播
input : tensor, 卷积后的 feature map, 一共有 batch_size 个
rois : tensor, 一共有 batch_size 个, 每个 batch_size 中有多个 roi
pooled_height, pooled_width : int, 表示一个 roi 的横宽划分成几个 bin (几份)
sampling_ratio : int, 表示每一个 bin 采样多少点

output : tensor 类型
有 roi 个 tensor, 每一个 tensor 有 channels 层, 每一层大小为 pooled_height*pooled_width, 里面储存了每个 bin 计算后的值
'''
# 效率太低, 已经弃用
def roi_align_forward(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio):
    num_rois = rois.size(0) # roi的个数
    #print('num_rois:    ',num_rois)
    # input 卷积后的 feature map, input.size(0)是输入的batch_size, 没有意义
    channels = input.size(1) # 卷积后的 feature map 的通道数
    height = input.size(2) # 卷积后的 feature map 的高
    width = input.size(3) # 卷积后的 feature map 的宽
    
    # 初始化 output , 全零 tensor
    output = torch.zeros(num_rois, channels, pooled_height, pooled_width)
    if cfg.MODEL.DEVICE == "cuda":
        output = output.to(cfg.MODEL.DEVICE)
    
    # 为了保持 output 和 input 格式一样, 对 input 进行 copy, 然后修改大小
    #output = input.clone()
    #output = output.resize_(num_rois, channels, pooled_height, pooled_width)
    
    roi_index = 0
    # 遍历每一个 roi
    for offset_bottom_rois in rois:
        print('roi_index:   ',roi_index)
        
        # roi batch index
        roi_batch_ind = int(offset_bottom_rois[0])
        
        # roi 始末位置在 feature map 里面的坐标
        # spatial_scale 将 roi 在原图中的坐标缩放至 featuremap 的坐标, 无量化损失
        roi_start_w = offset_bottom_rois[1] * spatial_scale
        roi_start_h = offset_bottom_rois[2] * spatial_scale
        roi_end_w = offset_bottom_rois[3] * spatial_scale
        roi_end_h = offset_bottom_rois[4] * spatial_scale
        
        # roi 的长宽
        roi_width = max(roi_end_w - roi_start_w, 1.0)
        roi_height = max(roi_end_h - roi_start_h, 1.0)
        
        # 每个 bin 的长宽
        bin_size_w = roi_width / pooled_width
        bin_size_h = roi_height / pooled_height
        
        # 确定每一个 roi 里面的每一个 bin 里面的采样点个数:
        # 如果有 sampling_ratio, 则采样点数目为: sampling_ratio * sampling_ratio
        # sampling_ratio 是每一行/列的采样数目
        # 如果没有 sampling_ratio, 则需要计算一下
        # 如果需要列/行的采样点不一样, 则使用 pooled_height, pooled_width
        if sampling_ratio > 0:
            roi_bin_grid_h = sampling_ratio
            roi_bin_grid_w = sampling_ratio
        else:
            roi_bin_grid_h = math.ceil(roi_height / pooled_height)
            roi_bin_grid_w = math.ceil(roi_width / pooled_width)
            
        # 确定每个 roi 的 bin 里面的采样点个数 count, 一般取 4
        count = roi_bin_grid_h * roi_bin_grid_w
        
        # 一个 feature map 有多个通道, 每个通道都要计算, 需要遍历每一个通过
        for channel_index in range(channels):
            print('channel_index:   ',channel_index)
            
            # ph,pw 从 x,y 两个方向遍历每一个 bin
            for ph in range(pooled_height):
                for pw in range(pooled_width):
                    
                    # 对每个bin中的采样点(x,y)利用双线性插值得到f(x,y), 最后取平均作为这个 bin 的值
                    output_val = 0
                    for iy in range(roi_bin_grid_h):
                        for ix in range(roi_bin_grid_w):
                            # 以上四次迭代, 分别是 bin 里面的 00, 01, 10, 11 四个点
                            # x, y 是这些采样点的坐标
                            # 获取坐标的原理是: 先找到是哪一个 bin, 然后根据 bin 框的长宽平均分成这几个点, 取坐标
                            '''
                                roi_start_h : y 方向上 roi 起始坐标
                                ph : y 方向上哪一个 bin, 乘上 bin_size_h 后即找到那个 bin 的起始位置
                                bin_size_h / roi_bin_grid_h : y 方向上 bin size 根据采样点数量平均分成几份, 每一份的长度
                                再乘上 iy 得到那个点的位置, 加 0.5 应该是为了防止采样采到 bin 的边界上, 所以挪动 0.5 个像素
                                x 方向上同理
                            '''
                            y = roi_start_h + ph * bin_size_h + (iy + 0.5) * bin_size_h / roi_bin_grid_h
                            x = roi_start_w + pw * bin_size_w + (ix + 0.5) * bin_size_w / roi_bin_grid_w
                            # 双线性插值求出结果
                            val = bilinear_interpolate(input[roi_batch_ind][channel_index], height, width, y, x)
                            output_val = output_val + val
                            
                    # 取平均值
                    output_val /= count;
                    output[roi_index][channel_index][ph][pw] = output_val
        roi_index += 1
        
    return output


class _ROIAlign(Function):
    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale, sampling_ratio):
        ctx.save_for_backward(roi)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input.size()
        output = roi_align_forward(
            input, roi, spatial_scale, output_size[0], output_size[1], sampling_ratio
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        bs, ch, h, w = ctx.input_shape
        grad_input = roi_align_backward(
            grad_output,
            rois,
            spatial_scale,
            output_size[0],
            output_size[1],
            bs,
            ch,
            h,
            w,
            sampling_ratio,
        )
        return grad_input, None, None, None, None


roi_align = _ROIAlign.apply

class ROIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio):
        super(ROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    @amp.float_function
    def forward(self, input, rois):
        #print('input:  ',input)
        #print('rois:  ',rois)
        return roi_align(
            input, rois, self.output_size, self.spatial_scale, self.sampling_ratio
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ")"
        return tmpstr

