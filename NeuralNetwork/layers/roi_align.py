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
"""
input:
    bottom_data : torch.Size([256, 200, 272]), cuda if using gpu;
                  图像 256 通道, 高 200, 宽 272
    height : int, 图像高 200
    width : int, 图像宽 272
    y_sample_num, x_sample_num : y,x 方向上每个 bin 的采样点数
    y : torch.Size([y方向上bin数目*y方向上每个bin采样点数]), cuda if using gpu; 一般为 7*2=14
    x : torch.Size([x方向上bin数目*x方向上每个bin采样点数]), cuda if using gpu; 一般为 7*2=14
output:
    一张 feature map 的返回值
    torch.Size([256, 7, 7]), cuda if using gpu
"""
def bilinear_interpolate_kernel(bottom_data, height, width, y_sample_num, x_sample_num, y, x, device):
    
    # 防止超出 bin 的范围, 
    # 当下界低于 0 但是不低于 -1 时还可以补救, 将值变为 0 即可, 因为 f(0,0),f(0,1) 是向下取整
    for each in y:
        if each < -1.0 or each > height:
            return 0
    for each in x:
        if each < -1.0 or each > width:
            return 0 
    
    # 正常的 y 取值在 0-(height-1) 之间, 一共 height 大小, x 同理
    y = y.clamp(0, height-1)
    x = x.clamp(0, width-1)
    
    # f(0,0),f(0,1) 分别是 x,y 向下取整; clamp 限制范围
    y_low = y.floor().clamp(0, height-1)
    x_low = x.floor().clamp(0, width-1)
    
    # f(0,0),f(0,1) 分别是 x,y 向下取整; f(1,0),f(1,1) 理论上分别是 f(0,0)+1,f(0,1)+1 , 除非位置不够
    y_high = y_low + 1
    y_high = y_high.clamp(0, height-1)
    x_high = x_low + 1
    x_high = x_high.clamp(0, width-1)
    
    # 以下开始, 已经换坐标系了, 改为 xy 所在像素格的坐标系
    ly = y - y_low
    lx = x - x_low
    hy = 1.0 - ly
    hx = 1.0 - lx
    
    y_low = y_low.long()
    x_low = x_low.long()
    y_high = y_high.long()
    x_high = x_high.long()
    
    # f(0,0),f(1,0),f(0,1),f(1,1) 的值
    v1 = bottom_data[:,y_low]
    v1 = v1[:,:,x_low]
    v2 = bottom_data[:,y_low]
    v2 = v2[:,:,x_high]
    v3 = bottom_data[:,y_high]
    v3 = v3[:,:,x_low]
    v4 = bottom_data[:,y_high]
    v4 = v4[:,:,x_high]
    
    m = len(hy)
    n = len(ly)
    w1 = hy.reshape(m,1) * hx
    w2 = hy.reshape(m,1) * lx
    w3 = ly.reshape(n,1) * hx
    w4 = ly.reshape(n,1) * lx
    
    # 计算公式
    # result : torch.Size([256, 14, 14])
    result = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
    
    # 求平均值
    in_result = torch.zeros(result.size(0), int(result.size(1)/y_sample_num), result.size(2)).to(device)
    for index in range(y_sample_num):
        temp = result[:,index::y_sample_num]
        in_result += temp
    
    output = torch.zeros(in_result.size(0), in_result.size(1), int(in_result.size(2)/x_sample_num)).to(device)
    for index in range(x_sample_num):
        temp = in_result[:,:,index::x_sample_num]
        output += temp
    
    return output/4


'''
roi align 前向传播
input : tensor, 卷积后的 feature map, 一共有 batch_size 个
rois : tensor, 一共有 batch_size 个, 每个 batch_size 中有多个 roi
pooled_height, pooled_width : int, 表示一个 roi 的横宽划分成几个 bin (几份), 一般都取 7
sampling_ratio : int, 表示每一个 bin 采样多少点

output : tensor 类型
有 roi 个 tensor, 每一个 tensor 有 channels 层, 每一层大小为 pooled_height*pooled_width, 里面储存了每个 bin 计算后的值
'''
def roi_align_forward(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio):
    num_rois = rois.size(0) # roi的个数
    #print('num_rois:    ',num_rois)
    # input 卷积后的 feature map, input.size(0)是输入的batch_size, 没有意义
    channels = input.size(1) # 卷积后的 feature map 的通道数
    height = input.size(2) # 卷积后的 feature map 的高
    width = input.size(3) # 卷积后的 feature map 的宽
    
    if cfg.MODEL.DEVICE == "cuda":
        device = "cuda"
    else:
        device = "cpu"
    
    # offset 是让 roi 缩小的辅助张量
    offset = torch.empty(num_rois, 5) + 1 # 随机生成张量, 加1是为了防止生成的元素为0导致后续无法乘积
    offset = offset.to(device)
    offset[:,1:5] = spatial_scale
    offset[:,0] = rois[:,0]
    offset_rois = rois * offset # 将 roi 缩小至特定大小
    
    # 参数: roi_width, roi_height, bin_size_w, bin_size_h, roi_bin_grid_w, roi_bin_grid_h, count
    # 分别是: roi 的长宽, 每个 bin 的长宽, 每个 bin 分别在长宽上采样多少, 每个 bin 采样点总数
    
    # 全1张量, 用于计算
    ones = torch.zeros(1,num_rois) + 1.0
    ones = ones.to(device)
        
    roi_width = torch.max(offset_rois[:,3] - offset_rois[:,1], ones) # roi 的长宽
    roi_height = torch.max(offset_rois[:,4] - offset_rois[:,2], ones) # roi 的长宽
    bin_size_w = roi_width / pooled_width # 每个 bin 的长宽
    bin_size_h = roi_height / pooled_height # 每个 bin 的长宽
    
    # 确定每一个 roi 里面的每一个 bin 里面的采样点个数:
    # 如果有 sampling_ratio, 则采样点数目为: sampling_ratio * sampling_ratio
    # sampling_ratio 是每一行/列的采样数目
    roi_bin_grid_w = sampling_ratio
    roi_bin_grid_h = sampling_ratio
    
    # 确定每个 roi 的 bin 里面的采样点个数 count, 一般取 4
    count = roi_bin_grid_w * roi_bin_grid_h
    
    # 计算 x, y 张量集合; x, y 是这些采样点的坐标
    # ph,pw 从 x,y 两个方向遍历每一个 bin
    ph = torch.Tensor(list((range(pooled_height)))).to(device)
    pw = torch.Tensor(list((range(pooled_width)))).to(device)
    iy = torch.Tensor(list((range(roi_bin_grid_h)))).to(device)
    ix = torch.Tensor(list((range(roi_bin_grid_w)))).to(device)
    
    # bin 里面有 00, 01, 10, 11 四个点
    # 获取坐标的原理是: 先找到是哪一个 bin, 然后根据 bin 框的长宽平均分成这几个点, 取坐标
    # 以下使用张量计算
    '''
        roi_start_h : y 方向上 roi 起始坐标
        ph : y 方向上哪一个 bin, 乘上 bin_size_h 后即找到那个 bin 的起始位置
        bin_size_h / roi_bin_grid_h : y 方向上 bin size 根据采样点数量平均分成几份, 每一份的长度
        再乘上 iy 得到那个点的位置, 加 0.5 应该是为了防止采样采到 bin 的边界上, 所以挪动 0.5 个像素
        x 方向上同理
        因此:
        y 有 num_rois 维度, 每个维度的元素数量: y方向上的bin数量 * y方向上每个bin的采样点数量
        x 方向上同理
    '''
    # 计算 y
    temp1 = (bin_size_h.t() * ph).t()
    temp2 = ((bin_size_h / roi_bin_grid_h).t() * (iy + 0.5)).t()
    temp3 = torch.tensor([]).to(device)
    for each in temp1:
        submatrix = each + temp2
        temp3 = torch.cat([temp3,submatrix],dim=0)
    y = offset_rois[:,2].reshape(num_rois,1) + temp3.t()
    
    # 计算 x
    temp1 = (bin_size_w.t() * pw).t()
    temp2 = ((bin_size_w / roi_bin_grid_w).t() * (ix + 0.5)).t()
    temp3 = torch.tensor([]).to(device)
    for each in temp1:
        submatrix = each + temp2
        temp3 = torch.cat([temp3,submatrix],dim=0)
    x = offset_rois[:,1].reshape(num_rois,1) + temp3.t()
    
    # 初始化 output
    output = torch.tensor([]).to(device)
    
    # 遍历每一个 roi
    for each in range(num_rois):
        # roi batch index
        roi_batch_ind = int(offset_rois[each][0])
        
        """
        双线性插值求出结果:
        对每个bin中的采样点(x,y)利用双线性插值得到f(x,y), 最后取平均作为这个 bin 的值
        input[roi_batch_ind]:
            一个 feature map 有多个通道, 每个通道都要计算, 需要遍历每一个通过
        val(output):
            channel 个维度, 各维度是 m*n 个 bin, 每个 bin 只有一个值, 即计算出来的平均值
        """
        val = bilinear_interpolate_kernel(input[roi_batch_ind], height, width, sampling_ratio, sampling_ratio, y[each], x[each], device)
        output = torch.cat([output,val.unsqueeze(0)],dim=0)
        
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

