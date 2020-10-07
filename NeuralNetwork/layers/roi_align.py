#-*- coding:utf-8 -*-

import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from apex import amp

'''
def bilinear_interpolate(
                         height,
                         width,
                         pooled_height,
                         pooled_width,
                         iy_upper,
                         ix_upper,
                         roi_start_h,
                         roi_start_w,
                         bin_size_h,
                         bin_size_w,
                         roi_bin_grid_h,
                         roi_bin_grid_w
                         ):
    pass

def RoIAlignForward_kernel():
    pass

def roi_align_forward(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio):
    num_rois = rois.size(0)
    channels = input.size(1)
    height = input.size(2)
    width = input.size(3)
    
    output = empty({num_rois, channels, pooled_height, pooled_width}, input.options())
    output_size = num_rois * pooled_height * pooled_width * channels
    
    if output.numel() == 0 :
        return output
    
    RoIAlignForward_kernel(
                           output_size,
                           input.data(),
                           spatial_scale,
                           channels,
                           height,
                           width,
                           pooled_height,
                           pooled_width,
                           sampling_ratio,
                           rois.data(),
                           output.data()
                           )
    
    
def roi_align_backward():
    pass
'''

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
        print('input:  ',input)
        print('rois:  ',rois)
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
