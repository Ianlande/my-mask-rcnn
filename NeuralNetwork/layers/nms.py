#-*- coding:utf-8 -*-

from NeuralNetwork import _C
#from ._utils import _C

from apex import amp

# Only valid with fp32 inputs - give AMP the hint
nms = amp.float_function(_C.nms)
