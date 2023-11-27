# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import random
import typing as tp

import torch
from torch import nn
from torch.nn import functional as F
import torchaudio as ta

from .common import (
    ConvSequence, ScaledEmbedding, SubjectLayers,
    DualPathRNN, ChannelMerger, ChannelDropout, pad_multiple, SpatialAttention, SubjectLayer
)


class SimpleConv(nn.Module):   
  def __init__(self, F_out, inchans, outchans, K, n_subjects=None):
    super().__init__()
    self.D2 = 320
    self.outchans = outchans
    self.spatial_attention = SpatialAttention(inchans, outchans, K)
    self.conv = nn.Conv2d(outchans, outchans, 1, padding='same')
    if n_subjects:
      self.subject_layer = SubjectLayer(outchans, n_subjects)
    self.conv_blocks = nn.Sequential(*[self.generate_conv_block(k) for k in range(5)]) # 5 conv blocks
    self.final_convs = nn.Sequential(
      nn.Conv2d(self.D2, self.D2*2, 1),
      nn.GELU(),
      nn.Conv2d(self.D2*2, F_out, 1)
    )
    
  def generate_conv_block(self, k):
    kernel_size = (1,3)
    padding = 'same' # (p,0)
    return nn.Sequential(OrderedDict([
      ('conv1', nn.Conv2d(self.outchans if k==0 else self.D2, self.D2, kernel_size, dilation=pow(2,(2*k)%5), padding=padding)),
      ('bn1',   nn.BatchNorm2d(self.D2)), 
      ('gelu1', nn.GELU()),
      ('conv2', nn.Conv2d(self.D2, self.D2, kernel_size, dilation=pow(2,(2*k+1)%5), padding=padding)),
      ('bn2',   nn.BatchNorm2d(self.D2)),
      ('gelu2', nn.GELU()),
      ('conv3', nn.Conv2d(self.D2, self.D2*2, kernel_size, padding=padding)),
      ('glu',   nn.GLU(dim=1))
    ]))

  def forward(self, x, batch):
    subjects = batch.subject_index
    x = self.spatial_attention(x, batch).unsqueeze(2) # add dummy dimension at the end
    x = self.conv(x)
    x = self.subject_layer(x, subjects)
        
    for k in range(len(self.conv_blocks)):
      if k == 0:
        x = self.conv_blocks[k](x)
      else:
        x_copy = x
        for name, module in self.conv_blocks[k].named_modules():
          if name == 'conv2' or name == 'conv3':
            x = x_copy + x # residual skip connection for the first two convs
            x_copy = x.clone() # is it deep copy?
          x = module(x)
    x = self.final_convs(x)
        
    return x.squeeze(2)
