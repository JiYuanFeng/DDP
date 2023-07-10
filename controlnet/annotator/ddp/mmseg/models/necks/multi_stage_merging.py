# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, xavier_init
from mmseg.ops import resize
from mmcv.runner import BaseModule

from ..builder import NECKS


@NECKS.register_module()
class MultiStageMerging(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 align_corners=False,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(MultiStageMerging, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.align_corners = align_corners
        self.down = ConvModule(
            sum(in_channels),
            out_channels,
            kernel_size,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        outs = list()
        size = inputs[0].shape[2:]
        for index, input in enumerate(inputs):
            input = resize(input,
                           size=size,
                           mode='bilinear',
                           align_corners=self.align_corners)
            outs.append(input)
        out = torch.cat(outs, dim=1)
        out = self.down(out)
        return [out]
