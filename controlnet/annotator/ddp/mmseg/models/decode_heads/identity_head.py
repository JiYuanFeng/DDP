from mmcv.runner import BaseModule, auto_fp16
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
import torch.nn as nn


@HEADS.register_module()
class IdentityHead(BaseDecodeHead):
    
    def __init__(self, **kwargs):
        super().__init__(in_channels=256, channels=256, **kwargs)
        self.conv_seg = nn.Identity()
    
    @auto_fp16()
    def forward(self, x):
        return x