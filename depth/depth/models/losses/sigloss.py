# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from depth.models.builder import LOSSES

@LOSSES.register_module()
class SigLoss(nn.Module):
    """SigLoss.

        We adopt the implementation in `Adabins <https://github.com/shariqfarooq123/AdaBins/blob/main/loss.py>`_.

    Args:
        valid_mask (bool): Whether filter invalid gt (gt > 0). Default: True.
        loss_weight (float): Weight of the loss. Default: 1.0.
        max_depth (int): When filtering invalid gt, set a max threshold. Default: None.
        warm_up (bool): A simple warm up stage to help convergence. Default: False.
        warm_iter (int): The number of warm up stage. Default: 100.
    """

    def __init__(self,
                 valid_mask=True,
                 loss_weight=1.0,
                 max_depth=None,
                 warm_up=False,
                 loss_name='sigloss',
                 warm_iter=100):
        super(SigLoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth

        self.eps = 0.001 # avoid grad explode

        # HACK: a hack implementation for warmup sigloss
        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0
        self._loss_name = loss_name

    def sigloss(self, input, target):
        if self.valid_mask:
            valid_mask = target > 0
            if self.max_depth is not None:
                valid_mask = torch.logical_and(target > 0, target <= self.max_depth)
            input = input[valid_mask]
            target = target[valid_mask]
        
        if self.warm_up:
            if self.warm_up_counter < self.warm_iter:
                g = torch.log(input + self.eps) - torch.log(target + self.eps)
                g = 0.15 * torch.pow(torch.mean(g), 2)
                self.warm_up_counter += 1
                return torch.sqrt(g)

        g = torch.log(input + self.eps) - torch.log(target + self.eps)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return torch.sqrt(Dg)

    def forward(self, depth_pred, depth_gt):
        """Forward function."""
        
        loss_depth = self.loss_weight * self.sigloss(depth_pred, depth_gt)
        return loss_depth

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
