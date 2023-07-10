
import mmcv
import copy
import torch

import numpy as np
import torch.nn as nn
from abc import ABCMeta, abstractmethod
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from depth.ops import resize
from depth.models.builder import build_loss


class DepthBaseDecodeHead(BaseModule, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.

    Args:
        in_channels (List): Input channels.
        channels (int): Channels after modules, before conv_depth.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        loss_decode (dict): Config of decode loss.
            Default: dict(type='SigLoss').
        sampler (dict|None): The config of depth map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        min_depth (int): Min depth in dataset setting.
            Default: 1e-3.
        max_depth (int): Max depth in dataset setting.
            Default: None.
        norm_cfg (dict|None): Config of norm layers.
            Default: None.
        classify (bool): Whether predict depth in a cls.-reg. manner.
            Default: False.
        n_bins (int): The number of bins used in cls. step.
            Default: 256.
        bins_strategy (str): The discrete strategy used in cls. step.
            Default: 'UD'.
        norm_strategy (str): The norm strategy on cls. probability 
            distribution. Default: 'linear'
        scale_up (str): Whether predict depth in a scale-up manner.
            Default: False.
    """

    def __init__(self,
                 in_channels,
                 channels=96,
                 conv_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 loss_decode=dict(
                     type='SigLoss',
                     valid_mask=True,
                     loss_weight=10),
                 sampler=None,
                 in_index=-1,
                 input_transform=None,
                 align_corners=False,
                 min_depth=1e-3,
                 max_depth=None,
                 norm_cfg=None,
                 classify=False,
                 n_bins=256,
                 bins_strategy='UD',
                 norm_strategy='linear',
                 scale_up=False,
                 dropout_ratio=0.1,
                 init_inputs=False,
                 use_eps=True,
                 ):
        super(DepthBaseDecodeHead, self).__init__()
        if init_inputs:
            self._init_inputs(in_channels, in_index, input_transform)
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg
        self.loss_decode = build_loss(loss_decode)
        self.align_corners = align_corners
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.norm_cfg = norm_cfg
        self.classify = classify
        self.n_bins = n_bins
        self.scale_up = scale_up
        self.dropout_ratio = dropout_ratio
        self.use_eps = use_eps

        if self.classify:
            assert bins_strategy in ["UD", "SID"], "Support bins_strategy: UD, SID"
            assert norm_strategy in ["linear", "softmax", "sigmoid"], "Support norm_strategy: linear, softmax, sigmoid"

            self.bins_strategy = bins_strategy
            self.norm_strategy = norm_strategy
            self.softmax = nn.Softmax(dim=1)
            self.conv_depth = nn.Conv2d(channels, n_bins, kernel_size=3, padding=1, stride=1)
        else:
            self.conv_depth = nn.Conv2d(channels, 1, kernel_size=3, padding=1, stride=1)
        

        self.fp16_enabled = False
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def extra_repr(self):
        """Extra repr."""
        s = f'align_corners={self.align_corners}'
        return s

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass
    
    @auto_fp16()
    @abstractmethod
    def forward(self, inputs, img_metas):
        """Placeholder of forward function."""
        pass

    def forward_train(self, img, inputs, img_metas, depth_gt, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `depth/datasets/pipelines/formatting.py:Collect`.
            depth_gt (Tensor): GT depth
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        depth_pred = self.forward(inputs, img_metas)
        losses = self.losses(depth_pred, depth_gt)

        log_imgs = self.log_images(img[0], depth_pred[0], depth_gt[0], img_metas[0])
        losses.update(**log_imgs)

        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `depth/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output depth map.
        """
        return self.forward(inputs, img_metas)

    def depth_pred(self, feat):
        """Prediction each pixel."""
        if self.classify:
            logit = self.conv_depth(feat)

            if self.bins_strategy == 'UD':
                bins = torch.linspace(self.min_depth, self.max_depth, self.n_bins, device=feat.device)
            elif self.bins_strategy == 'SID':
                bins = torch.logspace(self.min_depth, self.max_depth, self.n_bins, device=feat.device)

            # following Adabins, default linear
            if self.norm_strategy == 'linear':
                logit = torch.relu(logit)
                eps = 0.1
                logit = logit + eps
                logit = logit / logit.sum(dim=1, keepdim=True)
            elif self.norm_strategy == 'softmax':
                logit = torch.softmax(logit, dim=1)
            elif self.norm_strategy == 'sigmoid':
                logit = torch.sigmoid(logit)
                logit = logit / logit.sum(dim=1, keepdim=True)

            output = torch.einsum('ikmn,k->imn', [logit, bins]).unsqueeze(dim=1)

        else:
            if self.scale_up:
                if self.use_eps:
                    eps = self.max_depth
                else:
                    eps = 1
                output = self.sigmoid(self.conv_depth(feat)) * eps
            else:
                if self.use_eps:
                    eps = self.min_depth
                else:
                    eps = 0
                output = self.relu(self.conv_depth(feat)) + eps
        return output

    @force_fp32(apply_to=('depth_pred', ))
    def losses(self, depth_pred, depth_gt):
        """Compute depth loss."""
        loss = dict()
        depth_pred = resize(
            input=depth_pred,
            size=depth_gt.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners,
            warning=False)
        loss['loss_depth'] = self.loss_decode(
            depth_pred,
            depth_gt)
        return loss

    def log_images(self, img_path, depth_pred, depth_gt, img_meta):
        show_img = copy.deepcopy(img_path.detach().cpu().permute(1, 2, 0))
        show_img = show_img.numpy().astype(np.float32)
        show_img = mmcv.imdenormalize(show_img, 
                                      img_meta['img_norm_cfg']['mean'],
                                      img_meta['img_norm_cfg']['std'], 
                                      img_meta['img_norm_cfg']['to_rgb'])
        show_img = np.clip(show_img, 0, 255)
        show_img = show_img.astype(np.uint8)
        show_img = show_img[:, :, ::-1]
        show_img = show_img.transpose(0, 2, 1)
        show_img = show_img.transpose(1, 0, 2)

        depth_pred = depth_pred / torch.max(depth_pred)
        depth_gt = depth_gt / torch.max(depth_gt)

        depth_pred_color = copy.deepcopy(depth_pred.detach().cpu())
        depth_gt_color = copy.deepcopy(depth_gt.detach().cpu())

        return {"img_rgb": show_img, "img_depth_pred": depth_pred_color, "img_depth_gt": depth_gt_color}