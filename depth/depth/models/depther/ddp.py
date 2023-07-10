# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .. import builder
from ..builder import DEPTHER
from .encoder_decoder import DepthEncoderDecoder
from einops import rearrange, reduce, repeat
from mmcv.cnn import ConvModule

from ...core import add_prefix
from ...ops import resize


class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


@DEPTHER.register_module()
class DDP(DepthEncoderDecoder):
    """Encoder Decoder segmentors.
    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 bit_scale=1,
                 bits=8,
                 timesteps=1,
                 randsteps=1,
                 time_difference=1,
                 learned_sinusoidal_dim=16,
                 sample_range=(0, 0.999),
                 ddim=True,
                 rule=None,
                 min_depth=1e-3,
                 max_depth=80,
                 **kwargs):
        super(DDP, self).__init__(**kwargs)

        self.bit_scale = bit_scale
        self.BITS = bits
        self.timesteps = timesteps
        self.randsteps = randsteps
        self.time_difference = time_difference
        self.sample_range = sample_range
        self.ddim = ddim
        self.min_depth = min_depth
        self.max_depth = max_depth

        print("sample range:", sample_range)
        print("timesteps: {}, randsteps: {}".format(timesteps, randsteps))

        self.down = ConvModule(
            self.decode_head.in_channels[0] + 1,
            self.decode_head.in_channels[0],
            1,
            padding=0,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None
        )

        # time embeddings
        time_dim = self.decode_head.in_channels[0] * 4  # 1024
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(  # [2,]
            sinu_pos_emb,  # [2, 17]
            nn.Linear(fourier_dim, time_dim),  # [2, 1024]
            nn.GELU(),
            nn.Linear(time_dim, time_dim)  # [2, 1024]
        )

        for k, v in self.named_parameters():
            if rule == "freeze_backbone":
                if "backbone" in k:
                    v.requires_grad = False

    def encode_decode(self, img, img_metas, rescale=False):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)[0]
        out = self.sample(x, img_metas)

        out = torch.clamp(out, min=self.decode_head.min_depth, max=self.decode_head.max_depth)
        if rescale:
            out = resize(
                input=out,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        return out

    def forward_train(self, img, img_metas, depth_gt):
        """Forward function for training.
        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        # backbone & neck
        x = self.extract_feat(img)[0]  # bs, 256, h/4, w/4

        batch, c, h, w, device, = *x.shape, x.device

        # Mask -> bit2int [-1, 1]
        depth_gt_norm = resize(depth_gt.float(), size=(h, w), mode="bilinear")
        depth_gt_norm = depth_gt_norm.to(depth_gt.dtype)
        depth_gt_norm = ((depth_gt_norm - self.min_depth) / (self.max_depth - self.min_depth))
        depth_gt_norm = ((depth_gt_norm * 2) - 1) * self.bit_scale
        # corrupt
        t = torch.zeros((batch,), device=device).float().uniform_(self.sample_range[0], self.sample_range[1])  # [bs]
        input_t = self.time_mlp(t)  # [2, 1024]
        padd_t = self.right_pad_dims_to(img, t)  # [bs] -> [bs, 1, 1, 1]
        # random noise
        eps = torch.randn_like(depth_gt_norm)
        gt_crpt = torch.sqrt(self.gamma(padd_t)) * depth_gt_norm + torch.sqrt(1 - self.gamma(padd_t)) * eps

        # conditional input
        feat = torch.cat([x, gt_crpt], dim=1)  # [bs, 256+1, h/4, w/4]
        feat = self.down(feat)  # [bs, 256, h/4, w/4]

        losses = dict()
        loss_decode = self._decode_head_forward_train([feat], input_t, img_metas, depth_gt)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                [x], img_metas, depth_gt)
            losses.update(loss_aux)
        return losses

    def _decode_head_forward_train(self, x, t, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, t, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, t, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, t, img_metas, self.test_cfg)
        return seg_logits

    def right_pad_dims_to(self, x, t):
        padding_dims = x.ndim - t.ndim
        if padding_dims <= 0:
            return t
        return t.view(*t.shape, *((1,) * padding_dims))

    def decimal_to_bits(self, x):
        """ expects image tensor ranging from 0 to 1, outputs bit tensor ranging from -1 to 1 """
        device = x.device
        bits = self.BITS
        mask = 2 ** torch.arange(bits - 1, -1, -1, device=device)
        mask = rearrange(mask, 'd -> d 1 1')
        x = rearrange(x, 'b c h w -> b c 1 h w')

        bits = ((x & mask) != 0).float()
        bits = rearrange(bits, 'b c d h w -> b (c d) h w')
        bits = bits * 2 - 1
        return bits

    def bits_to_decimal(self, x):
        """ expects bits from -1 to 1, outputs image tensor from 0 to 1 """
        device = x.device
        bits = self.BITS
        # thresholding?
        x = (x > 0).int()
        mask = 2 ** torch.arange(bits - 1, -1, -1, device=device, dtype=torch.int32)

        mask = rearrange(mask, 'd -> d 1 1')
        x = rearrange(x, 'b (c d) h w -> b c d h w', d=bits)
        dec = reduce(x * mask, 'b c d h w -> b c h w', 'sum')
        return dec

    def gamma(self, t, ns=0.0002, ds=0.00025):
        return torch.cos(((t + ns) / (1 + ds)) * math.pi / 2) ** 2

    def _get_sampling_timesteps(self, batch, *, device):
        times = []
        for step in range(self.timesteps):
            t_now = 1 - step / self.timesteps
            t_next = max(1 - (step + 1 + self.time_difference) / self.timesteps, 0)
            time = torch.tensor([t_now, t_next], device=device)
            time = repeat(time, 't -> t b', b=batch)
            times.append(time)
        return times

    @torch.no_grad()
    def ddim_step(self, x_t, x_pred, t_now, t_next):
        alpha_now = self.gamma(t=t_now)
        alpha_next = self.gamma(t=t_next)
        x_pred = x_pred.clamp_(-self.bit_scale, self.bit_scale)
        eps = (1 / (1 - alpha_now).sqrt()) * (x_t - alpha_now.sqrt() * x_pred)
        x_next = alpha_next.sqrt() * x_pred + (1 - alpha_next).sqrt() * eps
        return x_next

    @torch.no_grad()
    def sample(self, x, img_metas):
        b, c, h, w, device = *x.shape, x.device
        time_pairs = self._get_sampling_timesteps(b, device=device)
        x = repeat(x, 'b c h w -> (r b) c h w', r=self.randsteps)
        depth_t = torch.randn((self.randsteps, 1, h, w), device=device)
        for times_now, times_next in time_pairs:
            feat = torch.cat([x, depth_t], dim=1)
            feat = self.down(feat)
            t = self.time_mlp(times_now)
            depth_pred = self._decode_head_forward_test([feat], t, img_metas=img_metas)
            depth_pred_ = ((depth_pred - self.min_depth) / (self.max_depth - self.min_depth))
            depth_pred_ = ((depth_pred_ * 2) - 1) * self.bit_scale
            times_now = self.right_pad_dims_to(feat, times_now)
            times_next = self.right_pad_dims_to(feat, times_next)
            sample_func = self.ddim_step if self.ddim else self.ddpm_step
            depth_t = sample_func(depth_t, depth_pred_, times_now, times_next)
        out = depth_pred.mean(dim=0, keepdim=True)
        return out



