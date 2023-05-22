import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.core import add_prefix
from mmseg.ops import resize
from torch.special import expm1
from einops import rearrange, reduce, repeat
from mmcv.cnn import ConvModule
import math
from PIL import Image
import numpy as np
import os

from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def beta_linear_log_snr(t):
    return -torch.log(expm1(1e-4 + 10 * (t ** 2)))


def alpha_cosine_log_snr(t, ns=0.0002, ds=0.00025):
    return -log((torch.cos((t + ns) / (1 + ds) * math.pi * 0.5) ** -2) - 1,
                eps=1e-5)  # not sure if this accounts for beta being clipped to 0.999 in discrete version


def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))


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


@SEGMENTORS.register_module()
class DDP(EncoderDecoder):
    """Encoder Decoder segmentors.
    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 bit_scale=0.1,
                 timesteps=1,
                 randsteps=1,
                 time_difference=1,
                 learned_sinusoidal_dim=16,
                 sample_range=(0, 0.999),
                 noise_schedule='cosine',
                 diffusion='ddim',
                 **kwargs):
        super(DDP, self).__init__(**kwargs)

        self.bit_scale = bit_scale
        self.timesteps = timesteps
        self.randsteps = randsteps
        self.diffusion = diffusion
        self.time_difference = time_difference
        self.sample_range = sample_range
        self.use_gt = False
        self.embedding_table = nn.Embedding(self.num_classes + 1, self.decode_head.in_channels[0])

        print(f" timesteps: {timesteps},"
              f" randsteps: {randsteps},"
              f" sample_range: {sample_range},"
              f" diffusion: {diffusion}")

        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        self.transform = ConvModule(
            self.decode_head.in_channels[0] * 2,
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

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)[0]
        if self.diffusion == "ddim":
            out = self.ddim_sample(x, img_metas)
        elif self.diffusion == 'ddpm':
            out = self.ddpm_sample(x, img_metas)
        else:
            raise NotImplementedError
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def forward_train(self, img, img_metas, gt_semantic_seg):
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
        gt_down = resize(gt_semantic_seg.float(), size=(h, w), mode="nearest")
        gt_down = gt_down.to(gt_semantic_seg.dtype)
        gt_down[gt_down == 255] = self.num_classes

        gt_down = self.embedding_table(gt_down).squeeze(1).permute(0, 3, 1, 2)
        gt_down = (torch.sigmoid(gt_down) * 2 - 1) * self.bit_scale

        # sample time
        times = torch.zeros((batch,), device=device).float().uniform_(self.sample_range[0],
                                                                      self.sample_range[1])  # [bs]

        # random noise
        noise = torch.randn_like(gt_down)
        noise_level = self.log_snr(times)
        padded_noise_level = self.right_pad_dims_to(img, noise_level)
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)
        noised_gt = alpha * gt_down + sigma * noise

        # conditional input
        feat = torch.cat([x, noised_gt], dim=1)
        feat = self.transform(feat)

        losses = dict()
        input_times = self.time_mlp(noise_level)
        loss_decode = self._decode_head_forward_train([feat], input_times, img_metas, gt_semantic_seg)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                [x], img_metas, gt_semantic_seg)
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

    def _get_sampling_timesteps(self, batch, *, device):
        times = []
        for step in range(self.timesteps):
            t_now = 1 - (step / self.timesteps) * (1 - self.sample_range[0])
            t_next = max(1 - (step + 1 + self.time_difference) / self.timesteps * (1 - self.sample_range[0]),
                         self.sample_range[0])
            time = torch.tensor([t_now, t_next], device=device)
            time = repeat(time, 't -> t b', b=batch)
            times.append(time)
        return times

    @torch.no_grad()
    def ddim_sample(self, x, img_metas):
        b, c, h, w, device = *x.shape, x.device
        time_pairs = self._get_sampling_timesteps(b, device=device)
        x = repeat(x, 'b c h w -> (r b) c h w', r=self.randsteps)
        mask_t = torch.randn((self.randsteps, self.decode_head.in_channels[0], h, w), device=device)
        for idx, (times_now, times_next) in enumerate(time_pairs):
            feat = torch.cat([x, mask_t], dim=1)
            feat = self.transform(feat)
            log_snr = self.log_snr(times_now)
            log_snr_next = self.log_snr(times_next)

            padded_log_snr = self.right_pad_dims_to(mask_t, log_snr)
            padded_log_snr_next = self.right_pad_dims_to(mask_t, log_snr_next)
            alpha, sigma = log_snr_to_alpha_sigma(padded_log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(padded_log_snr_next)

            input_times = self.time_mlp(log_snr)
            mask_logit = self._decode_head_forward_test([feat], input_times, img_metas=img_metas)  # [bs, 150, ]
            mask_pred = torch.argmax(mask_logit, dim=1)
            mask_pred = self.embedding_table(mask_pred).permute(0, 3, 1, 2)
            mask_pred = (torch.sigmoid(mask_pred) * 2 - 1) * self.bit_scale
            pred_noise = (mask_t - alpha * mask_pred) / sigma.clamp(min=1e-8)
            mask_t = mask_pred * alpha_next + pred_noise * sigma_next

        logit = mask_logit.mean(dim=0, keepdim=True)
        return logit

    @torch.no_grad()
    def ddpm_sample(self, x, img_metas):
        b, c, h, w, device = *x.shape, x.device
        time_pairs = self._get_sampling_timesteps(b, device=device)

        x = repeat(x, 'b c h w -> (r b) c h w', r=self.randsteps)
        mask_t = torch.randn((self.randsteps, 256, h, w), device=device)
        for times_now, times_next in time_pairs:
            feat = torch.cat([x, mask_t], dim=1)
            feat = self.transform(feat)

            log_snr = self.log_snr(times_now)
            log_snr_next = self.log_snr(times_next)

            padded_log_snr = self.right_pad_dims_to(mask_t, log_snr)
            padded_log_snr_next = self.right_pad_dims_to(mask_t, log_snr_next)
            alpha, sigma = log_snr_to_alpha_sigma(padded_log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(padded_log_snr_next)

            input_times = self.time_mlp(log_snr)
            mask_logit = self._decode_head_forward_test([feat], input_times, img_metas=img_metas)  # [bs, 150, ]
            mask_pred = torch.argmax(mask_logit, dim=1)
            mask_pred = self.embedding_table(mask_pred).permute(0, 3, 1, 2)
            mask_pred = (torch.sigmoid(mask_pred) * 2 - 1) * self.bit_scale

            c = -expm1(log_snr - log_snr_next)
            mean = alpha_next * (mask_t * (1 - c) / alpha + c * mask_pred)
            variance = (sigma_next ** 2) * c
            log_variance = log(variance)
            noise = torch.where(
                rearrange(times_next > 0, 'b -> b 1 1 1'),
                torch.randn_like(mask_t),
                torch.zeros_like(mask_t)
            )
            mask_t = mean + (0.5 * log_variance).exp() * noise

        logit = mask_logit.mean(dim=0, keepdim=True)
        return logit
