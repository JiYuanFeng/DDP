import math

import torch
from einops import rearrange, repeat
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from torch import nn
from torch.special import expm1

from mmdet3d.models import FUSIONMODELS
from .bevfusion import BEVFusion

__all__ = ["DDP"]

from ...ops.norm import resize


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def beta_linear_log_snr(t):
    return -torch.log(expm1(1e-4 + 10 * (t ** 2)))


def alpha_cosine_log_snr(t, ns=0.0002, ds=0.00025):
    return -log((torch.cos((t + ns) / (1 + ds) * math.pi * 0.5) ** -2) - 1, eps=1e-5)


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


@FUSIONMODELS.register_module()
class DDP(BEVFusion):
    def __init__(
            self,
            bit_scale=1,
            timesteps=1,
            randsteps=1,
            time_difference=1,
            learned_sinusoidal_dim=16,
            sample_range=(0, 0.999),
            noise_schedule='cosine',
            diffusion='ddim',
            threshold=0.5,
            feat_channels=512,
            tmp_channels=256,
            seq_length=1,
            **kwargs,
    ) -> None:
        super(DDP, self).__init__(**kwargs)

        self.bit_scale = bit_scale
        self.timesteps = timesteps
        self.randsteps = randsteps
        self.diffusion = diffusion
        self.time_difference = time_difference
        self.sample_range = sample_range
        self.num_classes = 6
        self.tmp_channels = tmp_channels
        self.feat_channels = feat_channels
        self.seq_length = seq_length
        self.embedding_table = nn.Embedding(self.num_classes + 1, self.tmp_channels)
        self.threshold = threshold

        print("sample range:", sample_range)
        print("timesteps: {}, randsteps: {}".format(timesteps, randsteps))

        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')
        self.transform = ConvModule(self.tmp_channels * self.seq_length + self.feat_channels, self.tmp_channels, 1,
                                    act_cfg=None)
        # self.transform = ConvModule(512, tmp_channels, 1, act_cfg=None)
        # time embeddings
        time_dim = self.tmp_channels * 4  # 1024
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(  # [2,]
            sinu_pos_emb,  # [2, 17]
            nn.Linear(fourier_dim, time_dim),  # [2, 1024]
            nn.GELU(),
            nn.Linear(time_dim, time_dim)  # [2, 1024]
        )

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def right_pad_dims_to(self, x, t, offset=0):
        padding_dims = x.ndim - t.ndim - offset
        if padding_dims <= 0:
            return t
        return t.view(*t.shape, *((1,) * padding_dims))

    def _get_sampling_timesteps(self, batch, *, device):
        times = []
        for step in range(self.timesteps):
            t_now = 1 - step / self.timesteps
            t_next = max(1 - (step + 1 + self.time_difference) / self.timesteps, 0)
            time = torch.tensor([t_now, t_next], device=device)
            time = repeat(time, 't -> t b', b=batch)
            times.append(time)
        return times

    @force_fp32(apply_to=("img"))
    def forward_single(
            self,
            img,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            metas,
            gt_masks_bev=None,
            gt_bboxes_3d=None,
            gt_labels_3d=None,
            motion_instance=None,
            motion_segmentation=None,
            **kwargs,
    ):
        features = []
        for sensor in (
                self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                feature = self.extract_camera_features(
                    img,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                )
            elif sensor == "lidar":
                # feature = self.extract_lidar_features(points)
                raise ValueError("lidar sensor not supported")
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature)

        if not self.training:
            # avoid OOM
            features = features[::-1]

        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        batch_size = x.shape[0]  # (bs, 80,128,128)
        x = self.decoder["backbone"](x)  # a list containing 3 features
        x = self.decoder["neck"](x)
        if not isinstance(x, list):
            x = [x]  # x with shape (1,256,128,128)
        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    batch, c, h, w, device, = *x[0].shape, x[0].device  # 4,256,128,128
                    # shape of gt_mask_bev: (bs, 6, 200,200) (single frame)
                    # shape of gt_mask_bev : (bs,seq_length,6,200,200) (with future frame)
                    _, seq_len, _, h_bev, w_bev = gt_masks_bev.shape
                    multi_factor = (torch.arange(self.num_classes, device=device) + 1).view(1, self.num_classes, 1,
                                                                                            1)  # (1,6,1,1)
                    multi_factor = multi_factor.unsqueeze(2)  # (1,6,1,1,1)
                    multi_factor = multi_factor.repeat(batch, 1, seq_len, 1, 1)  # (bs,num_classes,seq_length,1,1)
                    # switch axis
                    multi_factor = multi_factor.permute(0, 2, 1, 3, 4)  # (bs,seq_length,num_classes,1,1)
                    multi_factor = multi_factor.reshape(batch, seq_len * self.num_classes, 1, 1)
                    gt_masks_bev = gt_masks_bev.view(batch, seq_len * self.num_classes, h_bev, w_bev)
                    gt_down = gt_masks_bev * multi_factor
                    gt_down = resize(gt_down.float(), size=(h, w), mode="nearest").to(
                        gt_masks_bev.dtype)  # -> (bs,6*seq_len,128,128)
                    # feed in to nn.embedding -> (bs,6*seq_len,128,128,256)
                    # gt_down = self.embedding_table(gt_down).mean(dim=1).permute(0, 3, 1, 2)  # class embedding and averate
                    # the shape of self.embedding_table(gt_down) is (bs,num_class*seq_length,h,w,tmp_channels)
                    gt_down = self.embedding_table(gt_down).view(batch, seq_len, self.num_classes, h, w,
                                                                 self.tmp_channels)
                    # (bs,seq_len,num_class,128,128,tmp_channels)
                    # the shape of gt_down.mean(dim=2) is (bs,seq_len,128,128,tmp_channels)
                    gt_down = gt_down.mean(dim=2).permute(0, 4, 1, 2, 3)  # (bs,tmp_channels,seq_len,128,128)
                    # (bs,tmp_channels*seq_len,128,128)
                    # after embedding, its shape becomes (4,256,128,128)
                    gt_down = (torch.sigmoid(gt_down) * 2 - 1) * self.bit_scale
                    gt_down = gt_down.reshape(batch, self.tmp_channels * seq_len, h, w).contiguous()
                    # sample timesteps

                    times = torch.zeros((batch,), device=device) \
                        .float().uniform_(self.sample_range[0], self.sample_range[1])
                    # random noise

                    noise = torch.randn_like(gt_down)
                    noise_level = self.log_snr(times)
                    padded_noise_level = self.right_pad_dims_to(img, noise_level, offset=1)
                    alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)
                    noised_gt = alpha * gt_down + sigma * noise

                    # conditional input
                    feat = torch.cat([x[0], noised_gt], dim=1)
                    # shape of x is (bs,feat_channels,h,w)
                    # shape of noised_gt is (bs,tmp_channels,h,w)
                    feat = self.transform(feat)  # encode the concat feature
                    # shape of feat after transform is (bs,256,128,128)
                    input_times = self.time_mlp(noise_level)
                    # shape of input_times is (bs,1024)
                    # In the original paper, the author use x_t and t to predict the noise of X_0!! not the noise of x_t-1
                    # so here we also predict the GROUND_TRUTH --> we can then calculate the noise of X_0 (reparameterization trick)
                    # the reason why we do this is that we don't use MSEloss to predict the noise directly.
                    losses = head([feat], input_times, gt_masks_bev)
                    # one thing is different here, we use the decoder to predict the gt_masks
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":

                    if self.diffusion == "ddim":
                        logits = self.ddim_sample(x, head)
                    elif self.diffusion == "ddpm":
                        logits = self.ddpm_sample(x, head)
                    else:
                        raise ValueError(f"unsupported diffusion: {self.diffusion}")
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs

    @torch.no_grad()
    def ddim_sample(self, x, head):
        b, c, h, w, device = *x[0].shape, x[0].device
        time_pairs = self._get_sampling_timesteps(b, device=device)

        x = repeat(x[0], 'b c h w -> (r b) c h w', r=self.randsteps)
        outs = []
        mask_t = torch.randn((self.randsteps, self.seq_length*self.tmp_channels, h, w), device=device)
        for idx, (times_now, times_next) in enumerate(time_pairs):
            feat = torch.cat([x, mask_t], dim=1)
            br = x.shape[0]
            feat = self.transform(feat)

            log_snr = self.log_snr(times_now)
            log_snr_next = self.log_snr(times_next)

            padded_log_snr = self.right_pad_dims_to(mask_t, log_snr)
            padded_log_snr_next = self.right_pad_dims_to(mask_t, log_snr_next)
            alpha, sigma = log_snr_to_alpha_sigma(padded_log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(padded_log_snr_next)

            input_times = self.time_mlp(log_snr)
            mask_logit = head([feat], input_times, target=None)
            bs, seq_len, num_classes, h_bev, w_bev = mask_logit.shape
            assert bs == br
            # (bs, len(self.classes), self.seq_length,bev_h, bev_w)
            mask_pred = (mask_logit > self.threshold)

            multi_factor = (torch.arange(self.num_classes, device=device) + 1).view(1, self.num_classes, 1, 1)
            multi_factor = multi_factor.unsqueeze(2)
            multi_factor = multi_factor.repeat(br, 1, self.seq_length, 1, 1)
            multi_factor = multi_factor.permute(0, 2, 1, 3, 4)  # (bs,seq_length,num_classes,1,1)
            multi_factor = multi_factor.reshape(br, self.seq_length * self.num_classes, 1, 1)

            mask_pred = mask_pred.view(br, self.seq_length * self.num_classes, h_bev, w_bev)
            mask_pred = mask_pred * multi_factor
            mask_pred = resize(mask_pred.float(), size=(h, w), mode="nearest").to(torch.int64)
            mask_pred = self.embedding_table(mask_pred).view(br, seq_len, self.num_classes, h, w,
                                                             self.tmp_channels)
            mask_pred = mask_pred.mean(dim=2).permute(0, 4, 1, 2, 3)

            # mask_pred = self.embedding_table(mask_pred).mean(dim=1).permute(0, 3, 1, 2)
            mask_pred = (torch.sigmoid(mask_pred) * 2 - 1) * self.bit_scale
            mask_pred = mask_pred.reshape(br, seq_len * self.tmp_channels, h, w).contiguous()

            pred_noise = (mask_t - alpha * mask_pred) / sigma.clamp(min=1e-8)
            mask_t = mask_pred * alpha_next + pred_noise * sigma_next
            outs.append(mask_logit)
        outs = torch.cat(outs, dim=0)
        logit = outs.mean(dim=0, keepdim=True)
        return logit

    @torch.no_grad()
    def ddpm_sample(self, x, head):
        raise NotImplementedError("DDPM sampling not implemented for multi frames")
        b, c, h, w, device = *x[0].shape, x[0].device
        time_pairs = self._get_sampling_timesteps(b, device=device)

        x = repeat(x, 'b c h w -> (r b) c h w', r=self.randsteps)
        mask_t = torch.randn((self.randsteps, self.num_classes, h, w), device=device)
        for times_now, times_next in time_pairs:
            feat = torch.cat([x[0], mask_t], dim=1)
            feat = self.transform(feat)

            log_snr = self.log_snr(times_now)
            log_snr_next = self.log_snr(times_next)

            padded_log_snr = self.right_pad_dims_to(mask_t, log_snr)
            padded_log_snr_next = self.right_pad_dims_to(mask_t, log_snr_next)
            alpha, sigma = log_snr_to_alpha_sigma(padded_log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(padded_log_snr_next)

            input_times = self.time_mlp(log_snr)
            mask_logit = head([feat], input_times)  # [bs, 150, ]
            # mask_logits is the output of sigmoid, use threshold=0.5 to get mask
            mask_pred = (mask_logit > self.threshold).float()
            multi_factor = (torch.arange(self.num_classes, device=device) + 1).view(1, self.num_classes, 1, 1)
            mask_pred = mask_pred * multi_factor
            mask_pred = self.embedding_table(mask_pred).mean(dim=1).permute(0, 3, 1, 2)
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
