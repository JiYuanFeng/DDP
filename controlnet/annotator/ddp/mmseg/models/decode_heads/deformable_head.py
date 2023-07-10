import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16
import torch
import warnings
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize

try:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
except ImportError:
    warnings.warn(
        '`MultiScaleDeformableAttention` in MMCV has been moved to '
        '`mmcv.ops.multi_scale_deform_attn`, please update your MMCV')
    from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention
from mmcv.cnn.bricks.transformer import (build_transformer_layer_sequence,
                                         build_positional_encoding)
from torch.nn.init import normal_


@HEADS.register_module()
class DeformableHead(BaseDecodeHead):
    """Implements the DeformableEncoder.
    Args:
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
    """
    
    def __init__(self,
                 num_feature_levels,
                 encoder,
                 positional_encoding,
                 **kwargs):
        
        super().__init__(input_transform='multiple_select', **kwargs)
    
        self.num_feature_levels = num_feature_levels
        self.encoder = build_transformer_layer_sequence(encoder)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.embed_dims = self.encoder.embed_dims
        
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
                                                 f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
                                                 f' and {num_feats}.'
        # self.level_embeds = nn.Parameter(
        #     torch.Tensor(self.num_feature_levels, self.embed_dims))
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        # normal_(self.level_embeds)
    
    @staticmethod
    def get_reference_points(spatial_shapes, device):
        """Get the reference points used in decoder.
        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None]
        return reference_points
    
    @auto_fp16()
    def forward(self, inputs):
        
        mlvl_feats = inputs[-self.num_feature_levels:]
        
        feat_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            mask = torch.zeros((bs, h, w), device=feat.device, requires_grad=False)
            pos_embed = self.positional_encoding(mask)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            feat = feat.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
        feat_flatten = torch.cat(feat_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        
        reference_points = self.get_reference_points(spatial_shapes, device=feat.device)
        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=None,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index)
        memory = memory.permute(1, 2, 0)
        memory = memory.reshape(bs, c, h, w).contiguous()
        out = self.conv_seg(memory)
        return out