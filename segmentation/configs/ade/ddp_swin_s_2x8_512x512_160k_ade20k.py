_base_ = [
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/' \
                  'swin_small_patch4_window7_224_20220317-7ba6d6dd.pth'  # noqa
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='DDP',
    timesteps=3,
    bit_scale=0.01,
    accumulation=True,
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        pretrain_img_size=224,
        in_channels=3,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg),
    neck=[
        dict(
            type='FPN',
            in_channels=[96, 192, 384, 768],
            out_channels=256,
            act_cfg=None,
            norm_cfg=dict(type='GN', num_groups=32),
            num_outs=4),
        dict(
            type='MultiStageMerging',
            in_channels=[256, 256, 256, 256],
            out_channels=256,
            kernel_size=1,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=None)
    ],
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=0,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4)),
    decode_head=dict(
        type='DeformableHeadWithTime',
        in_channels=[256],
        channels=256,
        in_index=[0],
        dropout_ratio=0.,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        num_feature_levels=1,
        encoder=dict(
            type='DetrTransformerEncoder',
            num_layers=6,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                use_time_mlp=True,
                attn_cfgs=dict(
                    type='MultiScaleDeformableAttention',
                    embed_dims=256,
                    num_levels=1,
                    num_heads=8,
                    dropout=0.),
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=256,
                    feedforward_channels=1024,
                    ffn_drop=0.,
                    act_cfg=dict(type='GELU')),
                operation_order=('self_attn', 'norm', 'ffn', 'norm'))
        ),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
)
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=1.)
        }))
lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
evaluation = dict(interval=16000, metric='mIoU', save_best='mIoU')
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
