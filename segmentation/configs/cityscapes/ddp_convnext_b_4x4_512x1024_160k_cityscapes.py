_base_ = [
    '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
custom_imports = dict(imports='mmcls.models', allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/' \
                  'downstream/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth'  # noqa
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='DDP',
    timesteps=3,
    bit_scale=0.01,
    pretrained=None,
    backbone=dict(
        type='mmcls.ConvNeXt',
        arch='base',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    neck=[
        dict(
            type='FPN',
            in_channels=[128, 256, 512, 1024],
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
        num_classes=19,
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
        num_classes=19,
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
    samples_per_gpu=4,
    workers_per_gpu=4,
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
find_unused_parameters = True
evaluation = dict(interval=16000, metric='mIoU', save_best='mIoU')
