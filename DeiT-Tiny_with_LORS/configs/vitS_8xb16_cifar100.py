# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformerWlors',
        arch='deit-tiny',
        img_size=56,
        patch_size=4,
        attn_kernel_sizes=12*[32],
        attn_kernel_groups=[1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 6],
        ffn_kernel_sizes=12*[32],
        ffn_kernel_groups=[6, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        init_cfg=[dict(type='Kaiming',
                       layer='Conv2d',
                       mode='fan_in',
                       nonlinearity='linear')]),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=100,
        in_channels=192,
        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1)),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=.02),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)],
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=100, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=100, prob=0.5)])
    )

# dataset settings
dataset_type = 'CIFAR100'
img_norm_cfg = dict(
    mean=[129.304, 124.070, 112.434],
    std=[68.170, 65.392, 70.418],
    to_rgb=False)
train_pipeline = [
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data_root='/data/'
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix=data_root+'/cifar100',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix=data_root+'/cifar100',
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_prefix=data_root+'/cifar100',
        pipeline=test_pipeline,
        test_mode=True))

paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys={
        '.cls_token': dict(decay_mult=0.0),
        '.pos_embed': dict(decay_mult=0.0)
    })
# for batch in each gpu is 128, 8 gpu
# lr = 5e-4 * 128 * 8 / 512 = 0.001
optimizer = dict(
    type='AdamW',
    lr=5e-4 * 128 * 8 / 512,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=paramwise_cfg)
optimizer_config = dict(grad_clip=dict(max_norm=5.0))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=1e-2,
    warmup='linear',
    warmup_ratio=1e-3,
    warmup_iters=20,
    warmup_by_epoch=True)

runner = dict(type='EpochBasedRunner', max_epochs=300)
# checkpoint saving
checkpoint_config = dict(interval=20)
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
