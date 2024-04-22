samples_per_gpu = 2
img_scale = (1333, 800)
num_stages = 6
num_query = 100
model = dict(
    type='QueryBased',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='ChannelMapping',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=4),
    rpn_head=dict(
        type='InitialQueryGenerator', 
        num_query=num_query, content_dim=256),
    roi_head=dict(
        type='AdaMixerDecoder',
        featmap_strides=[4, 8, 16, 32],
        num_stages=num_stages,
        kernel_sizes=[16]*num_stages,
        kernel_groups=[1, 1, 2, 2, 3, 3],
        mix_out_proj_ranks=[8]*num_stages,
        stage_loss_weights=[1]*num_stages,
        feat_channels=256,
        content_dim=256,
        in_points=64,
        n_groups=2,
        out_points=128,
        bbox_head=[
            dict(
                type='AdaMixerDecoderStage',
                num_classes=80,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=1,
                feedforward_channels=2048,
                dropout=0.0,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=False,
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.5, 0.5, 1.0, 1.0]))
            for _ in range(num_stages) ]),
    train_cfg=dict(
        rpn=None,
        rcnn=[
            dict(
                assigner=dict(
                    type='HungarianAssigner',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                    iou_cost=dict(type='IoUCost', iou_mode='giou',
                                  weight=2.0)),
                sampler=dict(type='PseudoSampler'),
                pos_weight=1)
            for _ in range(num_stages)]),
    test_cfg=dict(rpn=None, rcnn=dict(max_per_img=num_query)))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                           (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                           (736, 1333), (768, 1333), (800, 1333)],
                multiscale_mode='value',
                keep_ratio=True)
        ],
            [
            dict(
                type='Resize',
                img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                multiscale_mode='value',
                keep_ratio=True),
            dict(
                type='RandomCrop',
                crop_type='absolute_range',
                crop_size=(384, 600),
                allow_negative_crop=True),
            dict(
                type='Resize',
                img_scale=[(480, 1333), (512, 1333), (544, 1333),
                           (576, 1333), (608, 1333), (640, 1333),
                           (672, 1333), (704, 1333), (736, 1333),
                           (768, 1333), (800, 1333)],
                multiscale_mode='value',
                override=True,
                keep_ratio=True)
        ]]),
            dict(type='Normalize',
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375],
                 to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize',
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375],
                 to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])]
        )]

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=2*samples_per_gpu,
    train=dict(
        type='CocoDataset',
        ann_file=
        'coco/annotations/instances_train2017.json',
        img_prefix='coco/train2017/',
        pipeline=train_pipeline),
    val=dict(
        type='CocoDataset',
        ann_file='coco/annotations/instances_val2017.json',
        img_prefix='coco/val2017/',
        pipeline=test_pipeline),
    test=dict(
        type='CocoDataset',
        ann_file='coco/annotations/image_info_test-dev2017.json',
        img_prefix='coco/test2017/',
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox')
optimizer = dict(type='AdamW', lr=2.5e-05, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=1.0, norm_type=2))
lr_config = dict(policy='step',
                 warmup='linear', warmup_iters=500, warmup_ratio=0.001, 
                 step=[24, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)
checkpoint_config = dict(interval=1)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
log_level = 'INFO'
load_from = None
resume_from = None
find_unused_parameters = True