_base_ = [
    '../_base_/datasets/tinyperson.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
evaluation = dict(interval=12)
checkpoint_config = dict(interval=12)

# optimizer
optimizer = dict(lr=0.01, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',  # FCOS tricks
    # warmup='constant',
    warmup_iters=500,  # ORIGINAL
    warmup_ratio=1.0 / 3,
    step=[8, 11]
)
runner = dict(type='EpochBasedRunner', max_epochs=12)

# model settings
model = dict(
    type='FCOS',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,  # NEW
        # start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=1,  # NEW
        # num_classes=80
        regress_ranges=((-1, 16), (16, 32), (32, 64), (64, 128), (128, 1e8)),  # NEW
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[4, 8, 16, 32, 64],  # NEW
        # strides=[8, 16, 32, 64, 128],
        norm_cfg=None,  # NEW
        norm_on_bbox=True,  # FCOS tricks
        centerness_on_reg=True,  # FCOS tricks
        dcn_on_last_conv=False,  # FCOS tricks
        center_sampling=True,  # FCOS tricks
        conv_bias=True,  # FCOS tricks
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),  # NEW
        # loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,  # NEW
        # nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),  # FCOS tricks
        # nms=dict(type='nms', iou_threshold=0.5),
        # max_per_img=100
        max_per_img=1000  # NEW
    ))
