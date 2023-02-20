dataset_type = 'TinyPersonDataset'
data_root = '../data/tinyperson/tiny_set/'
classes = ('person',)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadSubImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='CroppedTilesFlipAug',
        tile_shape=(640, 512),
        tile_overlap=(100, 100),
        scale_factor=[1.0],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'mini_annotations/tiny_set_train_sw640_sh512_all_erase.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'mini_annotations/tiny_set_test_all.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'mini_annotations/tiny_set_test_all.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline
    )
)
evaluation = dict(interval=12, metric='bbox', iou_thrs=[0.25, 0.5, 0.75], proposal_nums=[1000], eval_mr=True)
