_base_ = [
    './retinanet_r50_fpn_1x.py'
]
model = dict(
    backbone=dict(
        _delete_=True,
        type='HRNet',
        norm_cfg=dict(type='BN', requires_grad=True),
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,)),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384))),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://msra/hrnetv2_w48')),
    neck=dict(
        _delete_=True,
        type='HRFPN',
        in_channels=[48, 96, 192, 384],
        out_channels=256)
)
