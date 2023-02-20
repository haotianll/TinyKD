_base_ = [
    '../tinyperson/faster_rcnn_r50_fpn_1x.py'
]
teacher_config = 'faster_rcnn_hr48_fpn_1x'

model = dict(
    type='KD_TwoStage',
    kd_config=[
        dict(
            loss_name='loss_kd_{}',
            loss_stages=5,
            kd_indices=[0, 1, 2, 3, 4],
            loss_kd=dict(
                type='KDLoss',
                lambda1=1.0,
                lambda2=1.0,
                alpha_fg=1.0,
                alpha_bg=0.5,
                attn_cfg=dict(patch_size=16, temp=0.05),
                extra=dict(
                    stu_channel=[256, 256, 256, 256, 256],
                    tea_channel=[256, 256, 256, 256, 256],
                )
            )
        ),
    ],
    teacher_config=f'configs_tiny/tinyperson/{teacher_config}.py',
    teacher_ckpt=f'work_dirs/{teacher_config}/epoch_12.pth',
    teacher_inherit=['rpn_head.', 'roi_head.'],
)
