_base_ = [
    '../aitod/aitod_fcos_r50_1x.py'
]
teacher_config = 'aitod_fcos_hr48_1x'

model = dict(
    type='KD_SingleStage',
    kd_config=[
        dict(
            loss_name='loss_kd_{}',
            loss_stages=5,
            kd_indices=[0, 1, 2, 3, 4],
            loss_kd=dict(
                type='KDLoss',
                lambda1=0.5,
                lambda2=0.5,
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
    teacher_config=f'configs_tiny/aitod/{teacher_config}.py',
    teacher_ckpt=f'work_dirs/{teacher_config}/epoch_12.pth',
    teacher_inherit=['bbox_head.']
)