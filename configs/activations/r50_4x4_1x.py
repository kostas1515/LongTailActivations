_base_ = [
    '../lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py'
]

# data = dict(train=dict(oversample_thr=0.0))
data = dict(train=dict(oversample_thr=0.0),samples_per_gpu=4)

model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="Icloglog",activation='sigmoid'),
                                         init_cfg = dict(type='Constant',val=0.001, bias=-5, override=dict(name='fc_cls')))))

# work_dir='./experiments/r50_4x4_1x_clone/'
work_dir='./experiments/test/'

# get_stats=1

# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=1000,
#     warmup_ratio=0.001,
#     step=[8, 11])
