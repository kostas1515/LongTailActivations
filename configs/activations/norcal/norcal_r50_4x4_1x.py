_base_ = [
    '../../lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py'
]

# data = dict(train=dict(oversample_thr=0.0))

# model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="Icloglog",activation='normal'),
#                                          init_cfg = dict(type='Constant',val=0.01, bias=-3.45, override=dict(name='fc_cls')))))

# model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="Icloglog",activation='norcal',use_sigmoid=False))))

model = dict(roi_head=dict(bbox_head=dict(type='Shared2FCLocalBBoxHead',dim=4)))

# work_dir='./experiments/gumbel_debug/'
work_dir='./experiments/test/'

# get_stats=1