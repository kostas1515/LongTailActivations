_base_ = [
    '../../lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py'
]

# data = dict(train=dict(oversample_thr=0.0))

# model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="Icloglog",activation='normal'),
#                                          init_cfg = dict(type='Constant',val=0.01, bias=-3.45, override=dict(name='fc_cls')))))

model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="IIFLoss",variant='raw'),
                                         init_cfg = dict(type='Constant',val=0.001, bias=0.0, override=dict(name='fc_cls')))))

evaluation = dict(interval=1, metric=['bbox', 'segm'])

# learning policy
lr_config = dict(
    policy='step',
   warmup='linear',
   warmup_iters=500,
   warmup_ratio=0.001,
    step=[3, 5])

runner = dict(type='EpochBasedRunner', max_epochs=6)

work_dir='./experiments/iif/iif_decoup_rfs_r50_4x4_1x_6e/'
load_from='./experiments/baselines/r50_4x4_1x_softmax/latest.pth'
selectp=1
# work_dir='./experiments/test/'
