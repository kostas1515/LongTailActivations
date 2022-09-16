_base_ = [
    '../lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py'
]

# data = dict(train=dict(oversample_thr=0.0))

# model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="Icloglog",activation='normal'),
#                                          init_cfg = dict(type='Constant',val=0.01, bias=-3.45, override=dict(name='fc_cls')))))

model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="LDAM"),
                                         init_cfg = dict(type='Constant',val=0.001, bias=0.0, override=dict(name='fc_cls')))))

work_dir='./experiments/ldam/ldam_r50_rfs_4x4_1x/'
# work_dir='./experiments/test/'

# load_from = './experiments/baselines/r50_rfs_4x4_1x_softmax/epoch_12.pth'
# # Train which part, 0 for all, 1 for cls, 2 for bbox_head
# selectp = 1
