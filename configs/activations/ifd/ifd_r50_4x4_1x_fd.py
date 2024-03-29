_base_ = [
    '../../lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py'
]

data = dict(train=dict(oversample_thr=0.0))

# model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="Icloglog",activation='normal'),
#                                          init_cfg = dict(type='Constant',val=0.01, bias=-3.45, override=dict(name='fc_cls')))))

model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="IFDLoss",variant='fd'),
                                         init_cfg = dict(type='Constant',val=0.001, bias=0.0, override=dict(name='fc_cls')))))

work_dir='./experiments/iif/ifd_r50_4x4_1x_fd/'
resume_from='./experiments/iif/ifd_r50_4x4_1x_fd/latest.pth'
# work_dir='./experiments/test/'
