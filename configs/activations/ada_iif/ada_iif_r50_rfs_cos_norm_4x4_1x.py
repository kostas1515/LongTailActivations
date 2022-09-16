_base_ = [
    '../../lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py'
]

# data = dict(train=dict(oversample_thr=0.0))

model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="ADAIIFLoss"),
                                         cls_predictor_cfg=dict(type='NormedLinear', tempearture=8),
                                         ),
                           mask_head=dict(predictor_cfg=dict(type='NormedConv2d', tempearture=20))))

work_dir='./experiments/ada_iif/ada_iif_r50_rfs_cos_norm_4x4_1x/'
# work_dir='./experiments/test/'
