_base_ = [
    '../../lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py'
]


model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="Icloglog",activation='normal'),
                                         cls_predictor_cfg=dict(type='NormedLinear', tempearture=20,init_bias=-3.2),
                                         init_cfg = dict(type='Constant',val=0.001, bias=-3.2, override=dict(name='fc_cls')))
                          ,
                          mask_head=dict(predictor_cfg=dict(type='NormedConv2d', tempearture=20))))

work_dir='./experiments/normal_r50_rfs_cos_norm_4x4_1x_20/'
# work_dir='./experiments/test'