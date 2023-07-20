_base_ = [
    '../../lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py'
]

data = dict(train=dict(oversample_thr=0.0))

# model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="Icloglog",activation='normal'),
#                                          init_cfg = dict(type='Constant',val=0.01, bias=-3.45, override=dict(name='fc_cls')))))

model = dict(roi_head=dict(
    bbox_head=dict(type='Shared2FCLocalBBoxHead',loss_cls=dict(type="IIFLoss",variant='base10_obj'),
                                          cls_predictor_cfg=dict(type='NormedLinear', tempearture=8),
                                         init_cfg = dict(type='Constant',val=0.001, bias=0.0, override=dict(name='fc_cls'))),
    mask_head=dict(predictor_cfg=dict(type='NormedConv2d', tempearture=20))),
    test_cfg=dict(rcnn=dict(
            score_thr=0.0001,
            nms=dict(type='nms', iou_threshold=0.3),
            max_per_img=300,
            mask_thr_binary=0.4)))

work_dir='./experiments/iif/iif_base10_obj_r50_4x4_1x/'
# work_dir='./experiments/test/'