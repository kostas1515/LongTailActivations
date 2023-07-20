_base_ = [
    '../lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py'
]

model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="EQL", use_sigmoid=True, lambda_=0.0011, version="v1",use_classif='sigmoid'))))

work_dir = './experiments/eql_r50_4x4_1x_normal'
