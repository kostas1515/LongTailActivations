_base_ = [
    '../lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py'
]

model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="EQLv2",alpha=4.0))))

# optimizer = dict(type='SGD', lr=0.04, momentum=0.9, weight_decay=0.0001)

work_dir = './experiments/eqlv2_r50_4x4_1x'
