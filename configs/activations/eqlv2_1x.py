_base_ = ['./r50_4x4_1x.py']

model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="EQLv2"))))

work_dir = './experiments/eqlv2_1x'