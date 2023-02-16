_base_ = '../faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
    roi_head=dict(
        type='StandardLocalRoIHead'))

data = dict(
    samples_per_gpu=1)

work_dir = './experiments/test'