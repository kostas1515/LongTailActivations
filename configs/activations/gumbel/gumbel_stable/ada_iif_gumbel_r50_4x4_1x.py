_base_ = [
    '../../../lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py'
]

data = dict(train=dict(oversample_thr=0.0),samples_per_gpu=4)

# model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="Icloglog",activation='normal'),
#                                          init_cfg = dict(type='Constant',val=0.01, bias=-3.45, override=dict(name='fc_cls')))))
# use temp=2 for iif

model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="GumbelCrossEntropyLoss",temperature=1.5,use_iif='adaptive'),
                                         init_cfg = dict(type='Constant',val=0.001, bias=-2, override=dict(name='fc_cls')))))

work_dir='./experiments/activations/gumbel_stable/ada_iif_gumbel_r50_4x4_1x/'
# work_dir='./experiments/test/'

# resume_from = './experiments/activations/gumbel_stable/ada_iif_gumbel_r50_4x4_1x/epoch_6.pth'


# get_stats=1