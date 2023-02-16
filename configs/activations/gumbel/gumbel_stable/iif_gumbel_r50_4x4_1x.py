_base_ = [
    '../../../lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py'
]

data = dict(train=dict(oversample_thr=0.0))

# model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="Icloglog",activation='normal'),
#                                          init_cfg = dict(type='Constant',val=0.01, bias=-3.45, override=dict(name='fc_cls')))))
# use temp=2 for iif, 1.5 for adaptive

model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="GumbelCrossEntropyLoss",temperature=2,use_iif='base10_obj'),
                                         init_cfg = dict(type='Constant',val=0.001, bias=-2, override=dict(name='fc_cls')))))

work_dir='./experiments/activations/gumbel_stable/iif_gumbel_r50_4x4_1x/'
# work_dir='./experiments/test/'

# resume_from='./experiments/activations/gumbel_stable/iif_gumbel_r50_4x4_1x/epoch_3.pth'

# lr_config = dict(warmup_iters=1000)


# get_stats=1