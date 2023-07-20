# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmdet.models.builder import HEADS
from mmcv.runner import force_fp32
from mmdet.models.roi_heads.bbox_heads import ConvFCBBoxHead
import torch.nn.functional as F
from mmdet.core import multiclass_nms
import pandas as pd
import numpy as np
import torch
from skimage.filters import gaussian


@HEADS.register_module()
class ConvFCLocalBBoxHead(ConvFCBBoxHead):
    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 dim=2,
                 *args,
                 **kwargs):
        super(ConvFCLocalBBoxHead,
              self).__init__(num_shared_convs, num_shared_fcs, num_cls_convs,
                             num_cls_fcs, num_reg_convs, num_reg_fcs,
                             conv_out_channels, fc_out_channels, conv_cfg,
                             norm_cfg, init_cfg, *args, **kwargs)
        try:
            self.wy,self.wbg = self.get_local_weights('./lvis_files/lvis_statistics.csv',1203,dim=dim)
        except FileNotFoundError:
            self.wy,self.wbg = self.get_local_weights('../lvis_files/lvis_statistics.csv',1203,dim=dim)
        
    def get_local_weights(self,lvis_file,num_categories,dim=32,base=10):
        
        def logarithm(x,base):
            return np.log(x) / np.log(base)
        
        df = pd.read_csv(lvis_file)
        cx = np.array(df['xmin'])+np.array(df['width'])/2
        cy = np.array(df['ymin'])+np.array(df['height'])/2
        step = 1/dim
        true_img_val = np.zeros((dim,dim))
        true_loc_bias_val = np.zeros((dim,dim,num_categories))
        categories = np.array(df['category'])-1

        for j in range(dim):
            for i in range(dim):
                dimx = [i*step,(i+1)*step]
                maskx = (cx>=dimx[0])&(cx<dimx[1])
                dimy = [j*step,(j+1)*step]
                masky = (cy>=dimy[0])&(cy<dimy[1])
                mask_final = maskx&masky
                true_img_val[j,i] = mask_final.sum()
                g = categories[mask_final]
                bins = np.bincount(g,minlength=num_categories)
#                 true_loc_bias_val[j,i,:] = bins/bins.sum() #original
                true_loc_bias_val[j,i,:] = (bins)/(bins.sum())

#         true_pobj=(true_img_val/len(categories)) #original
        pobj = (true_img_val)/len(categories)
#         p_bg = 1 - (true_img_val)/len(categories)

        py=np.expand_dims(pobj, axis=-1)*true_loc_bias_val
#         weights = -logarithm(py,base)
        fg_weights = -np.log(py)
        fg_weights[np.isinf(fg_weights)]=0
        fg_weights[np.isnan(fg_weights)]=0
        
#         for i in range(py.shape[2]):
#             weights[:,:,i]=gaussian(weights[:,:,i], sigma=1)
#         smoothed_weights[smoothed_weights==0]=1
        
#         obj_weights = -logarithm(true_pobj,base)
#         obj_weights[np.isinf(obj_weights)]=0
#         obj_weights[np.isnan(obj_weights)]=0
        
        p_bg = 1 - pobj
        bg_weight =  np.zeros((dim,dim))
#         print('before',fg_weights)
        
        ptarget = np.log(np.expand_dims(pobj, axis=-1)/1203)
        ptarget[np.isinf(ptarget)] = 0.0
        ptarget[np.isnan(ptarget)] = 0.0
        
        fg_weights = fg_weights + ptarget
#         print('after',fg_weights)
#         bg_weight =  np.zeros((dim,dim))
        
        
#         return torch.tensor(smoothed_weights-np.expand_dims(obj_weights, axis=-1),device='cuda',dtype=torch.float)
        return torch.tensor(fg_weights,device='cuda',dtype=torch.float),torch.tensor(bg_weight,device='cuda',dtype=torch.float)


    def get_norcal_weights(self,lvis_file,num_categories,dim=1):
    
        df = pd.read_csv(lvis_file)
        cx = np.array(df['xmin'])+np.array(df['width'])/2
        cy = np.array(df['ymin'])+np.array(df['height'])/2
        step = 1/dim
        true_img_val = np.zeros((dim,dim))
        true_loc_bias_val = np.zeros((dim,dim,num_categories))
        categories = np.array(df['category'])-1

        for j in range(dim):
            for i in range(dim):
                dimx = [i*step,(i+1)*step]
                maskx = (cx>=dimx[0])&(cx<dimx[1])
                dimy = [j*step,(j+1)*step]
                masky = (cy>=dimy[0])&(cy<dimy[1])
                mask_final = maskx&masky
                g = categories[mask_final]
                bins = np.bincount(g,minlength=num_categories)
                true_loc_bias_val[j,i,:] = (bins)+1
        
#         return torch.tensor(smoothed_weights-np.expand_dims(obj_weights, axis=-1),device='cuda',dtype=torch.float)

        weights = true_loc_bias_val**0.6
    
    
        bg_weights = np.ones((dim,dim))
        return torch.tensor(weights,device='cuda',dtype=torch.float),torch.tensor(bg_weights,device='cuda',dtype=torch.float)
  
    
    
    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            rois (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_shape (Sequence[int], optional): Maximum bounds for boxes,
                specifies (H, W, C) or (H, W).
            scale_factor (ndarray): Scale factor of the
               image arrange as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None

        Returns:
            tuple[Tensor, Tensor]:
                Fisrt tensor is `det_bboxes`, has the shape
                (num_boxes, 5) and last
                dimension 5 represent (tl_x, tl_y, br_x, br_y, score).
                Second tensor is the labels with shape (num_boxes, ).
        """
        

        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                bboxes.size()[0], -1)
        
        #calibration
        new_boxes = bboxes.view(bboxes.size(0), -1, 4)
        xc=((new_boxes[:,:,2]-new_boxes[:,:,0])/2 +new_boxes[:,:,0])/img_shape[1]
        yc=((new_boxes[:,:,3]-new_boxes[:,:,1])/2 +new_boxes[:,:,1])/img_shape[0]
                
        jc = torch.floor(yc*self.wy.shape[0])
        ic = torch.floor(xc*self.wy.shape[0])
        
        #select only one index
        jc = torch.tensor(torch.round(jc.mean(axis=1)),dtype=torch.long)     
        ic = torch.tensor(torch.round(ic.mean(axis=1)),dtype=torch.long)
        
        weights = self.wy[jc,ic,:]
        weights = torch.squeeze(weights,axis=1)
#         weights = weights/torch.unsqueeze(torch.norm(weights,dim=1),axis=1)
        
#         weights[weights==0]=1
        
        bg_weights = self.wbg[jc,ic]
        bg_weights = torch.unsqueeze(bg_weights,axis=1)

        
        
#         dummpy_prob = weights.new_ones((weights.size(0), 1))
        weights = torch.cat([weights, bg_weights], dim=1)
#         weights = torch.tensor(weights,dtype=torch.float)

                # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score+weights, dim=-1) if cls_score is not None else None
        
        
#         scores = scores/weights
        

#         scores /= scores.sum(dim=1, keepdim=True)
        
        
        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)

            return det_bboxes, det_labels
       
@HEADS.register_module()
class Shared2FCLocalBBoxHead(ConvFCLocalBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCLocalBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
        
            
#         self.wy,self.wbg = self.get_norcal_weights('./lvis_files/lvis_statistics.csv',1203,dim=1)
        

        
        

    
    