import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..builder import LOSSES
from .utils import weight_reduce_loss
from .accuracy import accuracy
import pandas as pd
import torch.distributed as dist



def _expand_onehot_labels(labels, label_weights, label_channels, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(
        valid_mask & (labels < label_channels), as_tuple=False)

    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1

    valid_mask = valid_mask.view(-1, 1).expand(labels.size(0),
                                               label_channels).float()
    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.view(-1, 1).repeat(1, label_channels)
        bin_label_weights *= valid_mask

    return bin_labels, bin_label_weights


def logsumexp(x):
    alpha=torch.exp(x)
    return alpha+torch.log(1.0-torch.exp(-alpha))

@LOSSES.register_module()
class GumbelCrossEntropyLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 temperature=1.0,
                 use_iif=None,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 num_classes=1203,
                 lvis_file='./lvis_files/idf_1204.csv',
                 **kwargs):
        """CrossEntropyLoss.

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(GumbelCrossEntropyLoss, self).__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.num_classes = num_classes
        
        # custom output channels of the classifier
        self.custom_cls_channels = True
        # custom activation of cls_score
        self.custom_activation = True
        # custom accuracy of the classsifier
        self.custom_accuracy = True
        self.temperature=temperature
        
        if use_iif is not None:
            self.use_iif = True
            if use_iif == 'adaptive':
                self.adaptive_iif = True
                self.moving_avg=2.0
                self.alpha=0.9
            else:
                self.adaptive_iif = False
                self.iif_weights = pd.read_csv(lvis_file)[use_iif].values.tolist()
                self.iif_weights = self.iif_weights[1:]+[1.0] #+1 for bg
                self.iif_weights = torch.tensor(self.iif_weights,device='cuda',dtype=torch.float).unsqueeze(0)
        else:
            self.use_iif = False
            

        self.cls_criterion = self.gumbel_cross_entropy
        
    def get_activation(self, cls_score,raw=False):
        """Get custom activation of cls_score.

        Args:
            cls_score (torch.Tensor): The prediction with shape (N, C).

        Returns:
            torch.Tensor: The custom activation of cls_score with shape
                 (N, C).
        """
        if raw is True:
            scores = 1/(torch.exp(torch.exp(-cls_score/self.temperature)))
            return scores
        else:
            if self.use_iif is True:
                if self.adaptive_iif is True:
                    avg_w = self.get_adaptive_weight(cls_score)
                    scores = self.get_activation(avg_w*cls_score,raw=True)
                else:
                    scores = self.get_activation(self.iif_weights*cls_score,raw=True)
                    
            
        
        return scores
    
    def get_cls_channels(self, num_classes):
        """Get custom classification channels.

        Args:
            num_classes (int): The number of classes.

        Returns:
            int: The custom classification channels.
        """
        assert num_classes == self.num_classes
        return num_classes + 1
    
    def get_accuracy(self, cls_score, labels):
        """Get custom accuracy w.r.t. cls_score and labels.

        Args:
            cls_score (torch.Tensor): The prediction with shape (N, C).
            labels (torch.Tensor): The learning label of the prediction.

        Returns:
            Dict [str, torch.Tensor]: The accuracy for objectness and classes,
                 respectively.
        """
        pos_inds = labels < self.num_classes
        acc_classes = accuracy(cls_score[pos_inds], labels[pos_inds])
        acc = dict()
        acc['acc_classes'] = acc_classes
        
        return acc
    

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls
    
    def get_adaptive_weight(self,pred):
        w = -torch.log10(self.get_activation(pred,raw=True))
        batch_w=[torch.zeros_like(w) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_w,w)
        batch_w=torch.cat(batch_w,axis=0)
        avg_w = batch_w.mean(axis=0).unsqueeze(0)
#         print('avg weight is:',avg_w)
        #make bg weight 1
        avg_w[torch.isinf(avg_w)]=1.0
        avg_w[torch.isnan(avg_w)]=1.0
        avg_w[:,-1]=1.0
        avg_w=torch.clamp(avg_w,max=6.2)
        
        self.moving_avg = self.moving_avg*self.alpha + (1-self.alpha)*avg_w
        
        
        return self.moving_avg
        


    def gumbel_cross_entropy(self,
                            pred,
                            label,
                            weight=None,
                            reduction='mean',
                            avg_factor=None,
                            class_weight=None,
                            ignore_index=-100):
        
        ignore_index = -100 if ignore_index is None else ignore_index
        if pred.dim() != label.dim():
            label, weight = _expand_onehot_labels(label, weight, pred.size(-1),
                                                ignore_index)   
        if weight is not None:
            weight = weight.float()
        
        if self.use_iif is True:
            if self.adaptive_iif is True:
                avg_w = self.get_adaptive_weight(pred)
                pred=torch.clamp(pred*avg_w,min=-5,max=12)
#                 pred=pred*avg_w
            else:
#                 pred = pred*self.iif_weights
                pred=torch.clamp(pred*self.iif_weights,min=-5,max=12)
                
        else:
            pred=torch.clamp(pred,min=-5,max=12)
            
        
        loss=torch.exp(-pred/self.temperature)*label.float() +(label.float()-1.0)*(logsumexp(-pred/self.temperature)-torch.exp(-pred/self.temperature))
        
        loss = weight_reduce_loss(
            loss, weight, reduction=reduction, avg_factor=avg_factor)
        loss=torch.clamp(loss,max=30.0)
        
        return loss
        
