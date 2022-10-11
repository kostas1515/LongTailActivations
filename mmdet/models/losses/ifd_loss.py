# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss
import pandas as pd
from .accuracy import accuracy
from .cross_entropy_loss import cross_entropy

@LOSSES.register_module()
class IFDLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 reduction='mean',
                 class_weight=None,
                 ignore_index=None,
                 loss_weight=1.0,
                 num_classes=1203,
                 frequency='./lvis_files/idf_1204.csv',
                 dimension='./lvis_files/fractal_dimension_v1.csv',
                 variant='base10_obj'):
        """CrossEntropyLoss.

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            ignore_index (int | None): The label index to be ignored.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(IFDLoss, self).__init__()
        assert (use_sigmoid is False)
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        
        self.fd_weights =pd.read_csv(dimension)['fractal_dimension'].values.tolist()
        self.fd_weights = torch.tensor(self.fd_weights,device='cuda',dtype=torch.float).unsqueeze(0)
        self.fd_weights[self.fd_weights<1.00000001]=2.0
        self.fd_weights = -torch.log(self.fd_weights - 1)
        
        if variant =='base':
            self.iif_weights = 0
            self.fd_weights = 0
        elif variant =='fd':
            self.iif_weights = 0
        elif variant =='no_fd':
            self.fd_weights = 0
            self.iif_weights = pd.read_csv(frequency)['raw'].values.tolist()
            self.iif_weights = self.iif_weights[1:]
            self.iif_weights = torch.tensor(self.iif_weights,device='cuda',dtype=torch.float).unsqueeze(0)
        else:
            self.iif_weights = pd.read_csv(frequency)[variant].values.tolist()
            self.iif_weights = self.iif_weights[1:]
            self.iif_weights = torch.tensor(self.iif_weights,device='cuda',dtype=torch.float).unsqueeze(0)
        
        

        
        
        


        self.cls_criterion = self.ifdloss
        
        # custom output channels of the classifier
        self.custom_cls_channels = True
        # custom activation of cls_score
        self.custom_activation = True
        # custom accuracy of the classsifier
        self.custom_accuracy = True
    
    def _split_cls_score(self, cls_score):
        # split cls_score to cls_score_classes and cls_score_objectness
        assert cls_score.size(-1) == self.num_classes + 2
        cls_score_classes = cls_score[..., :-2]
        cls_score_objectness = cls_score[..., -2:]
        return cls_score_classes, cls_score_objectness
    
        
        
    def get_activation(self, cls_score):
        """Get custom activation of cls_score.

        Args:
            cls_score (torch.Tensor): The prediction with shape (N, C + 2).

        Returns:
            torch.Tensor: The custom activation of cls_score with shape
                 (N, C + 1).
        """
        cls_score_classes, cls_score_objectness = self._split_cls_score(
            cls_score)
        score_classes = F.softmax(cls_score_classes, dim=-1)
        score_objectness = F.softmax(cls_score_objectness, dim=-1)
        score_pos = score_objectness[..., [0]]
        score_neg = score_objectness[..., [1]]
        score_classes = score_classes * score_pos
        scores = torch.cat([score_classes, score_neg], dim=-1)
        return scores
    
    def get_cls_channels(self, num_classes):
        """Get custom classification channels.

        Args:
            num_classes (int): The number of classes.

        Returns:
            int: The custom classification channels.
        """
        assert num_classes == self.num_classes
        return num_classes + 2
    
    def get_accuracy(self, cls_score, labels):
        """Get custom accuracy w.r.t. cls_score and labels.

        Args:
            cls_score (torch.Tensor): The prediction with shape (N, C + 2).
            labels (torch.Tensor): The learning label of the prediction.

        Returns:
            Dict [str, torch.Tensor]: The accuracy for objectness and classes,
                 respectively.
        """
        pos_inds = labels < self.num_classes
        obj_labels = (labels == self.num_classes).long()
        cls_score_classes, cls_score_objectness = self._split_cls_score(
            cls_score)
        acc_objectness = accuracy(cls_score_objectness, obj_labels)
        acc_classes = accuracy(cls_score_classes[pos_inds], labels[pos_inds])
        acc = dict()
        acc['acc_objectness'] = acc_objectness
        acc['acc_classes'] = acc_classes
        return acc

    def forward(self,
                cls_score,
                labels,
                label_weights=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction with shape (N, C + 2).
            labels (torch.Tensor): The learning label of the prediction.
            label_weights (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                 the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                 Options are "none", "mean" and "sum".
        Returns:
            Dict [str, torch.Tensor]:
            The dict of calculated losses
                 for objectness and classes, respectively.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        assert cls_score.size(-1) == self.num_classes + 2
        pos_inds = labels < self.num_classes
        # 0 for pos, 1 for neg
        obj_labels = (labels == self.num_classes).long()


        if label_weights is not None:
            label_weights = label_weights.float()
        else:
            label_weights = labels.new_ones(labels.size(), dtype=torch.float)

        cls_score_classes, cls_score_objectness = self._split_cls_score(
            cls_score)
        # calculate loss_cls_classes (only need pos samples)
        if pos_inds.sum() > 0:
            loss_cls_classes = self.loss_weight * self.cls_criterion(
                cls_score_classes[pos_inds], labels[pos_inds], reduction=reduction, avg_factor=avg_factor)
        else:
            loss_cls_classes = cls_score_classes[pos_inds].sum()
        # calculate loss_cls_objectness
        loss_cls_objectness = self.loss_weight * cross_entropy(
            cls_score_objectness, obj_labels, label_weights, reduction,
            avg_factor)


        loss_cls = dict()
        loss_cls['loss_cls_objectness'] = loss_cls_objectness
        loss_cls['loss_cls_classes'] = loss_cls_classes

        return loss_cls
    
    
    
    
    def ifdloss(self,
                  pred,
                  label,
                  weight=None,
                  reduction='mean',
                  avg_factor=None,
                  class_weight=None,
                  ignore_index=-100):
        
        """Calculate the CrossEntropy loss.

        Args:
            pred (torch.Tensor): The prediction with shape (N, C), C is the number
                of classes.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            reduction (str, optional): The method used to reduce the loss.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            class_weight (list[float], optional): The weight for each class.
            ignore_index (int | None): The label index to be ignored.
                If None, it will be set to default value. Default: -100.

        Returns:
            torch.Tensor: The calculated loss
        """
        # The default value of ignore_index is the same as F.cross_entropy
        ignore_index = -100 if ignore_index is None else ignore_index
        # element-wise losses
        pestim = (pred - self.iif_weights - self.fd_weights)
        
        
        loss = F.cross_entropy(
            pestim,
            label,
            weight=class_weight,
            reduction='none',
            ignore_index=ignore_index)
        

        # apply weights and do the reduction
        if torch.isinf(loss).sum()>0:
            loss[torch.isinf(loss)]=0.0
        
            
        if weight is not None:
            weight = weight.float()
        loss = weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
        
        
            
        
        return loss

