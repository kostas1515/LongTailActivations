import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..builder import LOSSES
from .utils import weight_reduce_loss
from .accuracy import accuracy
import pandas as pd


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


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         margins=None,
                         variant='sigmoid',
                         ignore_index=-100):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored.
            If None, it will be set to default value. Default: -100.

    Returns:
        torch.Tensor: The calculated loss.
    """
    # The default value of ignore_index is the same as F.cross_entropy
    
    ignore_index = -100 if ignore_index is None else ignore_index
    if pred.dim() != label.dim():
        label, weight = _expand_onehot_labels(label, weight, pred.size(-1),
                                              ignore_index)   
    if weight is not None:
        weight = weight.float()
    
    if (margins is not None):
        pred[:,:-1]-= margins
        
    
    if variant =='sigmoid': 
        loss = F.binary_cross_entropy_with_logits(pred, label.float(), reduction='none')
    elif variant =='gumbel':
        pestim = 1/(torch.exp(torch.exp(-pred)))
        loss = F.binary_cross_entropy(pestim, label.float(), reduction='none')
    elif variant =='normal':
        pestim=1/2+torch.erf(pred/(2**(1/2)))/2
        loss = F.binary_cross_entropy(pestim, label.float(), reduction='none')
    elif variant =='softmax':
        pred[:,:-1]+= margins
        lvis_img_freq = (pd.read_csv('./lvis_files/idf_1204.csv')['img_freq']).values
        lvis_img_freq = (torch.tensor(lvis_img_freq,dtype=torch.float,device='cuda')[1:]).unsqueeze(0)
        pred[:,:-1]+= torch.log(lvis_img_freq)
        loss = F.cross_entropy(pred, label.argmax(axis=1), reduction='mean')
        return loss
    
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)
#     print('loss is:',loss)
    
    return loss



@LOSSES.register_module()
class BalancedSoftmax(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 json_file='./lvis_files/idf_1204.csv',
                 variant='prob',
                 cls_variant='sigmoid',
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 num_classes=1203):
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
        super(BalancedSoftmax, self).__init__()
        self.lvis_weights = (pd.read_csv(json_file)[variant]).values
        self.lvis_weights = (torch.tensor(self.lvis_weights,dtype=torch.float,device='cuda')[1:]).unsqueeze(0)
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.cls_variant = cls_variant
        
        # custom output channels of the classifier
        self.custom_cls_channels = True
        # custom activation of cls_score
        self.custom_activation = True
        # custom accuracy of the classsifier
        self.custom_accuracy = True


        self.cls_criterion = binary_cross_entropy
        
    def get_activation(self, cls_score):
        """Get custom activation of cls_score.

        Args:
            cls_score (torch.Tensor): The prediction with shape (N, C).

        Returns:
            torch.Tensor: The custom activation of cls_score with shape
                 (N, C).
        """
        if self.cls_variant == 'sigmoid':
            scores = torch.sigmoid(cls_score)
        elif self.cls_variant == 'gumbel':
            scores = 1/(torch.exp(torch.exp(-cls_score)))
        elif self.cls_variant == 'normal':
            scores=1/2+torch.erf(cls_score/(2**(1/2)))/2
        elif self.cls_variant == 'softmax':
            scores = torch.softmax(cls_score,dim=-1)
        
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
        acc_classes = accuracy(cls_score, labels)
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
            margins=self.lvis_weights,
            variant=self.cls_variant,
            **kwargs)
        return loss_cls
