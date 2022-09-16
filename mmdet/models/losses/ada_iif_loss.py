# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss
import pandas as pd
from .accuracy import accuracy
import torch.distributed as dist


def _update_mean_score(self, cls_score, sampling_results, gt_labels, selected_labels, rank, alpha=0.9, bg_score=0.01):
        batch_gt_labels, batch_mean_scores = \
                self._compute_batch_mean_score(cls_score, sampling_results, gt_labels, selected_labels)

        if rank != -1:
            batch_gt_labels, batch_mean_scores = \
                collect_tensor_from_dist([batch_gt_labels, batch_mean_scores], [-1, 0])

        # compute the mean score for all collected scores
        # including both current batch samples and sampling samples together
        for gt_label in set(list(batch_gt_labels.cpu().numpy())):
            if gt_label == -1:
                continue
            batch_mean_score = batch_mean_scores[batch_gt_labels == gt_label]
            number_gt = len(batch_mean_score)
            number_alpha = alpha ** number_gt
            self.mean_score[gt_label] = number_alpha * self.mean_score[gt_label] + (
                    1 - number_alpha) * batch_mean_score.mean()

        self.mean_score[-1] = bg_score


@LOSSES.register_module()
class ADAIIFLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 reduction='mean',
                 class_weight=None,
                 ignore_index=None,
                 loss_weight=1.0,
                 num_classes=1203):
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
        super(ADAIIFLoss, self).__init__()
        assert (use_sigmoid is False)
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
                

        self.cls_criterion = self.cross_entropy
        
        # custom output channels of the classifier
        self.custom_cls_channels = True
        # custom activation of cls_score
        self.custom_activation = True
        # custom accuracy of the classsifier
        self.custom_accuracy = True
        
        
    def get_activation(self, cls_score):
        """Get custom activation of cls_score.

        Args:
            cls_score (torch.Tensor): The prediction with shape (N, C).

        Returns:
            torch.Tensor: The custom activation of cls_score with shape
                 (N, C).
        """
        w = -torch.log(torch.softmax(cls_score[...,:-1],dim=-1))
        batch_w=[torch.zeros_like(w) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_w,w)
        batch_w=torch.cat(batch_w,axis=0)
        avg_w = batch_w.mean(axis=0).unsqueeze(0)
        
        bg_weights=cls_score.new_ones(1,1)
        final_weights=torch.cat((avg_w,bg_weights),1)
        
        scores = torch.softmax(final_weights*cls_score,dim=-1)

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
                ignore_index=None,
                **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss. Options are "none", "mean" and "sum".
            ignore_index (int | None): The label index to be ignored.
                If not None, it will override the default value. Default: None.
        Returns:
            torch.Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if ignore_index is None:
            ignore_index = self.ignore_index

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(
                self.class_weight, device=cls_score.device)
        else:
            class_weight = None
            
        
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=ignore_index,
            **kwargs)
        return loss_cls
    
    
    
    
    def cross_entropy(self,
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
        
        w = -torch.log(torch.softmax(pred[...,:-1],dim=-1))
        batch_w=[torch.zeros_like(w) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_w,w)
        batch_w=torch.cat(batch_w,axis=0)
        avg_w = batch_w.mean(axis=0).unsqueeze(0)
        
        bg_weights=pred.new_ones(1,1)
        final_weights=torch.cat((avg_w,bg_weights),1)
#         print(final_weights)
        loss = F.cross_entropy(
            final_weights*pred,
            label,
            weight=class_weight,
            reduction='none',
            ignore_index=ignore_index)

        # apply weights and do the reduction
        
            
        if weight is not None:
            weight = weight.float()
        loss = weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

        return loss

