import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from ..builder import LOSSES
from .accuracy import accuracy

'''
Adaptive Class Supression Loss
Author: changewt
Source: https://github.com/CASIA-IVA-Lab/ACSL
Paper: https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Adaptive_Class_Suppression_Loss_for_Long-Tail_Object_Detection_CVPR_2021_paper.pdf
'''

@LOSSES.register_module()
class ACSL(nn.Module):

    def __init__(self, score_thr=0.7, json_file='../../datasets/lvis/data/lvis_v1_train.json', loss_weight=1.0,variant='sigmoid'):

        super(ACSL, self).__init__()

        self.score_thr = score_thr
        assert self.score_thr > 0 and self.score_thr < 1
        self.loss_weight = loss_weight

        assert len(json_file) != 0
        self.freq_group = self.get_freq_info(json_file)
        
        self.variant = variant
        
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
        if self.variant=='gumbel':
            scores = 1/(torch.exp(torch.exp(-cls_score)))
        elif self.variant=='normal':
            scores=1/2+torch.erf(cls_score/(2**(1/2)))/2
        elif self.variant=='softmax':
            scores= torch.softmax(cls_score,dim=-1)
        elif self.variant=='sigmoid':
            scores= torch.sigmoid(cls_score)

        return scores
    
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
        acc['acc_Cls'] = acc_classes
        
        return acc

    def get_freq_info(self, json_file):
        cats = json.load(open(json_file, 'r'))['categories']

        freq_dict = {'rare': [], 'common': [], 'freq': []}

        for cat in cats:
            if cat['frequency'] == 'r':
                freq_dict['rare'].append(cat['id'])
            elif cat['frequency'] == 'c':
                freq_dict['common'].append(cat['id'])
            elif cat['frequency'] == 'f':
                freq_dict['freq'].append(cat['id'])
            else:
                print('Something wrong with the json file.')

        return freq_dict

    def forward(self, cls_logits, labels, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        device = cls_logits.device
        self.n_i, self.n_c = cls_logits.size()
        # expand the labels to all their parent nodes
        target = cls_logits.new_zeros(self.n_i, self.n_c)
        # weight mask, decide which class should be ignored
        #weight_mask = cls_logits.new_zeros(self.n_i, self.n_c)
        
        #convert 1204 to 0 for bg index
        index_inversion = torch.tensor([cls_logits.shape[1]-1]+(torch.arange(cls_logits.shape[1]-1)).tolist(),device='cuda')
        cls_logits = torch.index_select(cls_logits, 1, index_inversion)
        labels = (labels+1)%self.n_c
        
        unique_label = torch.unique(labels)
        
        with torch.no_grad():
            if self.variant =='sigmoid':
                sigmoid_cls_logits = torch.sigmoid(cls_logits)
            elif self.variant =='gumbel':
                sigmoid_cls_logits = 1/(torch.exp(torch.exp(-(torch.clamp(cls_logits,min=-4,max=10)))))
            elif self.variant =='normal':
                sigmoid_cls_logits = 1/2+torch.erf(torch.clamp(cls_logits,min=-5,max=5)/(2**(1/2)))/2
            elif self.variant =='softmax':
                sigmoid_cls_logits = torch.softmax(cls_logits,dim=-1)
            
        # for each sample, if its score on unrealated class hight than score_thr, their gradient should not be ignored
        # this is also applied to negative samples
        high_score_inds = torch.nonzero(sigmoid_cls_logits>=self.score_thr)
        weight_mask = torch.sparse_coo_tensor(high_score_inds.t(), cls_logits.new_ones(high_score_inds.shape[0]), size=(self.n_i, self.n_c), device=device).to_dense()

        for cls in unique_label:
            cls = cls.item()
            cls_inds = torch.nonzero(labels == cls).squeeze(1)
            if cls == 0:
                # construct target vector for background samples
                target[cls_inds, 0] = 1
                # for bg, set the weight of all classes to 1
                weight_mask[cls_inds] = 0

                cls_inds_cpu = cls_inds.cpu()

                # Solve the rare categories, random choost 1/3 bg samples to suppress rare categories
                rare_cats = self.freq_group['rare']
                rare_cats = torch.tensor(rare_cats, device=cls_logits.device)
                choose_bg_num = int(len(cls_inds) * 0.01)
                choose_bg_inds = torch.tensor(np.random.choice(cls_inds_cpu, size=(choose_bg_num), replace=False), device=device)

                tmp_weight_mask = weight_mask[choose_bg_inds]
                tmp_weight_mask[:, rare_cats] = 1

                weight_mask[choose_bg_inds] = tmp_weight_mask

                # Solve the common categories, random choost 2/3 bg samples to suppress rare categories
                common_cats = self.freq_group['common']
                common_cats = torch.tensor(common_cats, device=cls_logits.device)
                choose_bg_num = int(len(cls_inds) * 0.1)
                choose_bg_inds = torch.tensor(np.random.choice(cls_inds_cpu, size=(choose_bg_num), replace=False), device=device)

                tmp_weight_mask = weight_mask[choose_bg_inds]
                tmp_weight_mask[:, common_cats] = 1

                weight_mask[choose_bg_inds] = tmp_weight_mask
                
                # Solve the frequent categories, random choost all bg samples to suppress rare categories
                freq_cats = self.freq_group['freq']
                freq_cats = torch.tensor(freq_cats, device=cls_logits.device)
                choose_bg_num = int(len(cls_inds) * 1.0)
                choose_bg_inds = torch.tensor(np.random.choice(cls_inds_cpu, size=(choose_bg_num), replace=False), device=device)

                tmp_weight_mask = weight_mask[choose_bg_inds]
                tmp_weight_mask[:, freq_cats] = 1

                weight_mask[choose_bg_inds] = tmp_weight_mask

                # Set the weight for bg to 1
                weight_mask[cls_inds, 0] = 1
                
            else:
                # construct target vector for foreground samples
                cur_labels = [cls]
                cur_labels = torch.tensor(cur_labels, device=cls_logits.device)
                tmp_label_vec = cls_logits.new_zeros(self.n_c)
                tmp_label_vec[cur_labels] = 1
                tmp_label_vec = tmp_label_vec.expand(cls_inds.numel(), self.n_c)
                target[cls_inds] = tmp_label_vec
                # construct weight mask for fg samples
                tmp_weight_mask_vec = weight_mask[cls_inds]
                # set the weight for ground truth category
                tmp_weight_mask_vec[:, cur_labels] = 1

                weight_mask[cls_inds] = tmp_weight_mask_vec
                
        if self.variant =='sigmoid': 
            cls_loss = F.binary_cross_entropy_with_logits(cls_logits, target.float(), reduction='none')
        elif self.variant =='gumbel':
            pestim = 1/(torch.exp(torch.exp(-(torch.clamp(cls_logits,min=-4,max=10)))))
            cls_loss = F.binary_cross_entropy(pestim, target.float(), reduction='none')
        elif self.variant =='normal':
            pestim=1/2+torch.erf(torch.clamp(cls_logits,min=-5,max=5)/(2**(1/2)))/2
            cls_loss = F.binary_cross_entropy(pestim, target.float(), reduction='none')
        elif self.variant =='softmax':
            cls_loss = F.cross_entropy(weight_mask*cls_logits, target.argmax(axis=1), reduction='mean')
            return cls_loss
        
        return torch.sum(weight_mask * cls_loss) / self.n_i
