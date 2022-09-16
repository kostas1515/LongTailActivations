from mmdet.core import bbox2roi
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads import StandardRoIHead
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler


@HEADS.register_module()
class StandardLocalRoIHead(StandardRoIHead):

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        if self.training:
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_results = dict(
                cls_score=cls_score, bbox_pred=bbox_pred,
                bbox_feats=bbox_feats)
        else:
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            #do calibration
            bbox_results = dict(
                cls_score=cls_score, bbox_pred=bbox_pred,
                bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)
        
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results
