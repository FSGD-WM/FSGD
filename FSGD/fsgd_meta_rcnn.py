# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional

import torch
from mmcv.runner import auto_fp16
from mmcv.utils import ConfigDict
from mmdet.models.builder import DETECTORS
from torch import Tensor

from .fsgd_query_support_detector import FSGDQuerySupportDetector


@DETECTORS.register_module()
class FSGDMetaRCNN(FSGDQuerySupportDetector):
    def __init__(self,
                 backbone: ConfigDict,
                 neck: Optional[ConfigDict] = None,
                 support_backbone: Optional[ConfigDict] = None,
                 support_neck: Optional[ConfigDict] = None,
                 rpn_head: Optional[ConfigDict] = None,
                 roi_head: Optional[ConfigDict] = None,
                 train_cfg: Optional[ConfigDict] = None,
                 test_cfg: Optional[ConfigDict] = None,
                 pretrained: Optional[ConfigDict] = None,
                 init_cfg: Optional[ConfigDict] = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            support_backbone=support_backbone,
            support_neck=support_neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.is_model_init = False
        self._forward_saved_support_dict = {
            'gt_labels': [],
            'roi_feats': [],
        }
        self.inference_support_dict = {}
    
    # separate the support feature extraction
    @auto_fp16(apply_to=('img', ))
    def extract_support_feat(self, img):
        feats = self.support_backbone(img, use_meta_conv=True)
        if self.support_neck is not None:
            feats = self.support_neck(feats)
        return feats

    def forward_model_init(self,
                           img: Tensor,
                           img_metas: List[Dict],
                           gt_bboxes: List[Tensor] = None,
                           gt_labels: List[Tensor] = None,
                           **kwargs):
        self.is_model_init = False
        assert len(gt_labels) == img.size(
            0), 'Support instance have more than two labels'
        feats = self.extract_support_feat(img)
        roi_feat = self.roi_head.extract_support_feats(feats)
        self._forward_saved_support_dict['gt_labels'].extend(gt_labels)
        self._forward_saved_support_dict['roi_feats'].extend(roi_feat)
        return {'gt_labels': gt_labels, 'roi_feat': roi_feat}

    def model_init(self):
        gt_labels = torch.cat(self._forward_saved_support_dict['gt_labels'])
        roi_feats = torch.cat(self._forward_saved_support_dict['roi_feats'])
        class_ids = set(gt_labels.data.tolist())
        self.inference_support_dict.clear()
        for class_id in class_ids:
            self.inference_support_dict[class_id] = roi_feats[
                gt_labels == class_id].mean([0], True)
        # set the init flag
        self.is_model_init = True
        # reset support features buff
        for k in self._forward_saved_support_dict.keys():
            self._forward_saved_support_dict[k].clear()

    def simple_test(self,
                    img: Tensor,
                    img_metas: List[Dict],
                    proposals: Optional[List[Tensor]] = None,
                    rescale: bool = False):
        assert self.with_bbox, 'Bbox head must be implemented.'
        assert len(img_metas) == 1, 'Only support single image inference.'
        if not self.is_model_init:
            # process the saved support features
            self.model_init()

        query_feats = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test(query_feats, img_metas)
        else:
            proposal_list = proposals
        return self.roi_head.simple_test(
            query_feats,
            copy.deepcopy(self.inference_support_dict),
            proposal_list,
            img_metas,
            rescale=rescale)