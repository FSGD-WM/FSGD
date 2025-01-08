# Copyright (c) OpenMMLab. All rights reserved.
import copy
from abc import abstractmethod
from typing import Dict, List, Optional, Union

from mmcv.runner import auto_fp16
from mmcv.utils import ConfigDict
from mmdet.models.builder import (DETECTORS, build_backbone, build_head,
                                  build_neck)
from mmdet.models.detectors import BaseDetector
from torch import Tensor
from typing_extensions import Literal


@DETECTORS.register_module()
class FSGDQuerySupportDetector(BaseDetector):
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
        super().__init__(init_cfg)
        if backbone is not None:
            backbone.pretrained = backbone.get('pretrained', pretrained )
        self.backbone = build_backbone(backbone)
        
        self.neck = build_neck(neck) if neck is not None else None
        
        if support_backbone is not None:
            support_backbone.pretrained = support_backbone.get('pretrained', pretrained )
        self.support_backbone = build_backbone(
            support_backbone
        ) if support_backbone is not None else self.backbone
        
        # support neck only forward support data.
        self.support_neck = build_neck(
            support_neck) if support_neck is not None else None
        assert roi_head is not None, 'missing config of roi_head'

        self.with_rpn = False
        self.rpn_with_support = False
        if rpn_head is not None:
            self.with_rpn = True
            if rpn_head.get('aggregation_layer', None) is not None:
                self.rpn_with_support = True
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = copy.deepcopy(rpn_head)
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            if roi_head.get('shared_head') is not None:
                roi_head.pretrained = roi_head.get('shared_head').get('pretrained', pretrained)
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @auto_fp16(apply_to=('img', ))
    def extract_query_feat(self, img: Tensor) -> List[Tensor]:
        feats = self.backbone(img)
        if self.with_neck:
            feats = self.neck(feats)
        return feats

    def extract_feat(self, img: Tensor) -> List[Tensor]:
        return self.extract_query_feat(img)

    @abstractmethod
    def extract_support_feat(self, img: Tensor):
        """Extract features of support data."""
        raise NotImplementedError

    @auto_fp16(apply_to=('img', ))
    def forward(self,
                query_data: Optional[Dict] = None,
                support_data: Optional[Dict] = None,
                img: Optional[List[Tensor]] = None,
                img_metas: Optional[List[Dict]] = None,
                mode: Literal['train', 'model_init', 'test'] = 'train',
                **kwargs) -> Dict:
        if mode == 'train':
            return self.forward_train(query_data, support_data, **kwargs)
        elif mode == 'model_init':
            return self.forward_model_init(img, img_metas, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, img_metas, **kwargs)
        else:
            raise ValueError(
                f'invalid forward mode {mode}, '
                f'only support `train`, `model_init` and `test` now')

    def train_step(self, data: Dict, optimizer: Union[object, Dict]) -> Dict:
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        # For most of query-support detectors, the batch size denote the
        # batch size of query data.
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data['query_data']['img_metas']))

        return outputs

    def val_step(self,
                 data: Dict,
                 optimizer: Optional[Union[object, Dict]] = None) -> Dict:
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        # For most of query-support detectors, the batch size denote the
        # batch size of query data.
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data['query_data']['img_metas']))

        return outputs

    def forward_train(self,
                      query_data: Dict,
                      support_data: Dict,
                      proposals: Optional[List] = None,
                      **kwargs) -> Dict:
        query_img = query_data['img']
        support_img = support_data['img']
        query_feats = self.extract_query_feat(query_img)
        support_feats = self.extract_support_feat(support_img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            if self.rpn_with_support:
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    query_feats,
                    support_feats,
                    query_img_metas=query_data['img_metas'],
                    query_gt_bboxes=query_data['gt_bboxes'],
                    query_gt_labels=None,
                    query_gt_bboxes_ignore=query_data.get(
                        'gt_bboxes_ignore', None),
                    support_img_metas=support_data['img_metas'],
                    support_gt_bboxes=support_data['gt_bboxes'],
                    support_gt_labels=support_data['gt_labels'],
                    support_gt_bboxes_ignore=support_data.get(
                        'gt_bboxes_ignore', None),
                    proposal_cfg=proposal_cfg)
            else:
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    query_feats,
                    copy.deepcopy(query_data['img_metas']),
                    copy.deepcopy(query_data['gt_bboxes']),
                    gt_labels=None,
                    gt_bboxes_ignore=copy.deepcopy(
                        query_data.get('gt_bboxes_ignore', None)),
                    proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(
            query_feats,
            support_feats,
            proposals=proposal_list,
            query_img_metas=query_data['img_metas'],
            query_gt_bboxes=query_data['gt_bboxes'],
            query_gt_labels=query_data['gt_labels'],
            query_gt_bboxes_ignore=query_data.get('gt_bboxes_ignore', None),
            support_img_metas=support_data['img_metas'],
            support_gt_bboxes=support_data['gt_bboxes'],
            support_gt_labels=support_data['gt_labels'],
            support_gt_bboxes_ignore=support_data.get('gt_bboxes_ignore',
                                                      None),
            **kwargs)
        losses.update(roi_losses)

        return losses

    def simple_test(self,
                    img: Tensor,
                    img_metas: List[Dict],
                    proposals: Optional[List[Tensor]] = None,
                    rescale: bool = False):
        raise NotImplementedError

    def aug_test(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward_model_init(self,
                           img: Tensor,
                           img_metas: List[Dict],
                           gt_bboxes: List[Tensor] = None,
                           gt_labels: List[Tensor] = None,
                           **kwargs):
        raise NotImplementedError

    @abstractmethod
    def model_init(self, **kwargs):
        raise NotImplementedError