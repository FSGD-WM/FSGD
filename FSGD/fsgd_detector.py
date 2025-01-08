import copy
import torch
from mmcv.utils import ConfigDict
from typing import Dict, List, Optional
from mmdet.models.builder import DETECTORS, build_neck
from mmcv.runner import auto_fp16
from .utils import TestMixins
from .fsgd_meta_rcnn import FSGDMetaRCNN
from torch import Tensor
import torch.nn.functional as F


@DETECTORS.register_module()
class FSGDDetector(FSGDMetaRCNN, TestMixins):
    def __init__(self,
                 with_refine=False,
                 backbone: ConfigDict = None,
                 neck: Optional[ConfigDict] = None,
                 support_backbone: Optional[ConfigDict] = None,
                 support_neck: Optional[ConfigDict] = None,
                 aggregation_layer: Optional[ConfigDict] = None,
                 rpn_head: Optional[ConfigDict] = None,
                 roi_head: Optional[ConfigDict] = None,
                 train_cfg: Optional[ConfigDict] = None,
                 test_cfg: Optional[ConfigDict] = None,
                 pretrained: Optional[ConfigDict] = None,
                 init_cfg: Optional[ConfigDict] = None,
                 use_meta_conv: Optional[bool] = None) -> None:
        super().__init__(backbone=backbone,
                         neck=neck,
                         support_backbone=support_backbone,
                         support_neck=support_neck,
                         rpn_head=rpn_head,
                         roi_head=roi_head,
                         train_cfg=train_cfg,
                         test_cfg=test_cfg,
                         pretrained=pretrained,
                         init_cfg=init_cfg)
        
        self.with_refine = with_refine

        self._forward_saved_support_dict = {
            'gt_labels': [],
            'roi_feats': [],
            'support_feats': [],
            'gt_bboxes': []
        }
        self.inference_support_feat_dict = {}
        self.use_meta_conv = True if use_meta_conv is None else use_meta_conv

    def forward_train(self,
                      query_data: Dict,
                      support_data: Dict,
                      proposals: Optional[List] = None,
                      **kwargs) -> Dict:
        query_img = query_data['img']
        support_img = support_data['img']
        query_feats = self.extract_query_feat(query_img)
        
        # stop gradient at RPN
        query_feats_rpn = query_feats
        query_feats_rcnn = query_feats
        support_feats = self.extract_support_feat(support_img)

        # support_feat upsample
        _, _, new_height, new_width = query_feats_rpn[0].shape
        support_nums = support_feats[0].shape[0]
        upsample_support = F.interpolate(support_feats[0], size=(
            new_height, new_width), mode='bilinear', align_corners=False)
        # faeture aggregation
        agg_features = []
        for query_i in query_feats_rpn[0]:
            agg_feature = []
            agg_feature.append(query_i * upsample_support)
            agg_feature.append(query_i - upsample_support)
            agg_feature.append(query_i.unsqueeze(
                0).repeat(support_nums, 1, 1, 1))
            agg_feature = torch.cat(agg_feature, dim=1)
            agg_feature, _ = torch.max(agg_feature, dim=0)
            agg_features.append(agg_feature)
        agg_features = torch.stack(agg_features)
        agg_features = [agg_features]

        losses = dict()
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            if self.rpn_with_support:
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    agg_features,
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
                    agg_features,
                    copy.deepcopy(query_data['img_metas']),
                    copy.deepcopy(query_data['gt_bboxes']),
                    gt_labels=None,
                    gt_bboxes_ignore=copy.deepcopy(
                        query_data.get('gt_bboxes_ignore', None)),
                    proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals
        # RCNN forward and loss
        roi_losses = self.roi_head.forward_train(
            query_feats_rcnn,
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

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        assert self.with_bbox, 'Bbox head must be implemented.'
        assert len(img_metas) == 1, 'Only support single image inference.'
        if not self.is_model_init:
            # process the saved support features
            self.model_init()

        query_feats = self.extract_feat(img)
        # attention aggregation
        support_feats = list(self.inference_support_feat_dict.values())
        support_feats = torch.cat(support_feats, dim=0)
        # feature aggregation
        _, _, new_height, new_width = query_feats[0].shape
        support_nums = support_feats.shape[0]
        upsample_support = F.interpolate(support_feats, size=(
            new_height, new_width), mode='bilinear', align_corners=False)
        agg_features = []
        for query_i in query_feats[0]:
            agg_feature = []
            agg_feature.append(query_i * upsample_support)
            agg_feature.append(query_i - upsample_support)
            agg_feature.append(query_i.unsqueeze(
                0).repeat(support_nums, 1, 1, 1))
            agg_feature = torch.cat(agg_feature, dim=1)
            agg_feature, _ = torch.max(agg_feature, dim=0)
            agg_features.append(agg_feature)
        agg_features = torch.stack(agg_features)
        agg_features = [agg_features]

        # rpn forward
        if proposals is None:
            proposal_list = self.rpn_head.simple_test(agg_features, img_metas)
        else:
            proposal_list = proposals
        # rcnn forward
        bbox_results = self.roi_head.simple_test(
            query_feats,
            copy.deepcopy(self.inference_support_dict),
            proposal_list,
            img_metas,
            rescale=rescale)
        if self.with_refine:
            return self.refine_test(bbox_results, img_metas)
        else:
            return bbox_results

    @auto_fp16(apply_to=('img', ))
    def extract_support_feat(self, img):
        feats = self.support_backbone(img, use_meta_conv=self.use_meta_conv)
        if self.support_neck is not None:
            feats = self.support_neck(feats)
        return feats

    def forward_model_init(self,
                           img: Tensor,
                           img_metas: List[Dict],
                           gt_bboxes: List[Tensor] = None,
                           gt_labels: List[Tensor] = None,
                           **kwargs):
        # `is_model_init` flag will be reset when forward new data.
        self.is_model_init = False
        assert len(gt_labels) == img.size(
            0), 'Support instance have more than two labels'

        feats = self.extract_support_feat(img)

        roi_feat = self.roi_head.extract_support_feats(feats)

        self._forward_saved_support_dict['gt_labels'].extend(gt_labels)
        self._forward_saved_support_dict['roi_feats'].extend(roi_feat)
        self._forward_saved_support_dict['support_feats'].extend(feats)

        return {'gt_labels': gt_labels, 'roi_feat': roi_feat, 'feats': feats}

    def model_init(self):
        gt_labels = torch.cat(self._forward_saved_support_dict['gt_labels'])
        roi_feats = torch.cat(self._forward_saved_support_dict['roi_feats'])
        support_feats = torch.cat(
            self._forward_saved_support_dict['support_feats'])


        class_ids = set(gt_labels.data.tolist())
        self.inference_support_dict.clear()
        self.inference_support_feat_dict.clear()
        for class_id in class_ids:
            self.inference_support_dict[class_id] = roi_feats[
                gt_labels == class_id].mean([0], True)
            self.inference_support_feat_dict[class_id] = support_feats[
                gt_labels == class_id].mean([0], True)

        self.is_model_init = True
        for k in self._forward_saved_support_dict.keys():
            self._forward_saved_support_dict[k].clear()
