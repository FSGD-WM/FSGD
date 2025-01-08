# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import auto_fp16
from mmdet.models.builder import HEADS
from mmdet.models.utils import build_linear_layer
from mmfewshot.detection.models.roi_heads.bbox_heads.meta_bbox_head import MetaBBoxHead


@HEADS.register_module()
class FSGDBBoxHead(MetaBBoxHead):
    def __init__(self,
                 reg_in_channels: int = 2048,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.with_cls:
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.in_channels,
                out_features=self.num_classes+1)
        if self.with_reg:
            out_dim_reg = 4 if self.reg_class_agnostic else 4 * self.num_classes
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=reg_in_channels,
                out_features=out_dim_reg)

    @auto_fp16()
    def forward(self, x_agg, x_query):
        if self.with_avg_pool:
            if x_agg.numel() > 0:
                x_agg = self.avg_pool(x_agg)
                x_agg = x_agg.view(x_agg.size(0), -1)
            else:
                x_agg = torch.mean(x_agg, dim=(-1, -2))
            if x_query.numel() > 0:
                x_query = self.avg_pool(x_query)
                x_query = x_query.view(x_query.size(0), -1)
            else:
                x_query = torch.mean(x_query, dim=(-1, -2))

        cls_score = self.fc_cls(x_agg) if self.with_cls else None
        bbox_pred = self.fc_reg(x_query) if self.with_reg else None

        return cls_score, bbox_pred
