_base_ = [
    './meta-rcnn_r50_c4.py',
]
norm_cfg = dict(type='BN', requires_grad=False)
custom_imports = dict(
    imports=[
        FSGD.FSGDDetector, 
        FSGD.FSGDRoIHead, 
        FSGD.FSGDBBoxHead,
        ],
    allow_failed_imports=False)

pretrained = 'open-mmlab://detectron2/resnet101_caffe'
# model settings
model = dict(
    type='FSGDDetector',
    pretrained=pretrained,
    support_backbone=dict(
        type='ResNetWithMetaConv',
        pretrained=pretrained,
        depth=101,
        num_stages=3,
        strides=(1, 2, 2),
        dilations=(1, 1, 1),
        out_indices=(2, ),
        frozen_stages=2,
        norm_cfg=norm_cfg,
        norm_eval=True,
        style='caffe'),
    backbone=dict(type='ResNet', pretrained=pretrained, depth=101),
    rpn_head=dict(
        type='RPNHead',
        in_channels=1024*3),
    roi_head=dict(
        type='FSGDRoIHead',
        shared_head=dict(pretrained=pretrained),
        bbox_head=dict(
            type='FSGDBBoxHead',
            in_channels=4096,
            reg_in_channels=2048,
            meta_cls_in_channels=2048,
            num_classes=12,
            num_meta_classes=12),
        aggregation_layer=dict(
            type='AggregationLayer',
            aggregator_cfgs=[
                dict(
                    type='DepthWiseCorrelationAggregator',
                    in_channels=2048,
                    out_channels=1024,
                    with_fc=True),
                dict(
                    type='DifferenceAggregator',
                    in_channels=2048,
                    out_channels=1024,
                    with_fc=True),
            ],
            init_cfg=[
                dict(
                    type='Normal',
                    layer=['Conv1d', 'Conv2d', 'Linear'],
                    mean=0.0,
                    std=0.001),
                dict(type='Normal', layer=['BatchNorm1d'], mean=1.0, std=0.02)
            ])))
