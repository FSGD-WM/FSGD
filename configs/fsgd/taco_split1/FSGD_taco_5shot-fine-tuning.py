_base_ = [
    '../../_base_/datasets/nway_kshot/few_shot_taco.py',
    '../../_base_/schedules/schedule.py','../FSGD.py',
    '../../_base_/default_runtime.py'
]


# classes splits are predefined in FewShotCocoDataset
# FewShotCocoDefaultDataset predefine ann_cfg for model reproducibility
data = dict(
    train=dict(
        save_dataset=True,
        num_used_support_shots=5,
        dataset=dict(
            type='FewShotTACODefaultDataset',
            ann_cfg=[dict(method='MetaRCNN', setting='5SHOT')],
            num_novel_shots=5,
            num_base_shots=5,
        )),
    model_init=dict(num_novel_shots=5, num_base_shots=5))
evaluation = dict(interval=1600)
checkpoint_config = dict(interval=1600)
optimizer = dict(lr=0.001)
lr_config = dict(warmup=None, step=[1600])
runner = dict(max_iters=1600)



load_from = 'work_dirs/FSGD_taco_base-training/iter_18000.pth'
model = dict(
    roi_head=dict(bbox_head=dict(num_classes=16, num_meta_classes=16)),
    frozen_parameters=[
    'backbone','support_backbone', 'shared_head', 'rpn_head', 'aggregation_layer']
)