_base_ = [
    '../../_base_/datasets/nway_kshot/base_taco.py',
    '../../_base_/schedules/schedule.py', '../FSGD.py',
    '../../_base_/default_runtime.py'
]
data=dict(
    samples_per_gpu=2,
)
lr_config = dict(warmup_iters=100, step=[12000, 16000])
evaluation = dict(interval=3000,metric='bbox', classwise=True)

checkpoint_config = dict(interval=3000)
runner = dict(max_iters=18000)
optimizer = dict(lr=0.0025)
# model settings
model = dict(roi_head=dict(bbox_head=dict(num_classes=12, num_meta_classes=12)))

