# ==============================================================================

# POSEC3D FINE-TUNING CONFIGURATION CHO BÀI TOÁN FALL DETECTION

# Dataset: 2 classes - ADL và FALL

# Framework: MMAction2 / MMEngine

# ==============================================================================

default_scope = 'mmaction'

# ------------------------------------------------------------------------------

# 1. THIẾT LẬP MÔ HÌNH POSEC3D

# ------------------------------------------------------------------------------

# Mô hình sử dụng kiến trúc Recognizer3D với backbone ResNet3dSlowOnly.

# Input của mô hình là 3D pseudo-heatmap được sinh từ keypoint người.

# Số channel đầu vào là 17, tương ứng với 17 keypoints theo định dạng COCO.

model = dict(
type='Recognizer3D',
backbone=dict(
    type='ResNet3dSlowOnly',
    depth=50,

    # Không khai báo pretrained trong backbone.
    # Việc nạp checkpoint pretrained được thực hiện bằng biến load_from ở cuối file.
    pretrained=None,

    in_channels=17,
    base_channels=32,
    num_stages=3,
    out_indices=(2,),
    stage_blocks=(4, 6, 3),

    conv1_stride_s=1,
    pool1_stride_s=1,
    inflate=(0, 1, 1),

    spatial_strides=(2, 2, 2),
    temporal_strides=(1, 1, 2),
    dilations=(1, 1, 1)
),

cls_head=dict(
    type='I3DHead',
    in_channels=512,

    # Bài toán Fall Detection gồm 2 lớp:
    # 0: ADL - Activities of Daily Living
    # 1: FALL - Fall action
    num_classes=2,

    spatial_type='avg',
    dropout_ratio=0.5,

    # class_weight giúp tăng độ ưu tiên cho lớp FALL nếu dữ liệu bị lệch lớp.
    # Có thể điều chỉnh sau khi thống kê số lượng mẫu ADL/FALL.
    loss_cls=dict(
        type='CrossEntropyLoss',
        loss_weight=1.0,
        class_weight=[0.3, 1.0]
    )
),

train_cfg=dict(),
test_cfg=dict(average_clips='prob')

)

# ------------------------------------------------------------------------------

# 2. CẤU HÌNH DATASET

# ------------------------------------------------------------------------------

# File .pkl cần có dạng mỗi sample gồm:

# keypoint:       [num_person, total_frames, 17, 2]

# keypoint_score: [num_person, total_frames, 17]

# label:          0 hoặc 1

dataset_type = 'PoseDataset'

data_root = 'data/processed/mmaction/'
ann_file_train = data_root + 'train_data.pkl'
ann_file_val = data_root + 'val_data.pkl'

# Danh sách keypoint trái/phải dùng cho phép lật ngang dữ liệu.

# Theo định dạng COCO 17 keypoints:

# 1/2: mắt, 3/4: tai, 5/6: vai, 7/8: khuỷu tay,

# 9/10: cổ tay, 11/12: hông, 13/14: gối, 15/16: cổ chân.

left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]

# ------------------------------------------------------------------------------

# 3. PIPELINE CHO TẬP TRAIN

# ------------------------------------------------------------------------------

train_pipeline = [
# Lấy 30 frame đại diện cho mỗi video/window.
dict(type='UniformSampleFrames', clip_len=30),
# Decode dữ liệu pose từ file .pkl.
dict(type='PoseDecode'),

# Crop vùng chứa người để giảm nhiễu nền.
dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),

# Resize trung gian trước khi random crop.
dict(type='Resize', scale=(56, 56)),

# Data augmentation.
dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),

# Resize về kích thước cuối cùng của heatmap.
dict(type='Resize', scale=(48, 48), keep_ratio=False),

# Lật ngang dữ liệu và hoán đổi keypoints trái/phải.
dict(
    type='Flip',
    flip_ratio=0.5,
    left_kp=left_kp,
    right_kp=right_kp
),

# Chuyển keypoint tọa độ sang 3D pseudo-heatmap.
dict(
    type='GeneratePoseTarget',
    sigma=0.6,
    use_score=True,
    with_kp=True,
    with_limb=False,
    double=False
),

# BẮT BUỘC:
# Định dạng lại heatmap thành [C, T, H, W] cho từng sample.
# Khi gom batch, input vào model sẽ là [B, C, T, H, W].
dict(type='FormatShape', input_format='NCTHW_Heatmap'),

# Đóng gói dữ liệu theo format đầu vào của MMAction2.
dict(type='PackActionInputs')

]

# ------------------------------------------------------------------------------

# 4. PIPELINE CHO TẬP VALIDATION / TEST

# ------------------------------------------------------------------------------

val_pipeline = [
dict(
type='UniformSampleFrames',
clip_len=30,
num_clips=1,
test_mode=True
),

dict(type='PoseDecode'),
dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
dict(type='Resize', scale=(48, 48), keep_ratio=False),

dict(
    type='GeneratePoseTarget',
    sigma=0.6,
    use_score=True,
    with_kp=True,
    with_limb=False,
    double=False
),

# BẮT BUỘC để tránh lỗi sai chiều tensor khi đưa vào Conv3D.
dict(type='FormatShape', input_format='NCTHW_Heatmap'),

dict(type='PackActionInputs')
]

# ------------------------------------------------------------------------------

# 5. DATALOADER

# ------------------------------------------------------------------------------

# batch_size=8 an toàn hơn trên Kaggle/Colab.

# Nếu GPU còn nhiều VRAM, có thể tăng lên 16.

train_dataloader = dict(
batch_size=8,
num_workers=2,
persistent_workers=True,

sampler=dict(
    type='DefaultSampler',
    shuffle=True
),

dataset=dict(
    type=dataset_type,
    ann_file=ann_file_train,
    pipeline=train_pipeline
)

)

val_dataloader = dict(
batch_size=8,
num_workers=2,
persistent_workers=True,

sampler=dict(
    type='DefaultSampler',
    shuffle=False
),

dataset=dict(
    type=dataset_type,
    ann_file=ann_file_val,
    pipeline=val_pipeline
)

)

test_dataloader = val_dataloader

# ------------------------------------------------------------------------------

# 6. EVALUATION METRIC

# ------------------------------------------------------------------------------

# Với bài toán 2 lớp, top5_acc không có nhiều ý nghĩa.

# Chỉ sử dụng top1 accuracy để đánh giá trong quá trình train.

val_evaluator = dict(
type='AccMetric',
metric_list=('top_k_accuracy',),
metric_options=dict(
top_k_accuracy=dict(topk=(1,))
)
)

test_evaluator = val_evaluator

# ------------------------------------------------------------------------------

# 7. TRAINING LOOP

# ------------------------------------------------------------------------------

# Fine-tuning từ checkpoint pretrained nên dùng số epoch vừa phải.

# Validation được thực hiện sau mỗi 2 epoch.

train_cfg = dict(
type='EpochBasedTrainLoop',
max_epochs=12,
val_interval=2
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ------------------------------------------------------------------------------

# 8. OPTIMIZER VÀ LEARNING RATE SCHEDULE

# ------------------------------------------------------------------------------

# Fine-tuning thường dùng learning rate nhỏ hơn train from scratch.

# Ở đây dùng AdamW với lr=1e-4.

optim_wrapper = dict(
type='OptimWrapper',
optimizer=dict(
type='AdamW',
lr=0.0001,
weight_decay=0.01
)
)

param_scheduler = [
dict(
type='CosineAnnealingLR',
T_max=12,
eta_min=0,
by_epoch=True
)
]

# ------------------------------------------------------------------------------

# 9. RUNTIME HOOKS

# ------------------------------------------------------------------------------

# Lưu checkpoint mỗi 2 epoch, giữ tối đa 3 checkpoint gần nhất.

# save_best='auto' sẽ tự chọn metric phù hợp để lưu checkpoint tốt nhất.

default_hooks = dict(
runtime_info=dict(type='RuntimeInfoHook'),

timer=dict(type='IterTimerHook'),

logger=dict(
    type='LoggerHook',
    interval=20,
    ignore_last=False
),

param_scheduler=dict(type='ParamSchedulerHook'),

checkpoint=dict(
    type='CheckpointHook',
    interval=2,
    save_best='auto',
    max_keep_ckpts=3
),

sampler_seed=dict(type='DistSamplerSeedHook'),

sync_buffers=dict(type='SyncBuffersHook')

)

# ------------------------------------------------------------------------------

# 10. ENVIRONMENT CONFIG

# ------------------------------------------------------------------------------

env_cfg = dict(
cudnn_benchmark=False,

mp_cfg=dict(
    mp_start_method='fork',
    opencv_num_threads=0
),

dist_cfg=dict(backend='nccl')

)

log_processor = dict(
type='LogProcessor',
window_size=20,
by_epoch=True
)

log_level = 'INFO'

# ------------------------------------------------------------------------------

# 11. FINE-TUNING CHECKPOINT

# ------------------------------------------------------------------------------

# Đây là checkpoint PoseC3D SlowOnly-R50 pretrained trên NTU60 XSub keypoint.

# Mô hình pretrained học biểu diễn hành động từ skeleton/keypoint trước,

# sau đó được fine-tune lại cho bài toán Fall Detection 2 lớp.

load_from = 'https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint_20220815-38db104b.pth'

# Không dùng resume khi fine-tune lần đầu.

# resume=True chỉ dùng khi muốn train tiếp từ checkpoint của chính thí nghiệm hiện tại.

resume = False

# Cố định seed để kết quả dễ tái lập hơn.

randomness = dict(
seed=42,
deterministic=False
)
