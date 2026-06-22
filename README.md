# 🚨 Hệ Thống Cảnh Báo Té Ngã Tự Động Cho Người Cao Tuổi

> **Đồ án cuối kỳ - Xử Lý Ảnh Số (DIPR430685_06CLC)**
> **Nhóm 05** | GVHD: PGS.TS. Hoàng Văn Dũng | HK2 2025-2026

## Tổng quan

Hệ thống giám sát tự động không xâm nhập dựa trên Computer Vision, phát hiện và cảnh báo té ngã theo thời gian thực cho người cao tuổi trong môi trường trong nhà.

### Pipeline xử lý

```
Video Input → DIP Preprocessing → YOLOv8-Pose → Sliding Window → PoseC3D → Alert
```

## Cấu trúc thư mục

```
FinalProject/
├── configs/                    # File cấu hình (hyperparams, model config, paths)
├── data/
│   ├── raw/                    # Video gốc chưa xử lý
│   │   ├── ur_fall_detection/  # Dataset UR Fall Detection
│   │   ├── multiple_cameras_fall/ # Dataset Multiple Cameras Fall
│   │   └── custom_recordings/  # Video tự quay
│   │       ├── fall/           # Mẫu té ngã (positive)
│   │       └── non_fall/       # Mẫu sinh hoạt bình thường + hard negatives
│   ├── processed/              # Dữ liệu sau xử lý
│   │   ├── keypoints/          # Tọa độ 17 keypoints từ YOLOv8-Pose
│   │   ├── skeleton_sequences/ # Chuỗi skeleton sau sliding window
│   │   └── heatmaps_3d/        # 3D Pseudo-heatmaps cho PoseC3D
│   └── splits/                 # Train/Val/Test split files
├── demo/                       # Video demo & tài liệu trình bày
├── docs/
│   ├── figures/                # Hình ảnh cho báo cáo
│   └── references/             # Tài liệu tham khảo
├── models/
│   ├── pretrained/             # Trọng số pre-trained (YOLOv8-Pose, PoseC3D)
│   └── checkpoints/            # Model checkpoints sau fine-tuning
├── notebooks/                  # Jupyter notebooks (EDA, thử nghiệm)
├── outputs/
│   ├── logs/                   # Training logs, experiment logs
│   ├── metrics/                # Kết quả đánh giá (confusion matrix, F1, FPS)
│   └── visualizations/         # Ảnh/video kết quả trực quan
├── scripts/                    # Scripts tiện ích (download data, convert, etc.)
├── src/                        # Source code chính
│   ├── preprocessing/          # Module tiền xử lý ảnh (Gaussian Blur, CLAHE)
│   ├── pose_estimation/        # Module YOLOv8-Pose wrapper
│   ├── sliding_window/         # Module cửa sổ trượt & skeleton sequences
│   ├── posec3d/                # Module PoseC3D (heatmap generation, model, training)
│   ├── alerting/               # Module cảnh báo thời gian thực
│   ├── pipeline/               # Pipeline tích hợp end-to-end
│   └── utils/                  # Hàm tiện ích dùng chung
├── tests/                      # Unit tests
├── Nhom5.md                    # Đề cương đồ án chi tiết
├── requirements.txt            # Python dependencies
└── .gitignore
```

## Thành viên

| MSSV     | Họ tên                  |
| -------- | ----------------------- |
| 23110078 | Nguyễn Thái Bảo         |
| 23110110 | Lê Quang Hưng           |
| 23110111 | Lương Nguyễn Thành Hưng |

## Công nghệ

- **Ngôn ngữ:** Python
- **Xử lý ảnh:** OpenCV (Gaussian Blur, CLAHE)
- **Pose Estimation:** Ultralytics YOLOv8-Pose
- **Deep Learning:** PyTorch, PoseC3D
- **Đánh giá:** NumPy, Scikit-learn, Matplotlib

## Kế hoạch thực hiện

| Tuần | Nội dung                                  |
| ---- | ----------------------------------------- |
| 1-2  | Nghiên cứu tổng quan & Thu thập dữ liệu   |
| 3-4  | Tiền xử lý dữ liệu & Trích xuất đặc trưng |
| 5-6  | Xây dựng mô hình dữ liệu chuỗi thời gian  |
| 7-8  | Huấn luyện tinh chỉnh & Thực nghiệm       |
| 9-10 | Tổng kết, Đánh giá & Viết báo cáo         |
