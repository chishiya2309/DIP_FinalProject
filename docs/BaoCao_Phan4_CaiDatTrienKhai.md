# PHẦN 4: CÀI ĐẶT VÀ TRIỂN KHAI HỆ THỐNG

## 4.1. Môi trường phát triển và công nghệ sử dụng

### 4.1.1. Môi trường phần cứng

| Thành phần                            | Cấu hình                                                     |
| ------------------------------------- | ------------------------------------------------------------ |
| **Máy phát triển (Dev)**              | CPU Intel/AMD, RAM ≥ 8GB, HDD/SSD                            |
| **Môi trường huấn luyện (Train)**     | Kaggle Notebook — GPU NVIDIA Tesla T4 x 2                    |
| **Môi trường triển khai (Inference)** | CPU đa luồng thông thường (không yêu cầu GPU)                |
| **Thiết bị thu nhận**                 | Webcam USB / Điện thoại (qua IP Webcam/DroidCam) / Camera IP |

### 4.1.2. Công nghệ phần mềm

| Thành phần      | Công nghệ        | Phiên bản       | Vai trò                                           |
| --------------- | ---------------- | --------------- | ------------------------------------------------- |
| Ngôn ngữ        | Python           | 3.10+           | Ngôn ngữ chính                                    |
| Xử lý ảnh       | OpenCV           | 4.x             | DIP preprocessing, đọc camera, vẽ annotation      |
| Pose Estimation | Ultralytics YOLO | 11.x (8.3+)     | Phát hiện người + ước lượng keypoints             |
| Deep Learning   | PyTorch          | 2.x             | Xây dựng và suy luận mạng PoseBiGRU               |
| Tracking        | ByteTrack        | (built-in YOLO) | Theo dõi đối tượng đa khung hình                  |
| Giao diện       | CustomTkinter    | 5.x             | Xây dựng giao diện Desktop hiện đại               |
| Đánh giá        | Scikit-learn     | 1.x             | Tính toán metrics: Accuracy, F1, Confusion Matrix |
| Thông báo       | Telegram Bot API | —               | Gửi cảnh báo qua HTTP                             |
| Dữ liệu         | NumPy, Pandas    | —               | Xử lý ma trận và dữ liệu bảng                     |

### 4.1.3. Cấu trúc thư mục dự án

```
FinalProject/
├── main.py                         # Điểm vào chương trình
├── best.pt                         # Trọng số PoseBiGRU đã huấn luyện
├── yolo11s-pose.pt                 # Trọng số YOLO-Pose pre-trained
├── telegram_config.json            # Cấu hình Telegram Bot
├── fall_history.csv                # File log lịch sử cảnh báo
├── requirements.txt                # Danh sách thư viện phụ thuộc
│
├── core/                           # Lõi xử lý (Backend)
│   ├── camera_manager.py           # Quản lý nguồn video + DIP
│   ├── inference_manager.py        # Điều phối luồng inference
│   ├── inference_worker.py         # YOLO + PoseBiGRU + Heuristic
│   ├── model.py                    # Kiến trúc mạng PoseBiGRU-Attention
│   ├── logger.py                   # Ghi log CSV + Snapshot
│   └── telegram_notifier.py        # Gửi cảnh báo Telegram
│
├── src/                            # Module xử lý ảnh
│   └── preprocessing/
│       ├── adaptive_pipeline.py    # Pipeline DIP thích nghi
│       ├── denoising.py            # Khử nhiễu Gaussian + NLMeans
│       ├── filters.py              # Bộ lọc không gian
│       ├── backlight.py            # Xử lý ngược sáng
│       ├── sharpening.py           # Tăng cường biên
│       ├── white_balance.py        # Cân bằng trắng Grey World
│       └── config_manager.py       # Quản lý cấu hình DIP
│
├── ui/                             # Giao diện người dùng (Frontend)
│   ├── main_app.py                 # Controller chính — điều hướng views
│   ├── components/
│   │   ├── sidebar.py              # Thanh menu bên trái
│   │   └── video_panel.py          # Component hiển thị video camera
│   └── views/
│       ├── login_view.py           # Màn hình đăng nhập
│       ├── admin_view.py           # Màn hình giám sát (Admin Monitor)
│       ├── admin_settings_view.py  # Màn hình cài đặt hệ thống
│       ├── client_view.py          # Màn hình giám sát (Client — đơn giản)
│       └── history_view.py         # Màn hình lịch sử cảnh báo + Snapshot
│
├── train_hung/                     # Script huấn luyện mô hình
│   ├── train.py                    # Script train chính (chạy trên Kaggle)
│   ├── model.py                    # Bản sao kiến trúc model cho training
│   ├── extract_urfall_yolo11pose.py
│   ├── kaggle_extract_multicam.py
│   └── merge_custom_datasets.py
│
├── scripts/                        # Công cụ hỗ trợ
│   ├── run_extraction.py           # Trích xuất skeleton từ video
│   ├── visualize_skeleton.py       # Trực quan hóa skeleton trên video
│   └── benchmark_enhancement.py    # Benchmark module DIP
│
├── data/                           # Dữ liệu (raw + processed)
├── logs/snapshots/                 # Ảnh chụp bằng chứng khi ngã
├── demo/                           # Video demo dùng để test
└── docs/                           # Tài liệu báo cáo
```

## 4.2. Xây dựng giao diện người dùng (GUI)

### 4.2.1. Kiến trúc MVC đơn giản

Giao diện sử dụng pattern MVC (Model-View-Controller) đơn giản:

- **Model:** `CameraManager`, `InferenceManager` — xử lý dữ liệu
- **View:** Các class trong `ui/views/` — hiển thị giao diện
- **Controller:** `MainApp` — điều phối chuyển đổi giữa các màn hình

`MainApp` kế thừa `ctk.CTk`, quản lý một dictionary `self.views` chứa tất cả các view. Phương thức `show_view(name)` thực hiện: dừng camera ở view cũ → nâng view mới lên trên (tkraise) → bật camera nếu cần → cập nhật sidebar.

### 4.2.2. Các màn hình chính

**Màn hình đăng nhập (LoginView):**

- Phân quyền 2 vai trò: Admin (có quyền truy cập Cài đặt) và Client (chỉ giám sát)
- Mật khẩu mặc định: `admin123` cho Admin, bất kỳ ký tự nào cho Client
- Giao diện tối giản, tập trung vào 2 ô nhập liệu

`[Placeholder: Chèn hình — Ảnh chụp màn hình Đăng nhập]`

**Màn hình giám sát (AdminView / ClientView):**

- Hiển thị video stream trực tiếp từ camera với annotation YOLO-Pose (bounding box, skeleton, Track ID)
- Dòng trạng thái lớn ở dưới: "HỆ THỐNG AN TOÀN" (xanh lá) hoặc "⚠ PHÁT HIỆN TÉ NGÃ (ID: X) ⚠" (đỏ)
- Khi phát hiện ngã: phát âm thanh `Beep(1000Hz, 500ms)` qua loa máy tính
- AdminView có toàn bộ chức năng; ClientView là phiên bản đơn giản hóa

`[Placeholder: Chèn hình — Ảnh chụp màn hình Giám sát Camera khi hệ thống an toàn]`

`[Placeholder: Chèn hình — Ảnh chụp màn hình Giám sát Camera khi phát hiện té ngã (khung đỏ)]`

**Màn hình cài đặt hệ thống (AdminSettingsView):**

- **Nguồn Camera:** Dropdown chọn Webcam, file video, hoặc nhập URL Camera IP (HTTP/RTSP)
- **Ngưỡng cảnh báo (Confidence Threshold):** Thanh trượt 0.0 – 1.0, điều chỉnh độ nhạy của PoseBiGRU
- **Thời gian Cooldown:** Dropdown 3/5/10/15 giây
- **Module DIP:** Bật/tắt tăng cường video, bật/tắt tự động thích nghi
- **Telegram:** Bật/tắt, nhập Bot Token và Chat ID
- Nút "Lưu cấu hình" và "Xuất Log CSV"

`[Placeholder: Chèn hình — Ảnh chụp màn hình Cài đặt Hệ thống]`

**Màn hình lịch sử cảnh báo (HistoryView):**

- Bảng dữ liệu (`ttk.Treeview`) hiển thị toàn bộ lịch sử cảnh báo từ file CSV, sắp xếp mới nhất lên đầu
- 5 cột: Thời gian, Nguồn Camera, ID Nạn nhân, Trạng thái, Đường dẫn ảnh
- Bên phải: Khung xem trước ảnh Snapshot — click vào dòng lịch sử sẽ hiển thị ảnh bằng chứng tương ứng
- Nút "Làm mới" để reload dữ liệu

`[Placeholder: Chèn hình — Ảnh chụp màn hình Lịch sử Cảnh báo với ảnh Snapshot]`

### 4.2.3. Thanh menu bên trái (Sidebar)

Sidebar cố định bên trái hiển thị:

- Logo và tên trường, tên môn học, tên đề tài
- Danh sách thành viên nhóm
- 3 nút điều hướng: Giám sát Camera, Cài đặt Hệ thống, Lịch sử Cảnh báo
- Nút Đăng xuất

Các nút được ẩn/hiện tùy theo vai trò: Admin thấy cả 3 nút, Client chỉ thấy nút Giám sát và Lịch sử.

## 4.3. Huấn luyện mô hình PoseBiGRU

### 4.3.1. Bộ dữ liệu

Mô hình được huấn luyện trên bộ dữ liệu kết hợp từ nhiều nguồn:

| Nguồn dữ liệu                      | Loại          | Số lượng video | Mô tả                                                                       |
| ---------------------------------- | ------------- | -------------- | --------------------------------------------------------------------------- |
| **NTU RGB+D 60 (ntu60_hrnet.pkl)** | Pre-trained   | 56,000+ clips  | Bộ dữ liệu hành động quy mô lớn dùng để tiền huấn luyện (transfer learning) |
| **UR Fall Detection Dataset**      | Chuẩn quốc tế | 70 clips       | 30 clips ngã + 40 clips ADL, multi-camera                                   |
| **Multiple Cameras Fall Dataset**  | Chuẩn quốc tế | 192 clips      | 24 kịch bản × 8 cameras, đa góc nhìn                                        |
| **Dữ liệu tự thu thập**            | Custom        | —              | Video nhóm tự quay trong phòng học, phòng khách                             |

### 4.3.2. Quy trình chuẩn bị dữ liệu (Data Pipeline)

1. **Trích xuất skeleton:** Chạy video qua YOLOv11s-Pose trên Kaggle GPU → xuất tọa độ 17 keypoints cho mỗi frame → lưu thành file `.pkl` (pickle) dạng MMAction2 format.
2. **Cắt chuỗi (Sliding Window):** Chuỗi skeleton dài được cắt thành các clip có độ dài cố định 72 frames (≈ 3 giây ở 24 FPS). Sử dụng stride = 1 cho maximum overlap.
3. **Gán nhãn nhị phân:** Mỗi clip được gán nhãn FALL (1) hoặc NON-FALL (0).
4. **Tăng cường dữ liệu (Data Augmentation):** Random temporal cropping, speed jittering (thay đổi tốc độ phát), random horizontal flip.
5. **Cân bằng dữ liệu:** Sử dụng `WeightedRandomSampler` với `fall_weight = 1.25` để bù đắp sự mất cân bằng giữa 2 lớp (clip Non-Fall thường nhiều hơn clip Fall).

### 4.3.3. Siêu tham số huấn luyện

| Tham số                 | Giá trị                                    |
| ----------------------- | ------------------------------------------ |
| Optimizer               | AdamW                                      |
| Learning Rate           | 1e-3                                       |
| Weight Decay            | 1e-4                                       |
| Batch Size              | 128                                        |
| Epochs                  | 50                                         |
| Sequence Length         | 72 frames                                  |
| Hidden Size (GRU)       | 128                                        |
| Num GRU Layers          | 2                                          |
| Dropout                 | 0.3                                        |
| Gradient Clipping       | 1.0                                        |
| Early Stopping Patience | 8 epochs                                   |
| Loss Function           | CrossEntropyLoss (weighted)                |
| LR Scheduler            | ReduceLROnPlateau (factor=0.5, patience=3) |

### 4.3.4. Môi trường huấn luyện

Quá trình huấn luyện được thực hiện trên **Kaggle Notebook** với GPU NVIDIA Tesla T4 x 2:

1. Upload toàn bộ dữ liệu skeleton (`.pkl`) lên Kaggle Dataset
2. Upload script `train.py` và `model.py` lên Kaggle Notebook
3. Chạy training với lệnh: `python train.py --dataset-source combined --epochs 50 --balanced-sampler`
4. Sau khi train xong, download file `best.pt` (≈ 9MB) về máy local đặt vào thư mục gốc của dự án

`[Placeholder: Chèn hình — Ảnh chụp Kaggle Notebook đang train (output log, loss curve)]`

## 4.4. Kết nối Camera IP từ điện thoại

Hệ thống hỗ trợ biến điện thoại thành Camera IP — phù hợp cho việc demo bảo vệ đồ án:

1. Cài ứng dụng **IP Webcam** (Android) hoặc **DroidCam** (Android/iOS) trên điện thoại
2. Kết nối cùng mạng WiFi với máy tính
3. Bật server trên ứng dụng → ghi nhận địa chỉ IP (ví dụ: `http://192.168.1.15:8080`)
4. Trên phần mềm: Cài đặt → Nguồn Camera → Nhập URL Camera IP → Nhập `http://192.168.1.15:8080/video` → Lưu cấu hình

OpenCV (`cv2.VideoCapture`) hỗ trợ sẵn đọc HTTP video stream và RTSP protocol, không cần cài thêm thư viện.

`[Placeholder: Chèn hình — Ảnh chụp điện thoại đang chạy IP Webcam + ảnh laptop nhận stream]`
