# Hướng Dẫn Thu Thập Và Tổ Chức Dữ Liệu Video (Tuần 1-2)

Tài liệu này cung cấp hướng dẫn chi tiết các bước thực hiện để hoàn thành mục tiêu thu thập dữ liệu cho hệ thống **Cảnh báo té ngã tự động cho người cao tuổi**.

Mục tiêu: Xây dựng một bộ dữ liệu video phong phú, đa dạng, phản ánh sát với thực tế, bao gồm cả dữ liệu chuẩn từ cộng đồng nghiên cứu và dữ liệu tự quay để tăng cường độ khó cho mô hình (đặc biệt là các mẫu âm tính khó - hard negative samples).

---

## Phần 1: Thu Thập Dữ Liệu Từ Các Nguồn Chuẩn (Public Datasets)

Việc sử dụng các dataset chuẩn giúp thiết lập một baseline vững chắc cho mô hình và dễ dàng so sánh hiệu năng với các nghiên cứu khác.

### 1. UR Fall Detection Dataset

Đây là bộ dữ liệu phổ biến chứa cả video quay cảnh ngã và các hoạt động sinh hoạt bình thường.

- **Nguồn gốc:** Đại học Rzeszow.
- **Thành phần:** Chứa các video mô phỏng té ngã (falls) và các hoạt động hàng ngày (ADLs - Activities of Daily Living) như ngồi, nằm, cúi gập người.
- **Cách thực hiện:**
  1.  Chạy script tải tự động: Bạn chỉ cần mở Terminal và chạy lệnh `python scripts/download_ur_fall.py`.
  2.  Script này sẽ tự động tải toàn bộ 30 video té ngã và 40 video sinh hoạt (từ cả 2 góc camera) về thẳng thư mục dự án của chúng ta.
  3.  Lưu trữ: Dữ liệu sẽ được tự động lưu vào `data/raw/ur_fall_detection/`

### 2. Multiple Cameras Fall Dataset

Bộ dữ liệu này đặc biệt hữu ích vì ghi lại cùng một kịch bản té ngã từ nhiều góc camera khác nhau, giúp mô hình học được sự bất biến về góc nhìn (viewpoint invariance).

- **Nguồn gốc:** Viện nghiên cứu ViSion Lab.
- **Thành phần:** Các kịch bản té ngã và sinh hoạt được quay đồng thời từ 8 camera đặt ở các góc khác nhau trong phòng.
- **Cách thực hiện:**
  1.  Tìm kiếm "Multiple Cameras Fall Dataset" để tải về.
  2.  Trích xuất các video từ các góc camera phản ánh gần nhất với góc nhìn của camera giám sát trong nhà (thường là góc từ trên cao hướng xuống, camera gắn tường).
  3.  Lưu trữ vào thư mục: `data/raw/multiple_cameras_fall/`

---

## Phần 2: Thu Thập Dữ Liệu Tự Quay (Custom Recordings)

Dữ liệu chuẩn thường không đủ độ "khó" (thiếu đồ đạc che khuất, ánh sáng quá tốt, ít hành động gây nhầm lẫn). Việc tự quay giúp chúng ta kiểm soát các yếu tố này và rèn luyện mô hình chống lại hiện tượng "báo động giả" (False Alarms).

### 1. Chuẩn bị Môi trường và Thiết bị

- **Thiết bị:** Smartphone gắn trên Tripod hoặc Camera giám sát IP thông thường. Độ phân giải tối thiểu 720p.
- **Góc máy (Camera Angle):** Đặt camera ở độ cao từ 2.0m - 2.5m, chĩa góc nghiêng xuống (góc nhìn chuẩn của camera an ninh trong nhà).
- **Môi trường (Bối cảnh):** Phòng khách, phòng ngủ. Cố ý bố trí các vật cản như ghế sofa, bàn làm việc, tủ giường để tạo ra tình huống che khuất (occlusion).
- **Ánh sáng:** Quay ở nhiều điều kiện khác nhau (sáng ban ngày, đèn tuýp buổi tối, hơi tối - thiếu sáng).

### 2. Kịch bản Té Ngã Thực Tế (Positive Samples - FALL)

Cần mô phỏng các kiểu ngã phổ biến nhất ở người cao tuổi:

1.  **Ngã chúi về phía trước:** Vấp phải vật cản trên thảm/sàn.
2.  **Ngã ngửa về phía sau:** Trượt chân do sàn ướt hoặc mất thăng bằng.
3.  **Ngã ngang sang hai bên:** Chóng mặt, khụy gối từ từ rồi ngã sang hông.
4.  **Ngã từ vị trí đang ngồi/nằm:** Đang ngồi trên ghế/giường cố đứng lên nhưng mất thăng bằng và ngã xuống sàn.

_Lưu ý an toàn: Đảm bảo sử dụng thảm xốp dày (yoga mat) hoặc nệm khi quay các cảnh này để tránh chấn thương cho các thành viên trong nhóm._

### 3. Kịch bản Âm Tính Khó (Hard Negative Samples - NON-FALL)

Đây là phần QUAN TRỌNG NHẤT của dữ liệu tự quay. Đây là các hành động sinh hoạt có **quỹ đạo trọng tâm hạ thấp đột ngột hoặc quỹ đạo giống hệt việc té ngã**, dễ làm mô hình (và cả YOLO) nhầm lẫn.

1.  **Ngồi phịch (thụp) xuống:** Đang đứng và ngồi phịch xuống ghế sofa hoặc ghế đẩu (chuyển động đi xuống rất nhanh).
2.  **Cúi gập người sâu:** Cúi gập người hẳn xuống sàn để thắt dây giày hoặc nhặt một vật bị rơi (chìa khóa, bút).
3.  **Chủ động nằm ra sàn/giường:** Đi tới giường/sofa và thả người nằm ngả ra.
4.  **Tập thể dục:** Các động tác như chống đẩy (push-ups), plank, squat (nếu có thể).
5.  **Ngồi xổm (Squatting):** Đứng và đột ngột ngồi xổm xuống để lấy đồ trong tủ thấp.

---

## Phần 3: Tổ Chức Và Lưu Trữ

Sau khi đã có đầy đủ video, thực hiện phân loại và đổi tên file theo một quy tắc thống nhất để dễ dàng cho việc viết script xử lý sau này.

### Quy cách đặt tên file (Naming Convention)

Nên đặt tên theo cú pháp: `<nguồn>_<nhãn>_<môi trường>_<sốthứtự>.mp4`

Ví dụ:

- `custom_fall_phongkhach_01.mp4`
- `custom_nonfall_nhatdo_02.mp4`

### Cấu trúc lưu trữ

Di chuyển toàn bộ các file đã quay vào các thư mục tương ứng đã tạo trong project:

```text
FinalProject/data/raw/
├── ur_fall_detection/
│   ├── fall-01-cam0.mp4
│   └── adl-01-cam0.mp4
├── multiple_cameras_fall/
│   ├── chute01_cam1.avi
│   └── ...
└── custom_recordings/
    ├── fall/
    │   ├── custom_fall_phongkhach_01.mp4
    │   └── custom_fall_truotchan_02.mp4
    └── non_fall/
        ├── custom_nonfall_ngoipho_01.mp4
        ├── custom_nonfall_nhatdo_01.mp4
        └── custom_nonfall_namgiuong_01.mp4
```

### Bước tiếp theo (Tuần 3-4)

Sau khi thư mục `data/raw/` đã được lấp đầy, chúng ta sẽ viết script Python sử dụng thư viện `OpenCV` để đọc các video này, áp dụng bộ lọc DIP (Gaussian, CLAHE) và đưa qua `YOLOv8-Pose` để trích xuất tọa độ `keypoints` lưu vào thư mục `data/processed/`.
