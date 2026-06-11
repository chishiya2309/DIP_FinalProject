# Kế hoạch Tích hợp Model PoseBiGRU vào UI (Bản Cập Nhật)

Tài liệu này trình bày các bước chi tiết (To-do List) đã được chuẩn hóa kiến trúc để tích hợp mô hình PoseBiGRU vào giao diện. Mục tiêu tối thượng là: **Kết quả predict chính xác tuyệt đối như lúc train, không block UI, không memory leak và quản lý đa luồng (multi-threading) an toàn.**

## 0. Bước đệm quan trọng (Khuyến nghị)
- [ ] Tạo một script nhỏ (ví dụ: `test_worker.py`) để chạy thử `FallDetectorWorker` độc lập trên một video `.mp4`. In xác suất `prob_fall` theo từng frame ra console và so sánh với kết quả gốc của `train_hung`. Đảm bảo model dự đoán chuẩn trước khi gắn vào UI.

## 1. Thiết lập Core Inference Worker (`core/inference_worker.py`)
- [ ] Tạo file `core/model.py` (copy nguyên kiến trúc từ `train_hung/model.py`).
- [ ] Xây dựng class `FallDetectorWorker` chạy trên một luồng nền:
  - [ ] Khởi tạo YOLOv8-pose ở chế độ `track` (để lấy `track_id` ổn định).
  - [ ] Khởi tạo PoseBiGRU và load `best.pt`.
  - [ ] **Quản lý bộ nhớ theo người:** Sử dụng `Dict[track_id, collections.deque(maxlen=72)]` để lưu chuỗi frame của từng người (với `sequence_length=72` tương đương khi train).
  - [ ] **Cleanup cơ chế:** Nếu một `track_id` biến mất khỏi màn hình quá N frames (do ra khỏi cam hoặc occlusion), phải chủ động xóa entry khỏi Dict để tránh Memory Leak.
  - [ ] **Tiền xử lý nghiêm ngặt (Critical):** Phải copy nguyên hàm `fit_sequence()` từ `train_hung/train.py` sang worker. Khi một `track_id` có đủ 72 frames (chấp nhận trễ ~2s ban đầu), mới đưa qua `fit_sequence()` để tạo mask, timestamp và tensor chuẩn trước khi đưa vào `model(keypoints, mask, timestamps)`. Infer tuần tự (batch=1) cho từng `track_id`.

## 2. Đồng bộ hiển thị UI và Kết quả YOLO
- [ ] **Không đợi YOLO:** Thread Camera (`CameraManager`) đọc frame ở tốc độ cao (30fps) và gửi ngay cho UI hiển thị.
- [ ] **Cập nhật Bbox bất đồng bộ:** Lưu kết quả Bbox & Keypoints mới nhất của YOLO vào một `shared_state` dictionary (có dùng `threading.Lock()` bảo vệ). Giao diện UI chỉ cần lấy kết quả từ `shared_state` này để vẽ đè lên hình ảnh. (Chấp nhận bounding box trễ 1-2 frames so với hình ảnh thực, mắt người không nhận ra nhưng video vẫn mượt 30fps).
- [ ] Vẽ trạng thái độc lập: Chỉ tô viền Bounding Box màu Đỏ cho `track_id` nào đang có `fall_state = True`, các `track_id` khác vẫn viền Xanh bình thường.

## 3. Liên kết Cấu hình và Quản lý Worker
- [ ] **Đưa Cooldown vào Worker:** Xử lý logic Cooldown (chờ N giây sau khi ngã mới báo tiếp) ngay bên trong `FallDetectorWorker` để giữ nghiệp vụ tập trung. Chỉ set `fall_state = True` khi đã qua thời gian Cooldown.
- [ ] Lấy `Fall Confidence Threshold` từ UI (vd: 0.70) truyền trực tiếp vào Worker.
- [ ] **Quản lý Vòng đời Worker (Critical):** Khi Admin thay đổi Camera Source, **BẮT BUỘC** gọi `worker.stop_event.set()` và `worker.thread.join()` để dọn dẹp sạch sẽ luồng và bộ nhớ của worker cũ, giải phóng VRAM/RAM, rồi mới khởi tạo Worker mới.
- [ ] Hệ thống hiện tại giả định chạy 1 Camera tại 1 thời điểm (Single Stream). Camera ID truyền vào History log chính là Source string/ID hiện tại.

## 4. Xử lý Sự kiện Cảnh báo (Alert Events) trên UI
- [ ] Hàm `update_frame()` của UI chỉ cần đọc cờ `has_fall` tổng (hoặc danh sách `falling_track_ids`).
- [ ] Nếu có người ngã:
  - [ ] Chuyển đổi nhãn trạng thái từ `[HỆ THỐNG AN TOÀN]` sang `[⚠ PHÁT HIỆN TÉ NGÃ ⚠]`.
  - [ ] **Non-blocking Audio:** Gọi hàm phát âm thanh bíp không chặn (Sử dụng `winsound.PlaySound("alert.wav", winsound.SND_ASYNC)` trên Windows hoặc đưa vào một daemon thread rời, hoặc dùng `pygame.mixer`).
  - [ ] Đẩy sự kiện ghi log vào một Queue (LogQueue). Có một luồng phụ (hoặc hàm gom) xử lý việc ghi file `fall_history.csv` tuần tự, tránh tranh chấp file lock với Windows khi UI đang đọc file.

## 5. Hoàn thiện tính năng Lịch sử & Báo cáo (History & Export)
- [ ] Thay vì auto-reload bảng History (dễ dính file lock), thêm một nút **"Làm mới (Refresh)"** trên `AdminView` để Admin tự bấm lấy dữ liệu CSV mới nhất.
- [ ] Nút "Xuất Log CSV" sử dụng `tkinter.filedialog.asksaveasfilename()` để hỏi người dùng vị trí lưu an toàn, thay vì fix cứng đường dẫn Desktop (dễ gặp lỗi do OneDrive redirect).
- [ ] Thay thư viện `logging` ghi file thông thường bằng `logging.handlers.RotatingFileHandler` để file `app.log` không bị phình to vô hạn khi hệ thống chạy giám sát liên tục.
