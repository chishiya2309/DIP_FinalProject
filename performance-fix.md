# Performance Fix - Hệ thống cảnh báo té ngã

## Goal
Đảm bảo **tốc độ load nhanh** và giảm độ trễ/giật khi chạy real-time trên máy CPU-only (`torch.cuda.is_available() = False`).

**Targets:**
- App startup (login screen visible): **< 2 giây**
- Inference FPS ổn định: **10–15 FPS** (CPU-only threshold)
- UI render mượt: **≥ 25 FPS** (display thread độc lập)

---

## Root Cause (đã điều tra)

| # | Nguyên nhân | Impact |
|---|---|---|
| 🔴 | `MainApp.__init__` load YOLO + PoseBiGRU ngay lúc khởi động → **block UI thread** | **Startup chậm** |
| 🔴 | YOLO `track()` chạy `imgsz=960` trên CPU mỗi frame | **FPS thấp** |
| 🟡 | Inference loop không throttle (cap 100fps), YOLO + BiGRU mỗi frame | CPU 100% |
| 🟡 | `enhance_frame()` (CLAHE + bilateral + sharpen + WB) mỗi frame | Camera lag |
| 🟢 | Nhiều `frame.copy()` + `cvtColor` RGB↔BGR lặp | Overhead nhỏ |

---

## Priority Matrix

```
            High Impact
                ↑
  [Task 0] ──── [Task 1] ──── [Task 2]
  Measure    Fast Startup    Reduce imgsz
  Baseline    (lazy-load)
                ↓
  [Task 5]       [Task 3]       [Task 6]
  Throttle    Reduce copy    ONNX Export
  FPS         + enhance        (optional)
                ↓
            Low Impact
```

---

## Tasks

### ✅ Phase 0 – Đo baseline TRƯỚC khi fix (Bắt buộc)

- [x] **Task 0**: Thêm log đo thời gian startup và FPS *trước khi* thay đổi bất cứ gì.
  - Đo: thời gian từ `MainApp.__init__` → login screen xuất hiện (ms)
  - Đo: camera FPS + inference FPS sau 10 giây chạy
  - Lưu số liệu baseline vào `performance-fix.md` phần "Baseline Results"
  - **Verify**: log hiển thị đủ 3 metrics (startup_ms, cam_fps, infer_fps)

---

### 🚀 Phase 1 – Load Speed (Startup nhanh) ← ƯU TIÊN CAO NHẤT

- [x] **Task 1**: **Lazy-load InferenceManager** – không load model lúc `__init__`, chỉ load khi user vào màn hình giám sát.

  **Sub-steps:**
  1. Tách `InferenceManager` init ra khỏi `MainApp.__init__` → đặt trong hàm `_on_enter_monitor_view()`
  2. Thêm **loading state** cho nút/view giám sát:
     - Hiển thị spinner / progress bar "Đang tải model..." trong khi load
     - Disable nút camera cho đến khi model sẵn sàng
  3. Load model trong **background thread** (`threading.Thread`) để không block UI
  4. Callback khi xong: enable camera, ẩn spinner

  **Verify**: App mở < 2s (login hiện ngay), spinner xuất hiện khi vào monitor, camera tự bật sau khi model load xong.

---

### ⚡ Phase 2 – Inference Speed (Giảm lag khi chạy)

- [x] **Task 2**: **Giảm `imgsz`** YOLO từ 960 → 640 (thử trước), nếu accuracy đủ thì 480.
  - File: `core/inference_worker.py::process_frame`
  - **Verify**: in FPS log, so sánh trước/sau (expect +50–100% FPS)

- [x] **Task 3**: **Throttle inference loop** – chạy YOLO mỗi 2–3 frame, giới hạn target **12 FPS** inference.
  - File: `core/inference_manager.py::_inference_loop`
  - Dùng `time.sleep()` hoặc token bucket để throttle
  - Display thread vẫn chạy 25+ FPS (hiện frame gần nhất)
  - **Verify**: CPU usage giảm, video display không giật

---

### 🔧 Phase 3 – Reduce Overhead

- [ ] **Task 4**: **Cache/throttle `enhance_frame`** – phân tích điều kiện sáng/tối mỗi N=5 frame (cache quyết định filter), thay bilateral filter bằng `cv2.GaussianBlur` khi ánh sáng đủ tốt.
  - **Verify**: thời gian/frame của camera thread giảm ≥ 20%

- [ ] **Task 5**: **Giảm copy/convert thừa** – gộp `cvtColor` và loại bỏ `.copy()` không cần thiết giữa camera → inference → UI pipeline.
  - **Verify**: code chạy đúng, frame hiển thị bình thường

---

### 🎯 Phase 4 – Advanced (Tùy chọn)

- [ ] **Task 6**: Export YOLO sang **ONNX + OpenVINO** cho CPU inference. Expect: 2–3x speedup so với PyTorch CPU.
  - **Verify**: latency/frame giảm, accuracy detection không đổi đáng kể

---

## Baseline Results (điền sau Task 0)

| Metric | Before | After |
|--------|--------|-------|
| Startup time (ms) | TBD | TBD |
| Camera FPS | TBD | TBD |
| Inference FPS | TBD | TBD |
| CPU usage (avg %) | TBD | TBD |

---

## Done When

- [ ] Startup: **login screen visible < 2 giây** (không treo đợi model)
- [ ] Loading indicator hiển thị khi model đang load, camera enable sau khi sẵn sàng
- [ ] Inference FPS: **≥ 10 FPS** stable, CPU không liên tục 100%
- [ ] Display FPS: **≥ 25 FPS** (UI mượt, không giật)
- [ ] Baseline Results table đã điền đủ Before/After

---

## Execution Order

```
Task 0 (measure) → Task 1 (lazy-load) → Task 2 (imgsz) → Task 3 (throttle) → Task 4 → Task 5 → [Task 6]
```

**Lý do thứ tự:**
- Task 0 trước: có data để verify improvement sau này
- Task 1 trước Task 2: startup fix có UX impact cao nhất, rủi ro thấp
- Task 2+3 trước 4+5: impact cao, dễ rollback nếu cần
- Task 6 sau cùng: cần effort cao hơn, chỉ làm nếu 1–5 chưa đủ

---

## Notes

- CPU-only → YOLO @960 là thủ phạm số 1 cho FPS; lazy-load là thủ phạm số 1 cho startup.
- Nếu sau này có GPU CUDA: giữ `imgsz=960`, bật `half=True`, bỏ throttle, bỏ lazy-load hoặc pre-warm.
- Mỗi task: đo trước → fix → đo sau → ghi vào Baseline Results table.
