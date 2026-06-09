# Kết quả Skeleton Visualization - Verify Pipeline Tuần 3-4

## Tổng quan

Pipeline đã xử lý thành công **5 video mẫu**, tạo ra **25 ảnh visualization** tại `outputs/skeleton_viz/`.

Mỗi ảnh gồm **3 panel so sánh**: Original | Gaussian + CLAHE | Skeleton Overlay

---

## Kết quả theo Video

### 1. fall-01-cam0 (FALL) — Frame #80: Trước khi ngã

![fall-01 frame 80 - Người đang đứng, skeleton detect đúng toàn bộ 17 keypoints](C:/Users/lequa/.gemini/antigravity/brain/828c9fce-baf3-40ac-8aa1-d9a2b129aa66/fall01_frame80.png)

> [!TIP]
> **Nhận xét:** Skeleton detect chính xác — nhìn rõ đầu, vai, khuỷu tay, hông, đầu gối, mắt cá. Có 2 người trong scene, cả 2 đều được detect riêng biệt. CLAHE cải thiện rõ rệt vùng tối bên trái.

### 2. fall-01-cam0 (FALL) — Frame #120: Sau khi ngã

![fall-01 frame 120 - Người đã ngã, skeleton vẫn detect được dù tư thế nằm](C:/Users/lequa/.gemini/antigravity/brain/828c9fce-baf3-40ac-8aa1-d9a2b129aa66/fall01_frame120.png)

> [!IMPORTANT]
> **Nhận xét:** Đây là frame quan trọng — người đã ngã và nằm trên sàn. YOLO vẫn detect được skeleton dù confidence thấp hơn (0.63). Keypoints cho tư thế nằm vẫn hợp lý. Đây là minh chứng pipeline hoạt động đúng cho cả tình huống ngã.

### 3. adl-01-cam0 (ADL) — Frame #75: Hoạt động bình thường

![adl-01 frame 75 - Hoạt động bình thường, skeleton chính xác](C:/Users/lequa/.gemini/antigravity/brain/828c9fce-baf3-40ac-8aa1-d9a2b129aa66/adl01_frame75.png)

> [!TIP]
> **Nhận xét:** ADL scene — người đang đi bình thường. Skeleton chính xác, bbox confidence cao. Hiệu quả CLAHE rõ ràng: ảnh gốc khá tối (phòng ngủ), sau CLAHE chi tiết rõ hơn hẳn.

### 4. fall-10-cam0 (FALL) — Frame #65: Đang trong quá trình ngã

![fall-10 frame 65 - Người đang ngã, tư thế cúi/gập](C:/Users/lequa/.gemini/antigravity/brain/828c9fce-baf3-40ac-8aa1-d9a2b129aa66/fall10_frame65.png)

> [!NOTE]
> **Nhận xét:** Người đang trong quá trình ngã (tư thế cúi gập). YOLO detect confidence 0.79. Skeleton cho thấy phần thân trên gập xuống — đây chính là pattern mà PoseC3D sẽ học để phân loại.

---

## Đánh giá tổng thể

| Tiêu chí | Kết quả | Ghi chú |
|----------|---------|---------|
| Keypoints chính xác | ✅ Tốt | 17 joints map đúng vị trí trên cơ thể |
| Detect tư thế ngã | ✅ Hoạt động | Conf thấp hơn (~0.63) nhưng vẫn detect được |
| Hiệu quả CLAHE | ✅ Rõ rệt | Cải thiện rõ vùng tối, giúp detect tốt hơn |
| Edge case cam1 | ⚠️ Yếu | cam1 là depth camera → YOLO khó detect (13.8%) |
| Multi-person | ✅ Xử lý tốt | Tự detect nhiều người, mỗi người có skeleton riêng |

> [!IMPORTANT]
> **Kết luận:** Pipeline Tuần 3-4 **PASS verification**. Keypoints chính xác, preprocessing hiệu quả, edge cases xử lý hợp lý. Sẵn sàng chạy batch toàn bộ UR Fall dataset và chuyển sang Tuần 5-6 (Sliding Window + PoseC3D).
