# PHẦN 2: CƠ SỞ LÝ THUYẾT

## 2.1. Tổng quan về xử lý ảnh số và tiền xử lý

Tiền xử lý ảnh (Preprocessing) là giai đoạn nền tảng đối với hiệu năng của bất kỳ hệ thống thị giác máy tính nào. Quy trình xử lý ảnh cơ bản bao gồm: Thu nhận ảnh (Image Acquisition), Nâng cao chất lượng ảnh (Image Enhancement), Phân vùng ảnh (Segmentation) và Trích xuất đặc trưng (Feature Extraction) [6].

### 2.1.1. Kỹ thuật lọc và khử nhiễu trong miền không gian

Kỹ thuật xử lý trong miền không gian (Spatial Domain) can thiệp trực tiếp giá trị của từng pixel trên ma trận ảnh, biểu diễn qua $g(x,y) = T[f(x,y)]$ [7]. Bộ lọc Gaussian được sử dụng trong hệ thống, hoạt động dựa trên nguyên lý tính trung bình có trọng số của các pixel lân cận, giúp khử nhiễu tần số cao mà vẫn bảo toàn được đường biên cấu trúc cơ thể người — điều kiện tiên quyết để YOLO nội suy chính xác các keypoints.

### 2.1.2. Cân bằng sáng thích nghi (CLAHE)

Trong môi trường nhà ở, ánh sáng thường bất ổn (ngược sáng cửa sổ, góc khuất tạo bóng, ánh sáng yếu ban đêm). Kỹ thuật CLAHE (Contrast Limited Adaptive Histogram Equalization) được áp dụng — là phiên bản cải tiến của Histogram Equalization, chia ảnh thành các ô nhỏ (tiles) và cân bằng histogram cục bộ trên từng ô, đồng thời giới hạn biên độ tăng cường (clip limit) để tránh khuếch đại nhiễu. Phương pháp này giúp khôi phục chi tiết vùng tối mà không làm quá sáng các vùng đã đủ sáng [7].

## 2.2. Mô hình YOLO-Pose cho ước lượng tư thế người

### 2.2.1. Tổng quan bài toán Pose Estimation

Ước lượng tư thế người (Human Pose Estimation) là bài toán xác định vị trí các điểm mốc giải phẫu (keypoints) trên cơ thể từ ảnh 2D. Hệ thống sử dụng tiêu chuẩn COCO-17 Keypoints gồm 17 điểm: mũi, mắt (trái/phải), tai (trái/phải), vai (trái/phải), khuỷu tay (trái/phải), cổ tay (trái/phải), hông (trái/phải), đầu gối (trái/phải), mắt cá chân (trái/phải). Mỗi keypoint được biểu diễn bởi bộ ba $(x_i, y_i, c_i)$ — tọa độ 2D và độ tin cậy.

### 2.2.2. Kiến trúc YOLOv11s-Pose

Hệ thống sử dụng mô hình YOLOv11s-Pose (bản Small) thay vì bản Nano, vì bản Small có khả năng nhận diện người ở tư thế nằm chính xác hơn do số lượng tham số lớn hơn. Mô hình này thực hiện đồng thời hai tác vụ: (1) Phát hiện người (Object Detection) bằng bounding box, và (2) Ước lượng tọa độ 17 keypoints cho mỗi người phát hiện được. Bộ trọng số sử dụng đã được huấn luyện trước trên tập COCO và được đóng băng (freeze) — không cần huấn luyện lại.

Kết hợp với thuật toán theo dõi ByteTrack, hệ thống duy trì định danh (Track ID) cho mỗi người xuyên suốt video, cho phép phân tích trạng thái tư thế theo thời gian cho từng cá nhân riêng biệt.

## 2.3. Mạng hồi quy hai chiều PoseBiGRU-Attention

### 2.3.1. Tầm quan trọng của phân tích chuỗi thời gian

Hành vi té ngã là một quá trình biến thiên liên tục — không thể phán đoán chính xác chỉ từ một khung hình tĩnh duy nhất. Một người nằm trên sàn có thể là do đã ngã, hoặc đơn giản là đang nằm nghỉ. Sự khác biệt nằm ở **quỹ đạo chuyển động trước đó**: ngã thật sẽ kèm theo sự sụp đổ nhanh của trọng tâm cơ thể, trong khi nằm có chủ ý diễn ra từ từ và có kiểm soát. Do đó, việc sử dụng mạng hồi quy (RNN) để phân tích chuỗi tư thế theo thời gian là cần thiết.

### 2.3.2. Kiến trúc mô hình

Mô hình PoseBiGRU-Attention do nhóm nghiên cứu tự thiết kế và huấn luyện, bao gồm 5 khối chức năng chính:

**Khối 1 — Input & Features:** Nhận đầu vào là chuỗi Skeleton Sequence có kích thước $(B, T, 17, 3)$ — gồm $B$ batch, $T$ frames, 17 keypoints, mỗi keypoint có 3 giá trị $(x, y, confidence)$. Module Learnable Pose Normalization thực hiện chuẩn hóa tọa độ (dựa trên trung tâm cơ thể, tỉ lệ thân), tính toán vận tốc và gia tốc từng khớp, trích xuất đặc trưng góc xương, thống kê độ dài xương — tổng cộng cho ra 137 đặc trưng/frame.

**Khối 2 — Input Projection:** Chiếu 137 đặc trưng thô vào không gian biểu diễn 128 chiều bằng lớp Linear, kết hợp LayerNorm, hàm kích hoạt GELU và Dropout (p=0.3).

**Khối 3 — Bidirectional GRU (2 lớp):** Chuỗi 128 chiều đi qua 2 lớp GRU hai chiều (forward + backward), mỗi hướng có hidden_size = 128, cho đầu ra mỗi frame là vector 256 chiều. GRU hai chiều cho phép mô hình "nhìn" cả quá khứ và tương lai trong chuỗi — đặc biệt quan trọng khi giai đoạn té ngã nằm ở giữa clip.

**Khối 4 — Temporal Aggregation (3 nhánh song song):** Đầu ra chuỗi GRU $(B, T, 256)$ được tổng hợp đồng thời qua: (A) Temporal Attention — học trọng số attention $\alpha^{(t)}$ cho mỗi frame, tạo context vector; (B) Masked Mean Pooling — trung bình có trọng số các frame hợp lệ; (C) Masked Max Pooling — lấy giá trị cực đại. Ba vector 256 chiều được nối (concatenate) thành vector biểu diễn 768 chiều.

**Khối 5 — Classifier & Output:** Vector 768 chiều đi qua lớp Linear → LayerNorm → GELU → Dropout → Linear (256 → 2) → Softmax, cho ra xác suất 2 lớp: Fall và Non-Fall.

> _Xem chi tiết sơ đồ kiến trúc tại Hình 2.1 — ThuatToanModel.jpg_

`[Placeholder: Chèn hình ThuatToanModel.jpg — Sơ đồ kiến trúc chi tiết mạng PoseBiGRU-Attention]`

## 2.4. Cơ chế cửa sổ trượt (Sliding Window)

Chuỗi tư thế từ YOLO-Pose liên tục theo thời gian, nhưng mô hình PoseBiGRU nhận đầu vào có độ dài cố định $T$ frames. Thuật toán cửa sổ trượt (Sliding Window) được sử dụng: mỗi khi thu thập đủ $T$ frame tư thế mới nhất của một người (track ID), hệ thống đóng gói thành một chuỗi $(1, T, 17, 3)$ và đưa qua mô hình. Cửa sổ trượt theo kiểu FIFO — frame cũ nhất bị loại bỏ khi frame mới nhất được thêm vào — đảm bảo phân tích luôn phản ánh trạng thái hiện tại nhất.
