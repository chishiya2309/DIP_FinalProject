# PHẦN 6: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 6.1. Kết luận

Đề tài **"Xây dựng hệ thống cảnh báo té ngã tự động cho người cao tuổi"** đã hoàn thành các mục tiêu đề ra ban đầu:

**Về mặt xử lý ảnh số (DIP):** Hệ thống đã tích hợp thành công pipeline tiền xử lý thích nghi bao gồm khử nhiễu Gaussian, cân bằng sáng CLAHE, xử lý ngược sáng và cân bằng trắng. Module auto-calibration tự động đánh giá chất lượng khung hình đầu vào và chỉ kích hoạt các bộ lọc khi cần thiết, đảm bảo hiệu năng tối ưu.

**Về mặt nhận dạng đối tượng:** Mô hình YOLOv11s-Pose pre-trained kết hợp ByteTrack tracker đã thực hiện thành công bài toán phát hiện người, ước lượng tư thế 17 keypoints và theo dõi đa đối tượng xuyên suốt video stream.

**Về mặt thuật toán phát hiện té ngã:** Kiến trúc Two-tier Detection — kết hợp mạng PoseBiGRU-Attention (do nhóm tự thiết kế và huấn luyện) với hệ thống heuristic dựa trên cơ chế biểu quyết 8 chỉ số — đã cho phép phân biệt chính xác giữa hành vi té ngã thật và các hoạt động sinh hoạt gây nhiễu (cúi nhặt đồ, quỳ gối, ngồi thụp xuống).

**Về mặt hệ thống cảnh báo:** Hệ thống đã triển khai đầy đủ chuỗi cảnh báo đa kênh: âm thanh local, hiển thị trực quan trên UI, ghi log CSV, lưu ảnh Snapshot bằng chứng, và gửi cảnh báo qua Telegram Bot kèm ảnh đến người giám hộ — tất cả hoạt động non-blocking trên các thread riêng biệt.

**Về mặt giao diện:** Ứng dụng Desktop hoàn chỉnh với 5 màn hình chức năng (Đăng nhập, Giám sát Camera, Cài đặt Hệ thống, Lịch sử Cảnh báo) được thiết kế trực quan, dễ sử dụng cho người cao tuổi và người giám hộ.

Đặc biệt, toàn bộ hệ thống có thể chạy trên CPU thông thường mà không cần GPU, với mô hình PoseBiGRU chỉ nặng khoảng 9MB — hoàn toàn phù hợp để triển khai tại gia đình với chi phí phần cứng tối thiểu.

## 6.2. Hướng phát triển

Nhóm đề xuất các hướng phát triển mở rộng để nâng cao hệ thống trong tương lai:

1. **Hỗ trợ camera hồng ngoại (IR Night Vision):** Mở rộng pipeline DIP để xử lý ảnh hồng ngoại, cho phép giám sát 24/7 kể cả khi tắt đèn hoàn toàn.

2. **Multi-camera monitoring:** Hỗ trợ đồng thời nhiều nguồn camera trên cùng một giao diện, hiển thị dạng lưới (grid), phù hợp cho viện dưỡng lão hoặc bệnh viện.

3. **Tích hợp Edge AI:** Triển khai hệ thống trên các thiết bị edge computing như NVIDIA Jetson Nano hoặc Raspberry Pi với mô hình quantized (INT8), tạo thành sản phẩm nhúng hoàn chỉnh.

4. **Cải thiện Re-Identification:** Tích hợp mô hình ReID để xử lý tốt hơn trường hợp đa người, tránh hoán đổi ID khi người đi ngang qua nhau.

5. **Ứng dụng di động (Mobile App):** Xây dựng ứng dụng giám sát trên điện thoại cho người giám hộ, tích hợp push notification thay cho Telegram, xem video stream từ xa.

6. **Fine-tune YOLO trên dữ liệu người nằm:** Huấn luyện lại YOLO-Pose trên bộ dữ liệu bổ sung các trường hợp người nằm trên sofa, giường, sàn nhà — cải thiện khả năng phát hiện trong các tình huống che khuất.

---

# DANH MỤC TÀI LIỆU THAM KHẢO

| STT | Tài liệu |
|-----|---------|
| [1] | R. G. Stefanacci và J. R. Wilkinson, "Té ngã ở người cao tuổi," Cẩm nang Y khoa MSD (MSD Manuals), August 2025. [Online]. Available: https://www.msdmanuals.com/vi/professional/. [Accessed 05-06-2026]. |
| [2] | N. K. K. Võ, "Vì sao và nơi nào người cao tuổi dễ bị té ngã?," Vinmec, 22-07-2024. [Online]. Available: https://www.vinmec.com/. [Accessed 05-06-2026]. |
| [3] | A. P. Kaur, E. Nsugbe, A. Drahota, M. Oldfield, I. Mohagheghian, R. A. Sporea, "State-of-the-art fall detection techniques with emphasis on floor-based systems — A review," *Biomedical Engineering Advances*, vol. 9, p. 100179, 2025. |
| [4] | D. Hrubý, E. Hrubá, M. Černý, "Research of Fall Detection and Fall Prevention Technologies: A Review," *Sensors*, vol. 26, p. 1192, 2026. |
| [5] | K. Perliński, A. Faltyński, A. Świetlicka, "Human Fall Detection with Infrared Imaging: A Comparison of Graph Convolutional Networks and YOLO," *Sensors*, vol. 26, p. 2794, 2026. |
| [6] | H. V. Dũng, "Chapter 1. Introduction," *Bài giảng Xử lý ảnh số*, Trường ĐH Công nghệ Kỹ thuật TP.HCM. |
| [7] | H. V. Dũng, "Chapter 2. Image Enhancement in Spatial Domain," *Bài giảng Xử lý ảnh số*, Trường ĐH Công nghệ Kỹ thuật TP.HCM. |
| [8] | Ultralytics, "YOLOv8 Documentation," 2024. [Online]. Available: https://docs.ultralytics.com/. |
| [9] | K. Cho, B. van Merrienboer, C. Gulcehre, et al., "Learning phrase representations using RNN Encoder-Decoder for statistical machine translation," *EMNLP*, 2014. |
| [10] | N. Wojke, A. Bewley, D. Paulus, "Simple Online and Realtime Tracking with a Deep Association Metric," *ICIP*, 2017. |

---

# DANH MỤC HÌNH ẢNH

| STT | Hình | Mô tả |
|-----|------|-------|
| 1 | Hình 2.1 | Sơ đồ kiến trúc chi tiết mạng PoseBiGRU-Attention (ThuatToanModel.jpg) |
| 2 | Hình 3.1 | Sơ đồ kiến trúc tổng quát hệ thống (System Architecture) |
| 3 | Hình 3.2 | So sánh frame trước và sau DIP Enhancement |
| 4 | Hình 3.3 | Tin nhắn cảnh báo Telegram trên điện thoại |
| 5 | Hình 4.1 | Ảnh chụp màn hình Đăng nhập |
| 6 | Hình 4.2 | Ảnh chụp màn hình Giám sát Camera (an toàn) |
| 7 | Hình 4.3 | Ảnh chụp màn hình Giám sát Camera (phát hiện ngã) |
| 8 | Hình 4.4 | Ảnh chụp màn hình Cài đặt Hệ thống |
| 9 | Hình 4.5 | Ảnh chụp màn hình Lịch sử Cảnh báo với Snapshot |
| 10 | Hình 4.6 | Kaggle Notebook — Training output |
| 11 | Hình 4.7 | Điện thoại chạy IP Webcam + Laptop nhận stream |
| 12 | Hình 5.1 | Đồ thị Training Loss vs Validation Loss |
| 13 | Hình 5.2 | Confusion Matrix |
| 14 | Hình 5.3 | YOLO detection trước/sau DIP trong điều kiện thiếu sáng |
| 15 | Hình 5.4 | Ảnh minh họa các kịch bản test |

---

# PHÂN CÔNG CÔNG VIỆC

| Thành viên | MSSV | Công việc |
|-----------|------|-----------|
| Nguyễn Thái Bảo | 23110078 | `[Placeholder: Điền phân công cụ thể]` |
| Lê Quang Hưng | 23110110 | `[Placeholder: Điền phân công cụ thể]` |
| Lương Nguyễn Thành Hưng | 23110111 | `[Placeholder: Điền phân công cụ thể]` |
