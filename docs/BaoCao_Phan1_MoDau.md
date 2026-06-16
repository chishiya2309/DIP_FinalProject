# PHẦN 1: MỞ ĐẦU

## 1.1. Lý do chọn đề tài

Hiện nay, sự già hóa dân số đang đặt ra những thách thức to lớn cho toàn cầu trong việc xây dựng một môi trường sống độc lập và an toàn cho người cao tuổi. Trong đó, té ngã được ghi nhận là một trong những rủi ro sức khỏe nghiêm trọng nhất. Theo báo cáo từ Cẩm nang Y khoa MSD (MSD Manuals), ở những người từ 65 tuổi trở lên, té ngã là nguyên nhân hàng đầu gây tử vong do chấn thương và là nguyên nhân tử vong đứng thứ bảy nói chung [1]. Người cao tuổi có nguy cơ vấp ngã rất cao do sự suy giảm chức năng vận động, thị lực kém và các bệnh lý nền đi kèm [2]. Nghiên cứu cũng chỉ ra rằng, việc phát hiện chậm trễ và thiếu can thiệp y tế kịp thời sau khi ngã sẽ làm tăng đột biến tỷ lệ mắc các biến chứng nghiêm trọng như gãy xương, chấn thương sọ não, thậm chí dẫn đến tử vong. Do đó, việc thiết lập một hệ thống giám sát nhằm phát hiện và cảnh báo tức thời sự cố té ngã là yêu cầu vô cùng cấp thiết.

Để giải quyết bài toán này, nhiều hệ thống phát hiện té ngã (Fall Detection Systems) đã được nghiên cứu và phát triển. Phổ biến nhất là các thiết bị đeo trên người (wearable sensors) như đồng hồ thông minh hoặc vòng cổ tích hợp gia tốc kế. Tuy nhiên, các thiết bị này tồn tại nhược điểm chí mạng trong thực tiễn: tính hiệu quả của chúng phụ thuộc hoàn toàn vào việc người dùng có mang thiết bị liên tục hay không. Việc người cao tuổi thường xuyên quên đeo thiết bị, quên sạc pin, hoặc cảm thấy vướng víu đã làm giảm đáng kể độ tin cậy của hệ thống [3]. Bên cạnh đó, các hệ thống cảm biến gắn trên sàn nhà (smart flooring) dù khắc phục được nhược điểm của thiết bị đeo nhưng lại có chi phí lắp đặt, bảo trì rất đắt đỏ và khó triển khai trên diện rộng [3].

Đứng trước những hạn chế đó, các phương pháp tiếp cận không xâm nhập (unobtrusive methods) dựa trên camera giám sát và kỹ thuật Xử lý ảnh số (Digital Image Processing) đang nổi lên như một giải pháp tối ưu. Việc ứng dụng các thuật toán xử lý ảnh — từ các bộ lọc trong miền không gian (Spatial Domain Filters) để tiền xử lý và khử nhiễu, đến các kỹ thuật trích xuất đặc trưng (Feature Extraction) và nhận dạng tư thế người [4] — cho phép máy tính theo dõi và phân tích hành vi của con người một cách liên tục, hoàn toàn tự động mà không đòi hỏi người cao tuổi phải mang vác bất kỳ thiết bị nào trên cơ thể. Sự kết hợp giữa xử lý ảnh truyền thống và các mô hình học sâu hiện đại có khả năng phân biệt chính xác hành vi té ngã với các hoạt động sinh hoạt bình thường, đảm bảo tính thời gian thực (real-time) và giảm thiểu cảnh báo giả [5].

Nhận thấy rõ tính cấp thiết từ thực tiễn xã hội cùng với tiềm năng to lớn của công nghệ, nhóm quyết định lựa chọn đề tài: **"Xây dựng hệ thống cảnh báo té ngã tự động cho người cao tuổi"**.

## 1.2. Mục tiêu nghiên cứu

Mục tiêu tổng quan của đề tài là nghiên cứu, thiết kế và triển khai một hệ thống giám sát tự động không xâm nhập dựa trên công nghệ Thị giác máy tính (Computer Vision) và Xử lý ảnh số. Hệ thống nhằm mục đích theo dõi, phát hiện sớm và đưa ra cảnh báo theo thời gian thực đối với hành vi té ngã của người cao tuổi trong môi trường trong nhà (indoor environment).

Các mục tiêu kỹ thuật cụ thể bao gồm:

**Về thu nhận và tiền xử lý dữ liệu:** Áp dụng các kỹ thuật xử lý ảnh trong miền không gian nhằm khử nhiễu và cân bằng độ sáng, độ tương phản cho chuỗi video đầu vào. Module tiền xử lý phải hoạt động thích nghi (adaptive) với mọi điều kiện ánh sáng.

**Về nhận dạng và theo dõi đối tượng:** Tích hợp mô hình YOLO-Pose để nhận diện người và trích xuất liên tục các đặc trưng khung xương (Skeletal Keypoints) qua từng khung hình. Kết hợp bộ theo dõi ByteTrack để gán và duy trì định danh (ID) cho mỗi đối tượng xuyên suốt video.

**Về thuật toán phát hiện té ngã:** Xây dựng kiến trúc phát hiện hai tầng (Two-tier Detection): (1) Mạng PoseBiGRU-Attention do nhóm tự huấn luyện để phân loại chuỗi tư thế, kết hợp (2) Hệ thống quy tắc heuristic dựa trên hình học để tăng cường tốc độ phản ứng và xử lý các trường hợp biên. Hệ thống phải phân biệt được hành vi té ngã thật với các hoạt động gây nhiễu như cúi nhặt đồ, ngồi thụp xuống, hay quỳ gối.

**Về hệ thống cảnh báo:** Khi phát hiện té ngã, hệ thống phát âm thanh cảnh báo, hiển thị khung đỏ trên màn hình, lưu ảnh bằng chứng (Snapshot), ghi lịch sử CSV, và gửi cảnh báo qua Telegram kèm ảnh chụp đến người giám hộ.

## 1.3. Đối tượng và phạm vi nghiên cứu

### 1.3.1. Đối tượng nghiên cứu

- **Đối tượng giám sát:** Hành vi và tư thế động học của con người trong không gian sinh hoạt. Trọng tâm là phân biệt giữa sự cố "té ngã vô thức" và các hoạt động sinh hoạt thường ngày (ADLs).
- **Đối tượng xử lý kỹ thuật:** Các khung hình tĩnh và chuỗi khung hình liên tục được trích xuất từ nguồn video kỹ thuật số, cụ thể là các đặc trưng bộ khung xương (Skeletal Keypoints) của con người trên không gian ảnh 2D.

### 1.3.2. Phạm vi nghiên cứu

**Về môi trường:** Hệ thống tập trung hoạt động trong không gian kín (indoor), tạm thời bỏ qua bối cảnh ngoại cảnh đông người.

**Về thiết bị:** Hệ thống tiếp nhận nguồn dữ liệu hình ảnh quang học (RGB) từ camera tĩnh đơn tròng hoặc Camera IP (qua giao thức HTTP/RTSP). Loại trừ hoàn toàn wearable sensors.

**Về giải pháp phần mềm:** Sử dụng các kỹ thuật DIP cơ bản ở giai đoạn tiền xử lý, mô hình YOLO-Pose đã được huấn luyện trước (pre-trained) để phát hiện người, và mạng PoseBiGRU tự huấn luyện cho bài toán phân loại chuỗi hành vi.
