**TRƯỜNG ĐẠI HỌC CÔNG NGHỆ KỸ THUẬT THÀNH PHỐ HỒ CHÍ MINH**

**KHOA CÔNG NGHỆ THÔNG TIN**

![A logo of hands holding a book
Description automatically generated](media/image1.png){width="1.5666666666666667in" height="2.0in"}

**MÔN HỌC: XỬ LÝ ẢNH SỐ**

**[ĐỀ TÀI CUỐI KỲ:]{.underline}**

**XÂY DỰNG HỆ THỐNG CẢNH BÁO TÉ NGÃ TỰ ĐỘNG**

**CHO NGƯỜI CAO TUỔI**

**GVHD: *PGS.TS. Hoàng Văn Dũng***

**Mã LHP: *DIPR430685_06CLC***

**Học kỳ:** *II*

**Năm học: *2025 -- 2026***

**Danh sách sinh viên thực hiện: Nhóm 05**

| **MSSV** | **Họ tên**              |
|----------|-------------------------|
| 23110078 | Nguyễn Thái Bảo         |
| 23110110 | Lê Quang Hưng           |
| 23110111 | Lương Nguyễn Thành Hưng |

**Thành phố Hồ Chí Minh, tháng 06 năm 2026**

**MỤC LỤC**

[DANH MỤC HÌNH ẢNH](#_Toc231759728)

[PHẦN 1: MỞ ĐẦU [1](#_Toc231759729)](#_Toc231759729)

[1.1. Lý do chọn đề tài [1](#_Toc231759730)](#_Toc231759730)

[1.2. Mục tiêu nghiên cứu [2](#_Toc231759731)](#_Toc231759731)

[1.3. Đối tượng và phạm vi nghiên cứu: [3](#_Toc231759732)](#_Toc231759732)

[1.3.1. Đối tượng nghiên cứu [3](#_Toc231759733)](#_Toc231759733)

[1.3.2. Phạm vi nghiên cứu [3](#_Toc231759734)](#_Toc231759734)

[PHẦN 2: CƠ SỞ LÝ THUYẾT [4](#_Toc231759735)](#_Toc231759735)

[2.1. Tổng quan về xử lý ảnh số và tiền xử lý (Digital Image Processing) [4](#_Toc231759736)](#_Toc231759736)

[2.1.1. Khái niệm cơ bản và quy trình xử lý ảnh [4](#_Toc231759737)](#_Toc231759737)

[2.1.2. Kỹ thuật lọc và khử nhiễu trong miền không gian (Spatial Domain Filtering) [5](#_Toc231759738)](#_Toc231759738)

[2.1.3. Cải thiện độ tương phản và cân bằng sáng (Histogram Equalization) trong môi trường thực tế [6](#_Toc231759739)](#_Toc231759739)

[2.2. Trích xuất đặc trưng và nhận dạng mẫu (Feature Extraction & Pattern Recognition) [6](#_Toc231759740)](#_Toc231759740)

[2.2.1. Trích xuất đặc trưng hình học (Shape features) và vùng biên [6](#_Toc231759741)](#_Toc231759741)

[2.2.2. Đặc trưng động học (Motion features) trong phân loại hành vi [6](#_Toc231759742)](#_Toc231759742)

[2.2.3. Hạn chế của các phương pháp trích xuất đặc trưng truyền thống (Hand-crafted features) [6](#_Toc231759743)](#_Toc231759743)

[2.3. Mạng nơ-ron tích chập và mô hình YOLOv8 [6](#_Toc231759744)](#_Toc231759744)

[2.3.1 Kiến trúc mạng CNN và nguyên lý trích xuất đặc trưng tự động [6](#_Toc231759745)](#_Toc231759745)

[2.3.2. Tổng quan bài toán phát hiện đối tượng (Object Detection) và sự tiến hóa của họ YOLO [6](#_Toc231759746)](#_Toc231759746)

[2.3.3. Kiến trúc YOLOv8-pose: Tích hợp phát hiện người và nội suy điểm neo (keypoints) [6](#_Toc231759747)](#_Toc231759747)

[2.4. Bài toán ước lượng tư thế người (Human Pose Estimation) [6](#_Toc231759748)](#_Toc231759748)

[2.4.1. Khái niệm và các hướng tiếp cận hiện đại (Top-down vs Bottom-up) [7](#_Toc231759749)](#_Toc231759749)

[2.4.2. Hệ quy chiếu COCO-17 Keypoints và phương pháp biểu diễn bộ khung xương (Skeleton) [7](#_Toc231759750)](#_Toc231759750)

[2.4.3. Chuỗi tư thế (Skeleton Sequence) - Trích xuất và định dạng dữ liệu đầu vào [7](#_Toc231759751)](#_Toc231759751)

[2.5. Nhận dạng hành động theo chuỗi không gian - thời gian (Spatio-Temporal Action Recognition) [7](#_Toc231759752)](#_Toc231759752)

[2.5.1. Tầm quan trọng của phân tích chuỗi thời gian so với phân tích khung hình tĩnh (Static frame) [7](#_Toc231759753)](#_Toc231759753)

[2.5.2. Chuyển đổi dữ liệu động học: Từ Skeleton Sequence sang 3D Pseudo-Heatmaps (Bản đồ nhiệt giả 3D) [7](#_Toc231759754)](#_Toc231759754)

[2.5.2. Kiến trúc mạng PoseC3D: Cơ chế phân tích luồng không gian - thời gian bằng 3D Convolution [7](#_Toc231759755)](#_Toc231759755)

[2.6. Học chuyển giao (Transfer Learning) và chiến lược Fine-tuning [7](#_Toc231759756)](#_Toc231759756)

[2.6.1. Nguyên lý học chuyển giao và tính cấp thiết trong bài toán phát hiện té ngã [7](#_Toc231759757)](#_Toc231759757)

[2.6.2. Phân tích mô hình PoseC3D tiền huấn luyện (Pre-trained on NTU RGB+D/Kinetics) [7](#_Toc231759758)](#_Toc231759758)

[2.6.3. Kỹ thuật Data Augmentation (Tăng cường dữ liệu) chuyên biệt cho skeleton sequence [7](#_Toc231759759)](#_Toc231759759)

[2.6.4. Chiến lược Fine-tuning giải quyết mất cân bằng dữ liệu cho 2 nhãn: FALL và NON-FALL [7](#_Toc231759760)](#_Toc231759760)

[PHẦN 3: PHƯƠNG PHÁP VÀ THIẾT KẾ HỆ THỐNG ĐỀ XUẤT [8](#_Toc231759761)](#_Toc231759761)

[3.1. Kiến trúc tổng quát của hệ thống (System Architecture) [8](#_Toc231759762)](#_Toc231759762)

[3.2. Thu nhận và tiền xử lý dữ liệu video (DIP Preprocessing) [8](#_Toc231759763)](#_Toc231759763)

[3.3. Nhận dạng và trích xuất đặc trưng tư thế bằng YOLOv8-Pose [8](#_Toc231759764)](#_Toc231759764)

[3.4. Mô hình hóa chuỗi động học bằng cửa sổ trượt (Sliding Window) [9](#_Toc231759765)](#_Toc231759765)

[3.5. Phân loại hành vi té ngã bằng mạng PoseC3D [9](#_Toc231759766)](#_Toc231759766)

[3.6. Module cảnh báo thời gian thực (Real-time Alerting) [10](#_Toc231759767)](#_Toc231759767)

[PHẦN 4: THỰC NGHIỆM VÀ ĐÁNH GIÁ HỆ THỐNG [11](#_Toc231759768)](#_Toc231759768)

[4.1. Môi trường, công cụ và tài nguyên phát triển [11](#_Toc231759769)](#_Toc231759769)

[4.2. Bộ dữ liệu (Dataset) và quy trình chuẩn bị dữ liệu [11](#_Toc231759770)](#_Toc231759770)

[4.3. Các độ đo đánh giá hiệu năng (Evaluation Metrics) [12](#_Toc231759771)](#_Toc231759771)

[4.4. Kịch bản thử nghiệm và đánh giá độ bền vững (Test cases & Robustness) [12](#_Toc231759772)](#_Toc231759772)

[References [14](#_Toc231759773)](#_Toc231759773)

[]{#_Toc231759728 .anchor}DANH MỤC HÌNH ẢNH

[Hình 2. 1. Quy trình xử lý ảnh [5](#_Toc231759705)](#_Toc231759705)

[Hình 3. 1. Sơ đồ pipeline đề xuất cho hệ thống [8](#_Toc231759707)](#_Toc231759707)

[]{#_Toc231759729 .anchor}PHẦN 1: MỞ ĐẦU

[]{#_Toc231759730 .anchor}1.1. Lý do chọn đề tài

Hiện nay, sự già hóa dân số đang đặt ra những thách thức to lớn cho toàn cầu trong việc xây dựng một môi trường sống độc lập và an toàn cho người cao tuổi. Trong đó, té ngã được ghi nhận là một trong những rủi ro sức khỏe nghiêm trọng nhất. Theo báo cáo từ Cẩm nang Y khoa MSD (MSD Manuals), ở những người từ 65 tuổi trở lên, té ngã là nguyên nhân hàng đầu gây tử vong do chấn thương và là nguyên nhân tử vong đứng thứ bảy nói chung \[1\]. Người cao tuổi có nguy cơ vấp ngã rất cao do sự suy giảm chức năng vận động, thị lực kém và các bệnh lý nền đi kèm \[2\]. Nghiên cứu cũng chỉ ra rằng, việc phát hiện chậm trễ và thiếu can thiệp y tế kịp thời sau khi ngã sẽ làm tăng đột biến tỷ lệ mắc các biến chứng nghiêm trọng như gãy xương, chấn thương sọ não, thậm chí dẫn đến tử vong. Do đó, việc thiết lập một hệ thống giám sát nhằm phát hiện và cảnh báo tức thời sự cố té ngã là yêu cầu vô cùng cấp thiết.

Để giải quyết bài toán này, nhiều hệ thống phát hiện té ngã (Fall Detection Systems) đã được nghiên cứu và phát triển. Phổ biến nhất là các thiết bị đeo trên người (wearable sensors) như đồng hồ thông minh hoặc vòng cổ tích hợp gia tốc kế. Tuy nhiên, các thiết bị này tồn tại nhược điểm chí mạng trong thực tiễn: tính hiệu quả của chúng phụ thuộc hoàn toàn vào việc người dùng có mang thiết bị liên tục hay không. Việc người cao tuổi thường xuyên quên đeo thiết bị, quên sạc pin, hoặc cảm thấy vướng víu, khó chịu đã làm giảm đáng kể độ tin cậy của hệ thống \[3\]. Bên cạnh đó, các hệ thống cảm biến gắn trên sàn nhà (smart flooring) dù khắc phục được nhược điểm của thiết bị đeo nhưng lại có chi phí lắp đặt, bảo trì rất đắt đỏ và khó triển khai trên diện rộng \[3\].

Đứng trước những hạn chế đó, các phương pháp tiếp cận không xâm nhập (unobtrusive methods) dựa trên camera giám sát và kỹ thuật Xử lý ảnh số (Digital Image Processing) đang nổi lên như một giải pháp tối ưu. Việc ứng dụng các thuật toán xử lý ảnh - từ các bộ lọc trong miền không gian (Spatial Domain Filters) để tiền xử lý và khử nhiễu, đến các kỹ thuật trích xuất đặc trưng (Feature Extraction) và nhận dạng tư thế người \[4\] - cho phép máy tính theo dõi và phân tích hành vi của con người một cách liên tục, hoàn toàn tự động mà không đòi hỏi người cao tuổi phải mang vác bất kỳ thiết bị nào trên cơ thể. Sự kết hợp giữa xử lý ảnh truyền thống và các mô hình học sâu hiện đại (như YOLOv8) có khả năng phân biệt chính xác hành vi té ngã với các hoạt động sinh hoạt bình thường (như ngồi, nằm, hay cúi gập người), đảm bảo tính thời gian thực (real-time) và giảm thiểu cảnh báo giả \[5\].

Nhận thấy rõ tính cấp thiết từ thực tiễn xã hội cùng với tiềm năng to lớn của công nghệ, nhóm quyết định lựa chọn đề tài: **"Xây dựng hệ thống cảnh báo té ngã tự động cho người cao tuổi"**. Đề tài này không chỉ mang ý nghĩa nhân văn sâu sắc trong việc bảo vệ sức khỏe và tính mạng cộng đồng, mà còn là cơ hội tuyệt vời để nhóm nghiên cứu áp dụng trực tiếp các nền tảng kiến thức cốt lõi của môn học xử lý ảnh số (như tiền xử lý ảnh, phân vùng ảnh, và nhận dạng mẫu) vào một bài toán thực tế của kỹ thuật y sinh và an ninh thông minh.

[]{#_Toc231759731 .anchor}1.2. Mục tiêu nghiên cứu

Mục tiêu tổng quan của đề tài là nghiên cứu, thiết kế và triển khai một hệ thống giám sát tự động không xâm nhập (unobtrusive monitoring system) dựa trên công nghệ Thị giác máy tính (Computer Vision) và Xử lý ảnh số (Digital Image Processing). Hệ thống này nhằm mục đích theo dõi, phát hiện sớm và đưa ra cảnh báo theo thời gian thực (real-time) đối với hành vi té ngã của người cao tuổi trong môi trường sống trong nhà (indoor environment), từ đó hỗ trợ giảm thiểu tối đa các biến chứng nguy hiểm do việc phát hiện chậm trễ gây ra.

Để hoàn thành mục tiêu tổng quát nêu trên, nhóm nghiên cứu đề ra các mục tiêu kỹ thuật cụ thể như sau:

**Về mặt thu nhận và tiền xử lý dữ liệu:** Nghiên cứu và áp dụng các kỹ thuật xử lý ảnh trong miền không gian (Spatial Domain Filters) nhằm khử nhiễu và cân bằng độ sáng, độ tương phản cho chuỗi video thu nhận từ camera. Điều này giúp đảm bảo chất lượng dữ liệu đầu vào ổn định ngay cả trong điều kiện ánh sáng thực tế phức tạp tại không gian nhà ở.

**Về mặt nhận dạng và theo dõi đối tượng:** Tích hợp mô hình học sâu hiện đại (cụ thể là mạng nơ-ron tích chập YOLOv8) để phân vùng ảnh, nhận diện người và trích xuất liên tục các đặc trưng hình học như khung bao (Bounding Box) hoặc khung xương (Skeletal Keypoints) qua từng khung hình (frame).

**Về mặt thuật toán phát hiện té ngã:** Xây dựng và tinh chỉnh thuật toán phân tích hành vi dựa trên các tham số động học và hình học (ví dụ: tỷ lệ chênh lệch giữa chiều cao và chiều rộng của khung bao, hoặc vận tốc rơi của trọng tâm cơ thể). Hệ thống phải đạt được độ chính xác cao trong việc phân biệt rõ ràng hành vi "té ngã vô thức" với các hoạt động sinh hoạt bình thường có quỹ đạo tương đồng (như cúi gập người nhặt đồ, ngồi thụp xuống ghế, hoặc nằm có chủ ý) nhằm hạn chế tối đa tỷ lệ báo động giả (False Alarm Rate).

**Về mặt thiết kế hệ thống cảnh báo:** Hoàn thiện module đầu ra (output module) có khả năng xử lý tức thời. Khi hệ thống xác định có sự cố té ngã, ngay lập tức (\< 30 giây) kích hoạt các tín hiệu cảnh báo (như phát âm thanh hoặc đánh dấu khung cảnh báo đỏ trực quan trên màn hình giám sát) để người chăm sóc hoặc nhân viên y tế có thể can thiệp kịp thời.

[]{#_Toc231759732 .anchor}1.3. Đối tượng và phạm vi nghiên cứu:

[]{#_Toc231759733 .anchor}1.3.1. Đối tượng nghiên cứu

Đối tượng giám sát thực tế: Hành vi và tư thế động học của con người (hướng tới tệp người dùng là người cao tuổi) trong không gian sinh hoạt. Trọng tâm nghiên cứu là sự biến thiên về mặt hình học và quỹ đạo chuyển động nhằm phân biệt rõ ràng giữa sự cố "té ngã vô thức" và các hoạt động sinh hoạt thường ngày (Activities of Daily Living - ADLs) như đi lại, ngồi, nằm có chủ ý, hoặc cúi gập người.

Đối tượng xử lý kỹ thuật: Các khung hình (frames) tĩnh và chuỗi khung hình liên tục được trích xuất từ nguồn video kỹ thuật số. Cụ thể là các đặc trưng hình thái (khung bao đối tượng - Bounding Box, hoặc bộ khung xương - Skeletal Keypoints) của con người trên không gian ảnh 2D.

[]{#_Toc231759734 .anchor}1.3.2. Phạm vi nghiên cứu

Nhằm đảm bảo tính khả thi về mặt thời gian và bám sát mục tiêu của học phần Xử lý ảnh số, đề tài được giới hạn trong các điều kiện cụ thể sau:

**Về môi trường và không gian (Environmental Scope):**

Hệ thống tập trung hoạt động trong không gian kín (indoor environments) như phòng khách, phòng ngủ, hành lang tại các hộ gia đình hoặc viện dưỡng lão.

Nghiên cứu tạm thời bỏ qua các bối cảnh ngoại cảnh (outdoor) đông người hoặc các điều kiện môi trường có nhiễu động phức tạp (mưa, sương mù, chuyển động nền quá lớn).

**Về thiết bị thu nhận dữ liệu (Hardware Scope):**

Hệ thống chỉ tiếp nhận và xử lý nguồn dữ liệu hình ảnh quang học (RGB) hoặc hình ảnh hồng ngoại (Infrared) được ghi lại từ một camera tĩnh, đơn tròng (monocular fixed camera) đặt tại các góc nhìn ngang hoặc chéo trên cao.

Nghiên cứu hoàn toàn loại trừ các phương pháp thu thập dữ liệu có tính xâm nhập hoặc đeo trên người (wearable sensors) như gia tốc kế, con quay hồi chuyển, hay các công nghệ quét không gian 3D phức tạp như LiDAR, Radar.

**Về thuật toán và giải pháp phần mềm (Technical Scope):**

Sử dụng các kỹ thuật Xử lý ảnh số cơ bản ở giai đoạn tiền xử lý (Preprocessing) như lọc trong miền không gian (Spatial Domain Filters) để khử nhiễu, làm mịn ảnh và cân bằng sáng nhằm cải thiện chất lượng đầu vào.

Áp dụng mô hình học sâu đã được tối ưu hóa (YOLOv8) để thực hiện bài toán phát hiện đối tượng (Object Detection), thay vì tự xây dựng mô hình nhận diện từ đầu.

Thuật toán phân tích té ngã được giới hạn trong việc tính toán logic toán học dựa trên sự thay đổi của khung bao đối tượng (tỷ lệ chênh lệch giữa chiều cao và chiều rộng, tính toán trọng tâm) trong một khung thời gian ngắn, giúp đảm bảo tốc độ khung hình/giây (FPS) cao, đáp ứng yêu cầu cảnh báo tức thời. Khóa luồng nghiên cứu không đi sâu vào việc huấn luyện các mạng nơ-ron phân tích chuỗi thời gian (như RNN/LSTM) đòi hỏi tài nguyên điện toán khổng lồ.

[]{#_Toc231759735 .anchor}PHẦN 2: CƠ SỞ LÝ THUYẾT

[]{#_Toc231759736 .anchor}2.1. Tổng quan về xử lý ảnh số và tiền xử lý (Digital Image Processing)

Tiền xử lý ảnh (Preprocessing) là giai đoạn nền tảng và mang tính quyết định đối với hiệu năng của bất kỳ hệ thống thị giác máy tính nào. Trong phạm vi đề tài, các kỹ thuật xử lý ảnh số được áp dụng nhằm cải thiện chất lượng dữ liệu video đầu vào từ camera giám sát, triệt tiêu các yếu tố nhiễu loạn của môi trường, qua đó tạo ra nguồn dữ liệu tối ưu nhất trước khi đưa vào mạng nơ-ron học sâu để nhận dạng tư thế người.

[]{#_Toc231759737 .anchor}2.1.1. Khái niệm cơ bản và quy trình xử lý ảnh

Xử lý ảnh số (Digital Image Processing - DIP) là quá trình áp dụng các thuật toán máy tính để thao tác, phân tích và biến đổi một hình ảnh kỹ thuật số nhằm nâng cao chất lượng thị giác hoặc trích xuất các thông tin định lượng hữu ích. Theo kiến trúc chuẩn của một hệ thống thị giác máy, quy trình xử lý ảnh cơ bản thường trải qua các giai đoạn nối tiếp nhau bao gồm: Thu nhận ảnh (Image Acquisition), Nâng cao chất lượng ảnh (Image Enhancement), Khôi phục ảnh (Image Restoration), Phân vùng ảnh (Segmentation) và Trích xuất đặc trưng (Feature/Representation Extraction) \[6\]. Đối với hệ thống cảnh báo té ngã trong nhà, dữ liệu hình ảnh thô thường xuyên chứa nhiều thông tin dư thừa. Do đó, quy trình tiền xử lý đóng vai trò chắt lọc thông tin, giúp mô hình phân tích hành vi phía sau tập trung hoàn toàn vào đối tượng con người thay vì bị phân tâm bởi bối cảnh xung quanh.

<figure>
<img src="media/image2.png" style="width:6.51461in;height:2.34902in" alt="A diagram of a process Description automatically generated" />
<figcaption><p><span id="_Toc231759705" class="anchor"></span>Hình 2. 1. Quy trình xử lý ảnh</p></figcaption>
</figure>

[]{#_Toc231759738 .anchor}2.1.2. Kỹ thuật lọc và khử nhiễu trong miền không gian (Spatial Domain Filtering)

Kỹ thuật xử lý trong miền không gian (Spatial Domain) đề cập đến các phương pháp can thiệp và biến đổi trực tiếp giá trị của từng điểm ảnh (pixel) trên ma trận ảnh số, được biểu diễn qua hàm biến đổi $g(x,y)\  = \ T\lbrack f(x,y)\rbrack$ \[7\]. Trong môi trường hoạt động thực tế, cảm biến quang học của camera giám sát thường xuyên bị ảnh hưởng bởi nhiễu hạt (Gaussian noise) hoặc nhiễu muối tiêu (Salt-and-pepper noise) do điều kiện thiếu sáng. Để giải quyết thách thức này, các bộ lọc không gian làm mịn (Smoothing/Lowpass Spatial Filters) được tích hợp vào hệ thống. Cụ thể, bộ lọc Gaussian (Gaussian Blur) hoạt động dựa trên nguyên lý tính trung bình có trọng số của các điểm ảnh lân cận (neighborhood operations) được ưu tiên sử dụng. Khác với bộ lọc trung bình thông thường, bộ lọc Gaussian có khả năng khử nhiễu tần số cao (high-frequency noise) vô cùng hiệu quả nhưng vẫn bảo toàn được các đường biên (edges) cấu trúc cơ thể người \[7\]. Việc bảo toàn biên sắc nét này là điều kiện tiên quyết để mô hình nhận dạng phía sau có thể nội suy chính xác các điểm neo (keypoints) của bộ khung xương.

[]{#_Toc231759739 .anchor}2.1.3. Cải thiện độ tương phản và cân bằng sáng (Histogram Equalization) trong môi trường thực tế

Một thách thức đặc thù đối với các hệ thống giám sát chăm sóc sức khỏe người cao tuổi là sự bất ổn định về nguồn sáng tại không gian nhà ở (chẳng hạn như hiện tượng ngược sáng từ cửa sổ, ánh sáng yếu vào ban đêm, hoặc các góc khuất tạo bóng râm). Các máy ảnh và cảm biến hình ảnh tiêu chuẩn thường bị giới hạn về dải động (dynamic range), dẫn đến việc các vùng tối của bức ảnh bị mất hoàn toàn chi tiết. Để khắc phục, kỹ thuật xử lý lược đồ xám (Histogram Processing) được áp dụng nhằm tái phân bố lại tần suất xuất hiện của các giá trị cường độ sáng \[7\]. Phương pháp Cân bằng lược đồ xám (Histogram Equalization) sẽ kéo giãn phân bố cường độ sáng từ vùng tập trung hẹp sang toàn bộ dải giá trị khả dụng (từ 0 đến 255 đối với ảnh 8-bit). Quá trình này giúp nâng cao độ tương phản tổng thể một cách tự nhiên, khôi phục các chi tiết chìm trong vùng tối, đảm bảo rằng đối tượng sinh học luôn được tách biệt rõ ràng khỏi phông nền (background), qua đó duy trì tính liên tục của chuỗi dữ liệu đầu vào trong mọi điều kiện ánh sáng.

[]{#_Toc231759740 .anchor}2.2. Trích xuất đặc trưng và nhận dạng mẫu (Feature Extraction & Pattern Recognition)

[]{#_Toc231759741 .anchor}2.2.1. Trích xuất đặc trưng hình học (Shape features) và vùng biên

[]{#_Toc231759742 .anchor}2.2.2. Đặc trưng động học (Motion features) trong phân loại hành vi

[]{#_Toc231759743 .anchor}2.2.3. Hạn chế của các phương pháp trích xuất đặc trưng truyền thống (Hand-crafted features)

[]{#_Toc231759744 .anchor}2.3. Mạng nơ-ron tích chập và mô hình YOLOv8

[]{#_Toc231759745 .anchor}2.3.1 Kiến trúc mạng CNN và nguyên lý trích xuất đặc trưng tự động

[]{#_Toc231759746 .anchor}2.3.2. Tổng quan bài toán phát hiện đối tượng (Object Detection) và sự tiến hóa của họ YOLO

[]{#_Toc231759747 .anchor}2.3.3. Kiến trúc YOLOv8-pose: Tích hợp phát hiện người và nội suy điểm neo (keypoints)

[]{#_Toc231759748 .anchor}2.4. Bài toán ước lượng tư thế người (Human Pose Estimation)

[]{#_Toc231759749 .anchor}2.4.1. Khái niệm và các hướng tiếp cận hiện đại (Top-down vs Bottom-up)

[]{#_Toc231759750 .anchor}2.4.2. Hệ quy chiếu COCO-17 Keypoints và phương pháp biểu diễn bộ khung xương (Skeleton)

[]{#_Toc231759751 .anchor}2.4.3. Chuỗi tư thế (Skeleton Sequence) - Trích xuất và định dạng dữ liệu đầu vào

[]{#_Toc231759752 .anchor}2.5. Nhận dạng hành động theo chuỗi không gian - thời gian (Spatio-Temporal Action Recognition)

[]{#_Toc231759753 .anchor}2.5.1. Tầm quan trọng của phân tích chuỗi thời gian so với phân tích khung hình tĩnh (Static frame)

[]{#_Toc231759754 .anchor}2.5.2. Chuyển đổi dữ liệu động học: Từ Skeleton Sequence sang 3D Pseudo-Heatmaps (Bản đồ nhiệt giả 3D)

[]{#_Toc231759755 .anchor}2.5.2. Kiến trúc mạng PoseC3D: Cơ chế phân tích luồng không gian - thời gian bằng 3D Convolution

[]{#_Toc231759756 .anchor}2.6. Học chuyển giao (Transfer Learning) và chiến lược Fine-tuning

[]{#_Toc231759757 .anchor}2.6.1. Nguyên lý học chuyển giao và tính cấp thiết trong bài toán phát hiện té ngã

[]{#_Toc231759758 .anchor}2.6.2. Phân tích mô hình PoseC3D tiền huấn luyện (Pre-trained on NTU RGB+D/Kinetics)

[]{#_Toc231759759 .anchor}2.6.3. Kỹ thuật Data Augmentation (Tăng cường dữ liệu) chuyên biệt cho skeleton sequence

[]{#_Toc231759760 .anchor}2.6.4. Chiến lược Fine-tuning giải quyết mất cân bằng dữ liệu cho 2 nhãn: FALL và NON-FALL

[]{#_Toc231759761 .anchor}PHẦN 3: PHƯƠNG PHÁP VÀ THIẾT KẾ HỆ THỐNG ĐỀ XUẤT

[]{#_Toc231759762 .anchor}3.1. Kiến trúc tổng quát của hệ thống (System Architecture)

Thay vì tiếp cận theo hướng phân tích ảnh tĩnh (Static Image Classification) truyền thống vốn dễ gây ra cảnh báo giả, hệ thống được thiết kế dựa trên mô hình phân tích hành động động học theo không gian và thời gian (Spatio-Temporal Action Recognition). Sơ đồ luồng xử lý dữ liệu (Pipeline) của hệ thống bao gồm 6 giai đoạn chính:

<figure>
<img src="media/image3.jpeg" style="width:6.5in;height:2.16667in" alt="A diagram of a flowchart Description automatically generated" />
<figcaption><p><span id="_Toc231759707" class="anchor"></span>Hình 3. 1. Sơ đồ pipeline đề xuất cho hệ thống</p></figcaption>
</figure>

[]{#_Toc231759763 .anchor}3.2. Thu nhận và tiền xử lý dữ liệu video (DIP Preprocessing)

Để hệ thống hoạt động ổn định trong các điều kiện môi trường thực tế phức tạp (như ánh sáng yếu, bóng râm, nhiễu hạt từ camera giám sát), các kỹ thuật Tiền xử lý ảnh số (Digital Image Processing) được áp dụng trên từng khung hình đầu vào:

Khử nhiễu không gian (Spatial Filtering): Sử dụng bộ lọc Gaussian (Gaussian Blur) để làm mịn ảnh và loại bỏ các nhiễu tần số cao (high-frequency noise) mà không làm mất đi biên (edge) của đối tượng.

Cải thiện độ tương phản (Contrast Enhancement): Áp dụng kỹ thuật Cân bằng lược đồ xám cục bộ (CLAHE - Contrast Limited Adaptive Histogram Equalization) nhằm khôi phục chi tiết vùng tối, giúp hệ thống theo dõi đối tượng tốt hơn trong môi trường thiếu sáng.

[]{#_Toc231759764 .anchor}3.3. Nhận dạng và trích xuất đặc trưng tư thế bằng YOLOv8-Pose

Sau khi tiền xử lý, chuỗi khung hình được đưa qua mô hình YOLOv8-Pose nhằm thực hiện đồng thời hai tác vụ: phát hiện người (Human Detection) và uớc lượng tư thế (Pose Estimation).

**Chiến lược Zero-shot / Pre-trained:** Nhóm nghiên cứu tận dụng trực tiếp bộ trọng số đã được huấn luyện trước (pre-trained weights) trên tập dữ liệu chuẩn COCO. Do mô hình đã đạt độ chính xác (mAP) rất cao, nhóm sẽ đóng băng (freeze) các lớp mạng này mà không cần huấn luyện lại, giúp tiết kiệm tối đa tài nguyên tính toán.

**Đầu ra (Output):** Tại mỗi khung hình, YOLOv8-Pose sẽ nội suy và xuất ra tọa độ không gian 2D $\left( x_{i},y_{i},c_{i} \right)$ của 17 điểm neo (keypoints) quan trọng trên cơ thể người (đầu, vai, khuỷu tay, hông, đầu gối, v.v.), trong đó $c_{i}$ là độ tin cậy (confidence score) của điểm neo đó.

[]{#_Toc231759765 .anchor}3.4. Mô hình hóa chuỗi động học bằng cửa sổ trượt (Sliding Window)

Do hành vi té ngã là một quá trình biến thiên liên tục theo thời gian, việc phân tích đơn lẻ từng khung hình là không khả thi. Nhóm nghiên cứu áp dụng thuật toán Cửa sổ trượt (Sliding Window) để gom cụm dữ liệu:

Cửa sổ có kích thước $T$ frames (ví dụ: $T$ = 30 hoặc 40 frames) sẽ trượt dọc theo video stream với một bước trượt (stride) $S$ được xác định trước.

Kết quả thu được là một chuỗi tư thế (Pose Sequence) mô tả quỹ đạo chuyển động của 17 điểm neo trong một khoảng thời gian ngắn (tương đương 1-2 giây thực tế). Chuỗi dữ liệu không gian - thời gian này sẽ là đầu vào cho bộ phân loại hành vi.

[]{#_Toc231759766 .anchor}3.5. Phân loại hành vi té ngã bằng mạng PoseC3D

Để xử lý chuỗi Pose Sequence và đưa ra quyết định phân loại cuối cùng (Fall hoặc Non-Fall), hệ thống tích hợp kiến trúc mạng nơ-ron tích chập 3D (PoseC3D).

Tạo bản đồ nhiệt 3D (3D Pseudo-heatmaps): Thay vì xử lý trực tiếp mảng tọa độ tọa độ dễ bị nhiễu do hiện tượng che khuất (occlusion) làm mất keypoints, các chuỗi tọa độ được biến đổi thành các bản đồ nhiệt giả 3D có kích thước $H$ \* $W$ \* $T$. Cách tiếp cận này giúp mô hình đạt độ bền vững (robustness) cực cao ngay cả khi người cao tuổi bị che khuất một phần cơ thể bởi đồ nội thất.

**Tinh chỉnh mô hình (Fine-tuning):** Mạng PoseC3D sẽ được huấn luyện tinh chỉnh (fine-tune) trên bộ dữ liệu do nhóm thu thập. Để giải quyết bài toán mất cân bằng dữ liệu (Imbalanced Data) giữa tần suất ngã và sinh hoạt bình thường, hàm mất mát Focal Loss sẽ được sử dụng kết hợp với việc bổ sung các mẫu âm tính khó (Hard Negative Samples) - bao gồm các hành động dễ gây nhầm lẫn như: cúi người nhặt đồ, ngồi phịch xuống ghế sofa, hoặc nằm nghỉ.

*Phương án dự phòng (Fallback Plan):* Trong trường hợp việc suy luận bằng 3D-CNN trên phần cứng thực tế vượt quá khả năng xử lý thời gian thực, module này sẽ được tối ưu hóa hoặc thay thế bằng các kiến trúc chuỗi thời gian nhẹ hơn như Mạng nơ-ron hồi quy bộ nhớ ngắn hạn (LSTM) hoặc Mạng CNN 1D kết hợp cơ chế chú ý (1D-CNN + Attention) nhằm đảm bảo tiêu chí \> 25 FPS.

[]{#_Toc231759767 .anchor}3.6. Module cảnh báo thời gian thực (Real-time Alerting)

Khi hàm kích hoạt Softmax ở lớp đầu ra của PoseC3D dự đoán nhãn **Fall** với xác suất vượt qua ngưỡng (Threshold) được tinh chỉnh thực nghiệm, hệ thống sẽ lập tức tạo ra cờ báo động (Alert Flag). Tín hiệu này kích hoạt âm thanh cảnh báo tại loa giám sát và đánh dấu khung viền đỏ nổi bật trên màn hình của người chăm sóc, đảm bảo thời gian trễ từ lúc chạm sàn đến lúc cảnh báo là tối thiểu.

[]{#_Toc231759768 .anchor}PHẦN 4: THỰC NGHIỆM VÀ ĐÁNH GIÁ HỆ THỐNG

[]{#_Toc231759769 .anchor}4.1. Môi trường, công cụ và tài nguyên phát triển

Do hệ thống tích hợp cả hai giai đoạn: Tiền xử lý ảnh truyền thống (DIP) và Phân tích chuỗi động học bằng mạng học sâu 3D-CNN, môi trường thực nghiệm được cấu hình như sau:

Ngôn ngữ lập trình: Python.

Thư viện xử lý ảnh và trích xuất đặc trưng: \* OpenCV (cv2): Triển khai các thuật toán tiền xử lý ảnh số (Gaussian Blur, CLAHE) trên từng khung hình đầu vào.

Ultralytics: Triển khai mô hình tiền huấn luyện YOLOv8-Pose để trích xuất tọa độ 17 điểm neo (COCO keypoints).

Framework Học sâu (Deep Learning): PyTorch hoặc TensorFlow để xây dựng, tinh chỉnh (fine-tune) và suy luận mạng PoseC3D. Sử dụng thư viện NumPy và Scikit-learn để xử lý ma trận chuỗi thời gian và đánh giá mô hình.

Yêu cầu phần cứng: Môi trường huấn luyện đòi hỏi bộ xử lý đồ họa (GPU) có hỗ trợ kiến trúc CUDA (NVIDIA) để tăng tốc độ tính toán cho các phép tích chập 3D.

[]{#_Toc231759770 .anchor}4.2. Bộ dữ liệu (Dataset) và quy trình chuẩn bị dữ liệu

Dữ liệu huấn luyện và đánh giá được xây dựng qua hai giai đoạn:

Thu thập dữ liệu thô (Raw Videos): Kết hợp các bộ dữ liệu video chuẩn quốc tế (UR Fall Detection Dataset, Multiple Cameras Fall Dataset) với bộ dữ liệu tự thu thập từ camera giám sát trong các môi trường sinh hoạt thực tế.

Tiền xử lý và chuyển đổi dữ liệu (Data Preparation): Toàn bộ video thô sẽ được chạy qua module YOLOv8-Pose để trích xuất tọa độ khung xương. Dữ liệu sau trích xuất được định dạng thành các chuỗi không gian - thời gian (Skeleton Sequences) và áp dụng thuật toán cửa sổ trượt (Sliding Window) để cắt thành các clip ngắn (ví dụ: 30 frames/clip).

Tăng cường dữ liệu (Data Augmentation): Áp dụng các kỹ thuật tăng cường đặc thù cho chuỗi 3D như: xoay góc ngẫu nhiên (random rotation) để giả lập các góc đặt camera khác nhau, và cắt xén trục thời gian (temporal cropping) để tăng tính tổng quát cho mô hình PoseC3D.

[]{#_Toc231759771 .anchor}4.3. Các độ đo đánh giá hiệu năng (Evaluation Metrics)

Hiệu năng của hệ thống được đánh giá toàn diện trên hai khía cạnh: Độ chính xác của thuật toán và Khả năng đáp ứng thời gian thực.

**Đánh giá năng lực phân loại hành vi:** Dựa trên ma trận nhầm lẫn (Confusion Matrix), nhóm nghiên cứu sử dụng các độ đo:

Độ chính xác tổng thể (Accuracy): Tỷ lệ nhận diện đúng trên toàn bộ tập dữ liệu.

Độ nhạy (Recall/Sensitivity): Tỷ lệ phát hiện đúng các ca té ngã thực tế (rất quan trọng trong y tế để không bỏ lọt sự cố).

F1-Score: Trung bình điều hòa giữa Precision và Recall, giúp đánh giá khách quan trong bối cảnh dữ liệu mất cân bằng (Imbalanced Data).

Tỷ lệ báo động giả (False Alarm Rate - FAR): Tần suất hệ thống nhận diện nhầm các hoạt động bình thường thành té ngã.

**Đánh giá hiệu năng hệ thống (System Performance):**

Tốc độ khung hình (FPS): Đo lường tổng thời gian nội suy của toàn bộ luồng pipeline (từ bước tiền xử lý DIP → YOLOv8 → PoseC3D) nhằm đảm bảo tiêu chí xử lý ≥ 25 FPS.

Độ trễ cảnh báo (Latency): Thời gian tính từ khi kết thúc hành vi ngã trên video đến khi hệ thống phát tín hiệu cảnh báo.

[]{#_Toc231759772 .anchor}4.4. Kịch bản thử nghiệm và đánh giá độ bền vững (Test cases & Robustness)

Mô hình sau khi fine-tune sẽ được chạy thực nghiệm qua các kịch bản có độ khó tăng dần để kiểm chứng tính bền vững của luồng thuật toán:

Kịch bản Positive (Té ngã thực tế): Đánh giá các véc-tơ chuyển động ngã có tính đa dạng sinh học cao (ngã chúi về phía trước do vấp, ngã ngửa do trượt chân, ngã khụy gối do choáng váng).

Kịch bản Hard-Negative (Hành vi gây nhiễu): Thử nghiệm trên các hành vi sinh hoạt hàng ngày (ADLs) có quỹ đạo trọng tâm hạ thấp đột ngột (như ngồi phịch xuống ghế sofa, cúi gập người thắt dây giày, nhặt đồ rơi, hoặc chủ động nằm ra sàn/giường). Đây là kịch bản để chứng minh sự vượt trội của mạng 3D-CNN so với phương pháp xét ngưỡng Bounding Box truyền thống.

Kịch bản Thử thách môi trường (Environmental Robustness): Thiếu sáng/Ngược sáng: Kiểm tra hiệu quả của module Tiền xử lý (Cân bằng lược đồ xám CLAHE) trước khi đưa vào YOLO.

Bị che khuất (Occlusion): Thử nghiệm kịch bản đối tượng bị che khuất một phần cơ thể (bởi bàn, ghế, tủ) khi ngã để kiểm chứng khả năng nội suy bù đắp điểm neo (pseudo-heatmaps) của thuật toán PoseC3D.

# References

| \[1\] | R. G. Stefanacci và J. R. Wilkinson, "Té ngã ở người cao tuổi," August 2025. \[Trực tuyến\]. Available: https://www.msdmanuals.com/vi/professional/l%C3%A3o-khoa/t%C3%A9-ng%C3%A3-%E1%BB%9F-ng%C6%B0%E1%BB%9Di-cao-tu%E1%BB%95i/t%C3%A9-ng%C3%A3-%E1%BB%9F-ng%C6%B0%E1%BB%9Di-cao-tu%E1%BB%95i. \[Đã truy cập 05 06 2026\]. |
|----|----|
| \[2\] | N. K. K. Võ, "Vì sao và nơi nào người cao tuổi dễ bị té ngã?," 22 July 2024. \[Trực tuyến\]. Available: https://www.vinmec.com/vie/bai-viet/vi-sao-va-noi-nao-nguoi-cao-tuoi-de-bi-te-nga-vi. \[Đã truy cập 05 06 2026\]. |
| \[3\] | A. P. Kaur, E. Nsugbe, A. Drahota, M. Oldfield, I. Mohagheghian and R. A. Sporea, \"State-of-the-art fall detection techniques with emphasis on floor-based systems---A review,\" *Biomedical Engineering Advances,* vol. 9, p. 100179, 2025. |
| \[4\] | D. Hrubý, E. Hrubá and M. Černý, \"Research of Fall Detection and Fall Prevention Technologies: A Review,\" *Sensors,* vol. 26, p. 1192, 2026. |
| \[5\] | K. Perli´nski, A. Falty´nski and A. ´. Swietlicka, \"HumanFall Detection with Infrared Imaging: A Comparison of Graph Convolutional Networks and YOLO,\" *sensors,* vol. 26, p. 2794, 2026. |
| \[6\] | H. V. Dũng, \"Chapter 1. Introduction\". |
| \[7\] | H. V. Dũng, \"Chapter 2. Image Enhancement in Spatial Domain\". |
