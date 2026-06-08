# Module: Pipeline tích hợp End-to-End
#
# Chức năng:
#   - Kết nối toàn bộ 5 module thành luồng xử lý liên tục
#   - Quản lý video stream (camera/file) input
#   - Điều phối: Preprocessing → YOLOv8-Pose → Sliding Window → PoseC3D → Alert
#   - Đo lường FPS tổng thể (target: ≥ 25 FPS)
