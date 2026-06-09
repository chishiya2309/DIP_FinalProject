# Module: Tiền xử lý ảnh số (DIP Preprocessing)
#
# Chức năng:
#   - Gaussian Blur: Khử nhiễu tần số cao, bảo toàn biên cấu trúc cơ thể
#   - CLAHE: Cân bằng lược đồ xám cục bộ, cải thiện độ tương phản vùng tối
#
# Pipeline position: Stage 1 - Video Input → [THIS] → Pose Estimation

from src.preprocessing.filters import apply_clahe, apply_gaussian_blur, preprocess_frame
