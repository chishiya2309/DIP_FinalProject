# Module: Tiền xử lý ảnh số (DIP Preprocessing)
#
# Chức năng:
#   - Gaussian Blur: Khử nhiễu tần số cao, bảo toàn biên cấu trúc cơ thể
#   - CLAHE: Cân bằng lược đồ xám cục bộ, cải thiện độ tương phản vùng tối
#
# Pipeline position: Stage 1 - Video Input → [THIS] → Pose Estimation

from src.preprocessing.filters import apply_clahe, apply_gaussian_blur, preprocess_frame
from src.preprocessing.config_manager import EnhancementConfigManager
from src.preprocessing.sharpening import sharpen_frame, apply_unsharp_mask, apply_laplacian_sharpen
from src.preprocessing.denoising import denoise_frame, apply_bilateral_filter, apply_nlmeans_denoising, apply_median_filter
from src.preprocessing.backlight import correct_backlight, detect_backlight, apply_gamma_correction, apply_single_scale_retinex
from src.preprocessing.white_balance import balance_colors, apply_gray_world, apply_white_patch, apply_clamped_gray_world
from src.preprocessing.adaptive_pipeline import enhance_frame, analyze_frame_conditions

