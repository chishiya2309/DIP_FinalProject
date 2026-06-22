"""
Module tiền xử lý ảnh số (DIP Preprocessing Filters).

Triển khai 2 kỹ thuật xử lý trong miền không gian (Spatial Domain):
  1. Gaussian Blur - Khử nhiễu tần số cao, bảo toàn biên
  2. CLAHE - Cân bằng lược đồ xám cục bộ, cải thiện tương phản vùng tối

Pipeline position: Video Input → [THIS] → Pose Estimation
"""

import cv2
import numpy as np


def apply_gaussian_blur(frame: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Áp dụng bộ lọc Gaussian để khử nhiễu tần số cao.

    Bộ lọc Gaussian tính trung bình có trọng số của các pixel lân cận,
    giúp loại bỏ nhiễu hạt (Gaussian noise) từ camera giám sát trong
    điều kiện thiếu sáng, đồng thời bảo toàn các đường biên (edges)
    cấu trúc cơ thể người.

    Args:
        frame: Ảnh BGR đầu vào từ OpenCV (H, W, 3).
        kernel_size: Kích thước kernel Gaussian (phải là số lẻ).
        sigma: Độ lệch chuẩn của phân phối Gaussian.

    Returns:
        Ảnh BGR đã được làm mịn.
    """
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma)


def apply_clahe(frame: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """Áp dụng CLAHE để cải thiện độ tương phản trong điều kiện ánh sáng phức tạp.

    Thay vì cân bằng lược đồ toàn cục (dễ gây over-enhancement), CLAHE
    chia ảnh thành các ô nhỏ (tiles) và cân bằng cục bộ với giới hạn
    tương phản (clip limit), giúp:
      - Khôi phục chi tiết vùng tối (ngược sáng, góc khuất)
      - Giữ nguyên màu sắc tự nhiên (chỉ xử lý kênh Lightness)
      - Tránh bão hòa quá mức ở vùng sáng

    Chiến lược: BGR → LAB → CLAHE trên kênh L → LAB → BGR

    Args:
        frame: Ảnh BGR đầu vào từ OpenCV (H, W, 3).
        clip_limit: Ngưỡng giới hạn tương phản CLAHE.
        tile_grid_size: Kích thước lưới chia ô (rows, cols).

    Returns:
        Ảnh BGR đã được cải thiện tương phản.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def preprocess_frame(frame: np.ndarray, config: dict) -> np.ndarray:
    """Áp dụng toàn bộ pipeline tiền xử lý lên một frame.

    Thứ tự xử lý: Gaussian Blur → CLAHE
    (Khử nhiễu trước, rồi mới cải thiện tương phản để tránh khuếch đại nhiễu)

    Args:
        frame: Ảnh BGR đầu vào.
        config: Dict chứa tham số preprocessing từ configs/default.yaml.

    Returns:
        Ảnh BGR đã qua tiền xử lý.
    """
    gaussian_cfg = config.get("gaussian_blur", {})
    clahe_cfg = config.get("clahe", {})

    frame = apply_gaussian_blur(
        frame,
        kernel_size=gaussian_cfg.get("kernel_size", 5),
        sigma=gaussian_cfg.get("sigma", 1.0),
    )
    frame = apply_clahe(
        frame,
        clip_limit=clahe_cfg.get("clip_limit", 2.0),
        tile_grid_size=tuple(clahe_cfg.get("tile_grid_size", [8, 8])),
    )
    return frame
