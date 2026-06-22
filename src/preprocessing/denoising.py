import cv2
import numpy as np

def apply_bilateral_filter(frame: np.ndarray, d: int = 5, sigma_color: float = 75.0, sigma_space: float = 75.0) -> np.ndarray:
    """Khử nhiễu bảo toàn biên bằng Bilateral Filter.
    
    Tốt cho việc làm mịn các vùng phẳng (skin, background) mà không làm mờ biên (edges).
    Rất thích hợp cho YOLOv8-Pose nhận diện cấu trúc xương.
    """
    return cv2.bilateralFilter(frame, d, sigma_color, sigma_space)

def apply_nlmeans_denoising(frame: np.ndarray, h: float = 10.0, template_window: int = 7, search_window: int = 21) -> np.ndarray:
    """Khử nhiễu chất lượng cao bằng Non-Local Means Denoising.
    
    Phương pháp này tìm các khối tương đồng trên toàn bộ ảnh để tính trung bình.
    Khử nhiễu cực tốt nhưng có độ trễ lớn (thường dùng trong quality mode).
    """
    # fastNlMeansDenoisingColored tự động nhận diện ảnh 3 kênh BGR/RGB
    return cv2.fastNlMeansDenoisingColored(
        frame,
        None,
        h=h,
        hColor=h,
        templateWindowSize=template_window,
        searchWindowSize=search_window
    )

def apply_median_filter(frame: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Khử nhiễu muối tiêu bằng bộ lọc Median."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(frame, kernel_size)

def denoise_frame(frame: np.ndarray, config: dict) -> np.ndarray:
    """Hàm điều phối chọn bộ lọc khử nhiễu dựa trên cấu hình."""
    if not config.get("enabled", True):
        return frame
        
    method = config.get("method", "bilateral")
    
    if method == "bilateral":
        return apply_bilateral_filter(
            frame,
            d=config.get("bilateral_d", 9),
            sigma_color=config.get("sigma_color", 75.0),
            sigma_space=config.get("sigma_space", 75.0)
        )
    elif method == "nlmeans":
        return apply_nlmeans_denoising(
            frame,
            h=config.get("nlmeans_h", 10.0)
        )
    elif method == "median":
        return apply_median_filter(
            frame,
            kernel_size=config.get("median_kernel", 3)
        )
    return frame
