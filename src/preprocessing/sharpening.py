import cv2
import numpy as np

def apply_unsharp_mask(frame: np.ndarray, strength: float = 1.5, blur_kernel: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Làm rõ nét ảnh bằng kỹ thuật Unsharp Masking.
    
    Công thức: Output = Original + strength * (Original - GaussianBlur)
    Giúp tăng cường độ tương phản ở các đường biên mà không phóng đại quá mức nhiễu hạt.
    """
    # Đảm bảo kernel size lẻ
    if blur_kernel % 2 == 0:
        blur_kernel += 1
        
    blurred = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), sigma)
    high_pass = frame.astype(np.float32) - blurred.astype(np.float32)
    sharpened = frame.astype(np.float32) + strength * high_pass
    
    # Clip và chuyển đổi về uint8 để tránh tràn số (overflow/underflow)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def apply_laplacian_sharpen(frame: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Làm rõ nét bằng bộ lọc Laplacian (Spatial Domain Second Derivative).
    
    Công thức: Output = Original - strength * Laplacian(Original)
    Nhạy cảm hơn với chi tiết nhỏ, nhưng có thể làm tăng nhiễu tần số cao.
    """
    # Dùng kiểu dữ liệu CV_32F để tránh tràn số khi tính đạo hàm cấp 2
    laplacian = cv2.Laplacian(frame, cv2.CV_32F)
    sharpened = frame.astype(np.float32) - strength * laplacian
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def sharpen_frame(frame: np.ndarray, config: dict) -> np.ndarray:
    """Hàm điều phối chọn bộ lọc làm nét dựa trên cấu hình."""
    if not config.get("enabled", True):
        return frame
        
    method = config.get("method", "unsharp_mask")
    
    if method == "unsharp_mask":
        return apply_unsharp_mask(
            frame,
            strength=config.get("strength", 1.5),
            blur_kernel=config.get("blur_kernel", 5),
            sigma=config.get("sigma", 1.0)
        )
    elif method == "laplacian":
        return apply_laplacian_sharpen(
            frame,
            strength=config.get("strength", 1.0)
        )
    return frame
