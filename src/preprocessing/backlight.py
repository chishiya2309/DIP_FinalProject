import cv2
import numpy as np

def apply_gamma_correction(frame: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    """Áp dụng Gamma Correction để làm sáng các vùng tối.
    
    Công thức: Output = 255 * (Input / 255) ^ (1/gamma)
    - gamma > 1.0: Làm sáng vùng tối (thích hợp cho ngược sáng).
    - gamma < 1.0: Làm tối ảnh quá sáng.
    """
    # Xây dựng bảng tra cứu (Lookup Table - LUT) để tăng tốc độ xử lý
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(frame, table)

def apply_single_scale_retinex(frame: np.ndarray, sigma: float = 30.0) -> np.ndarray:
    """Khôi phục chi tiết vùng tối bằng Single Scale Retinex (SSR).
    
    Để tránh làm sai lệch màu sắc, ta chuyển sang hệ màu LAB, áp dụng SSR
    trên kênh L (Lightness), sau đó chuyển ngược lại BGR.
    """
    # Chuyển sang LAB
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Ép kiểu sang float32 để tính toán logarit
    l_float = l_channel.astype(np.float32)
    
    # Tính thành phần chiếu sáng (illumination) bằng Gaussian Blur
    blur = cv2.GaussianBlur(l_float, (0, 0), sigma)
    
    # Công thức Retinex: log(Reflection) = log(Intensity) - log(Illumination)
    # Thêm 1.0 để tránh log(0)
    l_log = np.log10(l_float + 1.0)
    blur_log = np.log10(blur + 1.0)
    retinex = l_log - blur_log
    
    # Chuẩn hóa về [0, 255]
    min_val = np.min(retinex)
    max_val = np.max(retinex)
    
    if max_val != min_val:
        retinex_norm = (retinex - min_val) / (max_val - min_val) * 255.0
    else:
        retinex_norm = l_float
        
    l_enhanced = np.clip(retinex_norm, 0, 255).astype(np.uint8)
    
    # Trộn lại và chuyển về BGR
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

def detect_backlight(
    frame: np.ndarray,
    threshold: float = 0.3,
    ratio_threshold: float = 12.0
) -> bool:
    """Phat hien xem frame co bi nguoc sang hay khong.
    
    Su dung ti le do sang giua vung toi nhat va vung sang nhat.
    - ratio_threshold = 12.0 (tang tu 6.0) de tranh false positive
      tren depth map camera (depth map co tuong phan cao tu nhien).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    overall_brightness = float(np.mean(gray))
    
    # Bao ve: Neuphong qua sang (brightness > 160), khong phai nguoc sang
    if overall_brightness > 160.0:
        return False
    
    pixels = gray.flatten()
    sorted_pixels = np.sort(pixels)
    
    n_pixels = len(sorted_pixels)
    k = int(n_pixels * 0.1)
    
    if k == 0:
        return False
        
    dark_mean = np.mean(sorted_pixels[:k])
    bright_mean = np.mean(sorted_pixels[-k:])
    
    if dark_mean == 0:
        dark_mean = 1.0
        
    ratio = bright_mean / dark_mean
    
    # Nguong 12.0 (tang tu 6.0):
    # - Depth map binh thuong: ratio > 50 (vi co gia tri 0 = chua do duoc)
    # - Nguoc sang that su: ratio ~ 8-15 voi overall brightness < 100
    # - Ket hop them dieu kien brightness < 120 de loc depth map
    return ratio > ratio_threshold and overall_brightness < 120.0

def correct_backlight(frame: np.ndarray, config: dict) -> np.ndarray:
    """Hàm điều phối xử lý ngược sáng dựa trên cấu hình."""
    if not config.get("enabled", True):
        return frame
        
    method = config.get("method", "gamma")
    
    if method == "gamma":
        return apply_gamma_correction(frame, gamma=config.get("gamma", 1.5))
    elif method == "retinex":
        return apply_single_scale_retinex(frame, sigma=config.get("retinex_sigma", 30.0))
        
    return frame
