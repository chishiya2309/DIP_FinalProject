import cv2
import numpy as np

def apply_gray_world(frame: np.ndarray) -> np.ndarray:
    """Cân bằng trắng Gray World không giới hạn (dùng cho ảnh studio/controlled).
    
    Cảnh báo: Với ảnh indoor phòng thường (ánh đèn vàng, tường beige),
    hàm này có thể làm lệch màu sang lạnh. Dùng apply_clamped_gray_world thay thế.
    """
    b, g, r = cv2.split(frame)
    
    mean_b = np.mean(b)
    mean_g = np.mean(g)
    mean_r = np.mean(r)
    
    if mean_b == 0 or mean_g == 0 or mean_r == 0:
        return frame
        
    mean_gray = (mean_b + mean_g + mean_r) / 3.0
    kb = mean_gray / mean_b
    kg = mean_gray / mean_g
    kr = mean_gray / mean_r
    
    b_new = np.clip(b * kb, 0, 255).astype(np.uint8)
    g_new = np.clip(g * kg, 0, 255).astype(np.uint8)
    r_new = np.clip(r * kr, 0, 255).astype(np.uint8)
    
    return cv2.merge([b_new, g_new, r_new])

def apply_clamped_gray_world(
    frame: np.ndarray,
    max_gain: float = 1.25,
    min_gain: float = 0.80
) -> np.ndarray:
    """Cân bằng trắng Gray World an toàn với gain bị giới hạn.
    
    Giới hạn mức điều chỉnh gain mỗi kênh trong khoảng [min_gain, max_gain]
    để tránh over-correction cắt bỏ quá nhiều kênh màu (đặc biệt kênh đỏ
    với ánh đèn vàng trong nhà). Phù hợp cho webcam feed thời gian thực.
    
    Args:
        frame: Ảnh BGR đầu vào.
        max_gain: Hệ số tối đa được phép tăng một kênh (mặc định 1.25 = +25%).
        min_gain: Hệ số tối thiểu được phép giảm một kênh (mặc định 0.80 = -20%).
    """
    b, g, r = cv2.split(frame)
    
    mean_b = float(np.mean(b))
    mean_g = float(np.mean(g))
    mean_r = float(np.mean(r))
    
    if mean_b < 1.0 or mean_g < 1.0 or mean_r < 1.0:
        return frame
        
    mean_gray = (mean_b + mean_g + mean_r) / 3.0
    
    # Clamp từng gain trong [min_gain, max_gain] để tránh phá hủy màu sắc
    kb = float(np.clip(mean_gray / mean_b, min_gain, max_gain))
    kg = float(np.clip(mean_gray / mean_g, min_gain, max_gain))
    kr = float(np.clip(mean_gray / mean_r, min_gain, max_gain))
    
    b_new = np.clip(b.astype(np.float32) * kb, 0, 255).astype(np.uint8)
    g_new = np.clip(g.astype(np.float32) * kg, 0, 255).astype(np.uint8)
    r_new = np.clip(r.astype(np.float32) * kr, 0, 255).astype(np.uint8)
    
    return cv2.merge([b_new, g_new, r_new])

def apply_white_patch(frame: np.ndarray) -> np.ndarray:
    """Cân bằng trắng bằng phương pháp White Patch (Max-RGB).
    
    Giả định điểm sáng nhất trong ảnh chính là màu trắng (255, 255, 255).
    """
    b, g, r = cv2.split(frame)
    
    max_b = np.max(b)
    max_g = np.max(g)
    max_r = np.max(r)
    
    if max_b == 0 or max_g == 0 or max_r == 0:
        return frame
        
    kb = 255.0 / max_b
    kg = 255.0 / max_g
    kr = 255.0 / max_r
    
    b_new = np.clip(b * kb, 0, 255).astype(np.uint8)
    g_new = np.clip(g * kg, 0, 255).astype(np.uint8)
    r_new = np.clip(r * kr, 0, 255).astype(np.uint8)
    
    return cv2.merge([b_new, g_new, r_new])

def balance_colors(frame: np.ndarray, config: dict) -> np.ndarray:
    """Hàm điều phối cân bằng màu sắc dựa trên cấu hình."""
    if not config.get("enabled", True):
        return frame
        
    method = config.get("method", "gray_world")
    max_gain = config.get("max_gain", 1.25)
    min_gain = config.get("min_gain", 0.80)
    
    if method == "gray_world":
        # Mặc định dùng clamped version để an toàn với ảnh indoor
        return apply_clamped_gray_world(frame, max_gain=max_gain, min_gain=min_gain)
    elif method == "gray_world_unclamped":
        return apply_gray_world(frame)
    elif method == "white_patch":
        return apply_white_patch(frame)
        
    return frame
