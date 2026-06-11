import cv2
import numpy as np
from .sharpening import sharpen_frame
from .denoising import denoise_frame
from .backlight import correct_backlight, detect_backlight
from .white_balance import balance_colors, apply_clamped_gray_world
from .filters import apply_clahe

def analyze_frame_conditions(frame: np.ndarray) -> dict:
    """Phan tich cac dieu kien anh sang, nhieu va do net cua frame.
    
    Returns:
        Dict cac chi so dac trung:
        - brightness: Do sang trung binh (0-255)
        - noise_level: Uoc luong muc do nhieu (std cua diff Gaussian)
        - sharpness: Do net (phuong sai Laplacian)
        - color_cast: Muc do lech mau (std giua trung binh 3 kenh R,G,B)
        - has_backlight: Co bi nguoc sang hay khong (bool)
        - contrast_ratio: std(L) / mean(L) - do tuong phan tuong doi
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. Do sang trung binh
    brightness = float(np.mean(gray))
    
    # 2. Do net (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian.var())
    
    # 3. Muc do nhieu (std cua noise component)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    noise_diff = cv2.absdiff(gray, blurred)
    noise_level = float(np.std(noise_diff))
    
    # 4. Lech mau (color cast)
    b, g, r = cv2.split(frame)
    mean_b = np.mean(b)
    mean_g = np.mean(g)
    mean_r = np.mean(r)
    color_cast = float(np.std([mean_b, mean_g, mean_r]))
    
    # 5. Nguoc sang
    has_backlight = detect_backlight(frame)
    
    # 6. Tuong phan tuong doi: std(L) / mean(L)
    # - Anh tot: contrast_ratio > 0.40
    # - Anh thieu tuong phan: contrast_ratio < 0.35
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)
    contrast_ratio = float(np.std(l_channel)) / (float(np.mean(l_channel)) + 1e-6)
    
    return {
        "brightness": brightness,
        "sharpness": sharpness,
        "noise_level": noise_level,
        "color_cast": color_cast,
        "has_backlight": has_backlight,
        "contrast_ratio": contrast_ratio,
    }

def run_calibration(frame: np.ndarray, config_manager) -> bool:
    """Hieu chuan tu dong dua tren 30 frames dau tien.
    
    Ghi lai baseline brightness va noise cua moi truong camera.
    """
    if not hasattr(config_manager, "calibration_history"):
        config_manager.calibration_history = []
        config_manager.is_calibrated = False
        
    if config_manager.is_calibrated:
        return True
        
    conditions = analyze_frame_conditions(frame)
    config_manager.calibration_history.append(conditions)
    
    if len(config_manager.calibration_history) >= 30:
        history = config_manager.calibration_history
        avg_brightness = np.mean([h["brightness"] for h in history])
        avg_noise = np.mean([h["noise_level"] for h in history])
        avg_contrast = np.mean([h["contrast_ratio"] for h in history])
        
        config_manager.is_calibrated = True
        print(f"\n[Auto-Calibration] Hoan tat hieu chuan camera!")
        print(f"  - Baseline Brightness : {avg_brightness:.2f}")
        print(f"  - Baseline Noise      : {avg_noise:.2f}")
        print(f"  - Baseline Contrast   : {avg_contrast:.3f}")
        
    return config_manager.is_calibrated

def enhance_frame(frame: np.ndarray, config_manager) -> np.ndarray:
    """Pipeline xu ly anh thich nghi chinh (Conservative Enhancement).
    
    Nguyen tac: "First, do no harm" — moi bo loc chi kich hoat khi
    do duoc rang anh thuc su can no, tranh over-processing lam giam chat luong.
    """
    config = config_manager.config["enhancement"]
    
    # Hieu chuan tu dong neu bat auto_detect
    if config.get("auto_detect", True) and not getattr(config_manager, "is_calibrated", False):
        run_calibration(frame, config_manager)
        
    # Tao ban sao local de tranh thay doi config goc
    # Dung `or {}` de an toan khi YAML bi loi indentation tra ve None
    sharpen_cfg = (config.get("sharpening") or {}).copy()
    denoise_cfg = (config.get("denoising") or {}).copy()
    backlight_cfg = (config.get("backlight") or {}).copy()
    wb_cfg = (config.get("white_balance") or {}).copy()
    clahe_cfg = (config.get("clahe") or {}).copy()
    
    # Phan tich dieu kien frame hien tai
    conditions = analyze_frame_conditions(frame)
    
    # --- Quyet dinh mac dinh tu config ---
    should_denoise = denoise_cfg.get("enabled", True)
    should_correct_backlight = backlight_cfg.get("enabled", True)
    should_balance_color = wb_cfg.get("enabled", True)
    should_sharpen = sharpen_cfg.get("enabled", True)
    should_clahe = True  # Se duoc ghi de neu auto_detect bat

    # --- Logic thich nghi (Conservative Adaptive Rules) ---
    if config.get("auto_detect", True):
        
        # 1. KHU NHIEU: Chi khi noise thuc su cao
        should_denoise = (
            (conditions["noise_level"] > 2.2) or
            (conditions["brightness"] < 60.0)
        )
        # Nhieu rat nang -> tang thong so Bilateral
        if should_denoise and conditions["noise_level"] > 3.5:
            denoise_cfg["bilateral_d"] = 7
            denoise_cfg["sigma_color"] = 100
            denoise_cfg["sigma_space"] = 100

        # 2. NGUOC SANG: Chi khi phat hien nguoc sang that su
        # detect_backlight() da duoc nang cap nguong len 12.0 + guard brightness < 120
        # de tranh false positive tren depth map (depth map co tuong phan cao tu nhien)
        should_correct_backlight = conditions["has_backlight"]
        
        # Phong toi hon nguong an toan (< 70) nhung khong nguoc sang
        # Chi boost sang khi THUC SU TOI de khong lam hong depth map
        if not should_correct_backlight and conditions["brightness"] < 70.0:
            should_correct_backlight = True
            backlight_cfg["method"] = "gamma"
            gamma_val = 1.0 + (70.0 - conditions["brightness"]) / 350.0
            backlight_cfg["gamma"] = float(np.clip(gamma_val, 1.0, 1.15))

        # 3. CAN BANG TRANG: Chi khi lech mau thuc su dang ke
        # Nguong 30.0 (tang tu 25.0) vi White Patch co the tao fringing
        # tren video Depth+RGB split do max pixel duoc tinh toan tren toan frame
        cast_threshold = wb_cfg.get("cast_threshold", 30.0)
        should_balance_color = conditions["color_cast"] > cast_threshold

        # 4. LAM NET: Chi khi anh thuc su mo VA nhieu thap
        if conditions["noise_level"] >= 3.5:
            # Nhieu qua nang -> khong sharpen de tranh phong dai hat nhieu
            should_sharpen = False
        else:
            should_sharpen = conditions["sharpness"] < sharpen_cfg.get("sharpness_threshold", 150.0)
            # Nhieu trung binh -> giam suc manh sharpen de an toan
            if should_sharpen and conditions["noise_level"] > 2.2:
                sharpen_cfg["strength"] = min(sharpen_cfg.get("strength", 1.5), 0.6)

        # 5. CLAHE: Chi khi anh thieu tuong phan thuc su
        # Anh da co tuong phan tot -> CLAHE se lam phang va lam toi anh
        clahe_threshold = clahe_cfg.get("contrast_threshold", 0.35)
        should_clahe = conditions["contrast_ratio"] < clahe_threshold

    # --- THUC THI PIPELINE ---
    enhanced = frame.copy()
    
    # Buoc 1: Khu nhieu (truoc tien de tranh khuyech dai nhieu o cac buoc sau)
    if should_denoise:
        enhanced = denoise_frame(enhanced, denoise_cfg)
        
    # Buoc 2: Xu ly nguoc sang / tang sang
    if should_correct_backlight:
        enhanced = correct_backlight(enhanced, backlight_cfg)
        
    # Buoc 3: Can bang mau sac (dung clamped version qua balance_colors)
    if should_balance_color:
        enhanced = balance_colors(enhanced, wb_cfg)
        
    # Buoc 4: Lam ro net
    if should_sharpen:
        enhanced = sharpen_frame(enhanced, sharpen_cfg)
        
    # Buoc 5: CLAHE - CHI chay khi anh thieu tuong phan thuc su
    if should_clahe:
        clip_limit = clahe_cfg.get("clip_limit", 1.5)
        tile_grid_size = tuple(clahe_cfg.get("tile_grid_size", [16, 16]))
        
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_enhanced = clahe_obj.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2BGR)
    
    return enhanced
