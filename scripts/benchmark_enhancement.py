import os
import sys
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import (
    EnhancementConfigManager,
    apply_unsharp_mask,
    apply_laplacian_sharpen,
    apply_bilateral_filter,
    apply_nlmeans_denoising,
    apply_median_filter,
    apply_gamma_correction,
    apply_single_scale_retinex,
    apply_gray_world,
    apply_white_patch,
    enhance_frame
)

def benchmark_filter(name, func, frame, *args, **kwargs):
    # Warmup
    for _ in range(5):
        _ = func(frame, *args, **kwargs)
        
    start_time = time.perf_counter()
    iterations = 50 if "nlmeans" not in name.lower() else 5  # Reduce loops for NLMeans because it is slow
    for _ in range(iterations):
        _ = func(frame, *args, **kwargs)
    end_time = time.perf_counter()
    
    avg_time = (end_time - start_time) / iterations * 1000  # ms
    print(f"  {name:<30} : {avg_time:6.2f} ms")
    return avg_time

def main():
    print("=" * 60)
    print("        VIDEO ENHANCEMENT BENCHMARK SCRIPT")
    print("=" * 60)
    
    # Test frame at standard resolution (640x480)
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    print(f"  Test frame size: {frame.shape[1]}x{frame.shape[0]}\n")
    
    # 1. Sharpening
    print("[1] SHARPENING")
    benchmark_filter("Unsharp Mask (strength=1.5)", apply_unsharp_mask, frame, strength=1.5)
    benchmark_filter("Laplacian Sharpen (strength=1.0)", apply_laplacian_sharpen, frame, strength=1.0)
    print()
    
    # 2. Denoising
    print("[2] DENOISING")
    benchmark_filter("Bilateral Filter (d=5)", apply_bilateral_filter, frame, d=5, sigma_color=75, sigma_space=75)
    benchmark_filter("Median Filter (k=3)", apply_median_filter, frame, kernel_size=3)
    benchmark_filter("NLMeans Denoise (h=10)", apply_nlmeans_denoising, frame, h=10.0)
    print()
    
    # 3. Backlight
    print("[3] BACKLIGHT CORRECTION")
    benchmark_filter("Gamma Correction (g=1.5)", apply_gamma_correction, frame, gamma=1.5)
    benchmark_filter("Single Scale Retinex (s=30)", apply_single_scale_retinex, frame, sigma=30.0)
    print()
    
    # 4. White Balance
    print("[4] WHITE BALANCE")
    benchmark_filter("Gray World WB", apply_gray_world, frame)
    benchmark_filter("White Patch WB", apply_white_patch, frame)
    print()
    
    # 5. Full Pipeline
    print("[5] FULL PIPELINE")
    manager = EnhancementConfigManager()
    
    # Disable auto-detect to measure constant execution of all enabled stages
    manager.update_param("enhancement", "auto_detect", False)
    benchmark_filter("Full Adaptive Pipeline (Static)", enhance_frame, frame, manager)
    
    # Re-enable auto-detect for normal adaptive mode
    manager.update_param("enhancement", "auto_detect", True)
    manager.is_calibrated = True
    benchmark_filter("Adaptive Pipeline (Auto Mode)", enhance_frame, frame, manager)
    
    print("-" * 60)
    print("  Performance evaluation complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
