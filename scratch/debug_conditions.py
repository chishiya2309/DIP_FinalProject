import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.adaptive_pipeline import analyze_frame_conditions

# Tạo ảnh nhiễu ngẫu nhiên
clean = np.ones((480, 640, 3), dtype=np.uint8) * 128
cv2.rectangle(clean, (100, 100), (540, 380), (200, 200, 200), -1)

# Thêm nhiễu nhẹ
noisy_light = clean.copy()
noise = np.random.normal(0, 3, clean.shape).astype(np.int16)
noisy_light = np.clip(noisy_light.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# Thêm nhiễu nặng (như webcam của user)
noisy_heavy = clean.copy()
noise = np.random.normal(0, 15, clean.shape).astype(np.int16)
noisy_heavy = np.clip(noisy_heavy.astype(np.int16) + noise, 0, 255).astype(np.uint8)

cond_clean = analyze_frame_conditions(clean)
cond_light = analyze_frame_conditions(noisy_light)
cond_heavy = analyze_frame_conditions(noisy_heavy)

print("=== Clean Frame ===")
for k, v in cond_clean.items():
    print(f"  {k}: {v}")
    
print("\n=== Light Noisy Frame ===")
for k, v in cond_light.items():
    print(f"  {k}: {v}")
    
print("\n=== Heavy Noisy Frame ===")
for k, v in cond_heavy.items():
    print(f"  {k}: {v}")
