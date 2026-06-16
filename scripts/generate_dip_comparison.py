import os
import sys
import cv2
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.adaptive_pipeline import enhance_frame
from src.preprocessing.config_manager import EnhancementConfigManager

def create_low_light_noisy_frame(frame):
    # Simulate low light and noise to make enhancement obvious
    # Darken
    dark_frame = cv2.convertScaleAbs(frame, alpha=0.4, beta=0)
    
    # Add noise
    noise = np.random.normal(0, 15, dark_frame.shape).astype(np.int16)
    noisy_frame = np.clip(dark_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return noisy_frame

def main():
    video_path = "../data/cam1.avi"
    output_path = "../docs/DIP_Enhancement_Comparison.jpg"
    
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        # Create a synthetic image if video not found
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        cv2.putText(img, "Test Image", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        frame = img
    else:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 50)  # Read frame 50
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Failed to read frame.")
            return

    # Create a challenging frame
    test_frame = create_low_light_noisy_frame(frame)
    
    # Initialize Config Manager
    config_manager = EnhancementConfigManager()
    
    # Run the adaptive enhancement
    print("Running DIP Enhancement...")
    enhanced_frame = enhance_frame(test_frame, config_manager)
    
    # Add labels
    cv2.putText(test_frame, "Before DIP (Low Light & Noisy)", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(enhanced_frame, "After DIP Enhancement", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Concatenate side by side
    comparison = np.hstack((test_frame, enhanced_frame))
    
    # Save
    cv2.imwrite(output_path, comparison)
    print(f"Comparison image saved to: {output_path}")

if __name__ == "__main__":
    main()
