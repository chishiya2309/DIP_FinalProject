# Module: Ước lượng tư thế người (Pose Estimation)
#
# Chức năng:
#   - Wrapper cho YOLOv8-Pose (pre-trained COCO weights, frozen layers)
#   - Phát hiện người (Human Detection) + trích xuất 17 COCO keypoints
#   - Output: (x_i, y_i, c_i) cho mỗi keypoint trên mỗi frame
#
# Pipeline position: Stage 2 - Preprocessing → [THIS] → Sliding Window
