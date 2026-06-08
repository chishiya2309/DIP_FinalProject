# Module: Cửa sổ trượt (Sliding Window)
#
# Chức năng:
#   - Phân cắt chuỗi keypoints thành các clip ngắn (T frames, stride S)
#   - Gom cụm dữ liệu thành Skeleton Sequences (Pose Sequences)
#   - VD: T=30 frames/clip ≈ 1-2 giây thực tế
#
# Pipeline position: Stage 3 - Pose Estimation → [THIS] → PoseC3D
