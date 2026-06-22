"""
Script visualize skeleton keypoints lên video frame để kiểm tra chất lượng.

Chức năng:
  1. So sánh Original vs Preprocessed (Gaussian + CLAHE)
  2. Vẽ skeleton (17 COCO keypoints + connections) lên frame
  3. Xuất ảnh PNG cho các frame đại diện (đầu, giữa, cuối video)

Usage:
    python scripts/visualize_skeleton.py
    python scripts/visualize_skeleton.py --video fall-01-cam0 --frames 0 50 100
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.filters import preprocess_frame

import yaml


# COCO-17 Skeleton connections (pairs of keypoint indices)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),      # nose → eyes
    (1, 3), (2, 4),      # eyes → ears
    (5, 6),               # shoulder → shoulder
    (5, 7), (7, 9),      # left arm
    (6, 8), (8, 10),     # right arm
    (5, 11), (6, 12),    # torso
    (11, 12),             # hip → hip
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

# Màu cho từng phần cơ thể (BGR)
COLORS = {
    "face": (255, 200, 50),      # Cyan nhạt - mặt
    "left_arm": (50, 255, 50),   # Xanh lá - tay trái
    "right_arm": (50, 50, 255),  # Đỏ - tay phải
    "torso": (255, 255, 50),     # Cyan - thân
    "left_leg": (50, 255, 200),  # Xanh ngọc - chân trái
    "right_leg": (200, 50, 255), # Tím - chân phải
}

# Map connection index → color group
CONNECTION_COLORS = [
    "face", "face",           # 0-1: nose-eyes
    "face", "face",           # 2-3: eyes-ears
    "torso",                  # 4: shoulder-shoulder
    "left_arm", "left_arm",   # 5-6: left arm
    "right_arm", "right_arm", # 7-8: right arm
    "torso", "torso",         # 9-10: torso verticals
    "torso",                  # 11: hip-hip
    "left_leg", "left_leg",   # 12-13: left leg
    "right_leg", "right_leg", # 14-15: right leg
]

KEYPOINT_NAMES = [
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
    "L_wrist", "R_wrist", "L_hip", "R_hip",
    "L_knee", "R_knee", "L_ankle", "R_ankle",
]


def draw_skeleton(frame: np.ndarray, keypoints: list, min_conf: float = 0.3) -> np.ndarray:
    """Vẽ skeleton lên frame.

    Args:
        frame: Ảnh BGR.
        keypoints: List 17 joints, mỗi joint = [x, y, confidence].
        min_conf: Ngưỡng confidence tối thiểu để vẽ.

    Returns:
        Frame đã vẽ skeleton.
    """
    overlay = frame.copy()
    kps = np.array(keypoints)

    # Vẽ connections (xương)
    for idx, (i, j) in enumerate(SKELETON_CONNECTIONS):
        if kps[i][2] >= min_conf and kps[j][2] >= min_conf:
            pt1 = (int(kps[i][0]), int(kps[i][1]))
            pt2 = (int(kps[j][0]), int(kps[j][1]))
            color = COLORS[CONNECTION_COLORS[idx]]
            cv2.line(overlay, pt1, pt2, color, 2, cv2.LINE_AA)

    # Vẽ keypoints (khớp)
    for idx, kp in enumerate(kps):
        if kp[2] >= min_conf:
            center = (int(kp[0]), int(kp[1]))
            cv2.circle(overlay, center, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(overlay, center, 4, (0, 0, 0), 1, cv2.LINE_AA)

    return overlay


def draw_bbox(frame: np.ndarray, bbox: list, conf: float, label: str) -> np.ndarray:
    """Vẽ bounding box lên frame."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    color = (0, 0, 255) if "fall" in label.lower() else (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} ({conf:.2f})"
    cv2.putText(frame, text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame


def create_comparison(original: np.ndarray, preprocessed: np.ndarray, skeleton: np.ndarray,
                      video_name: str, frame_idx: int, label: str) -> np.ndarray:
    """Tạo ảnh so sánh 3 panel: Original | Preprocessed | Skeleton.

    Args:
        original: Frame gốc.
        preprocessed: Frame sau Gaussian + CLAHE.
        skeleton: Frame có skeleton overlay.
        video_name: Tên video.
        frame_idx: Chỉ số frame.
        label: Nhãn (FALL/ADL).

    Returns:
        Ảnh ghép 3 panel.
    """
    h, w = original.shape[:2]

    # Resize nếu quá lớn
    max_w = 400
    if w > max_w:
        scale = max_w / w
        new_h, new_w = int(h * scale), max_w
        original = cv2.resize(original, (new_w, new_h))
        preprocessed = cv2.resize(preprocessed, (new_w, new_h))
        skeleton = cv2.resize(skeleton, (new_w, new_h))
        h, w = new_h, new_w

    # Header bar
    header_h = 40
    header = np.zeros((header_h, w * 3, 3), dtype=np.uint8)
    header[:] = (40, 40, 40)
    title = f"{video_name} | Frame #{frame_idx} | {label}"
    cv2.putText(header, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    # Labels cho mỗi panel
    labels_bar = np.zeros((30, w * 3, 3), dtype=np.uint8)
    labels_bar[:] = (30, 30, 30)
    panel_labels = ["Original", "Gaussian + CLAHE", "Skeleton Overlay"]
    for i, pl in enumerate(panel_labels):
        x_offset = i * w + w // 2 - len(pl) * 5
        cv2.putText(labels_bar, pl, (x_offset, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # Ghép 3 panel
    panels = np.hstack([original, preprocessed, skeleton])
    result = np.vstack([header, labels_bar, panels])

    return result


def visualize_video(video_name: str, frame_indices: list[int] | None, config: dict, output_dir: str):
    """Visualize skeleton cho một video cụ thể.

    Args:
        video_name: Tên video (không .mp4).
        frame_indices: Danh sách frame cần visualize. None = tự chọn đầu/giữa/cuối.
        config: Dict config.
        output_dir: Thư mục lưu ảnh output.
    """
    data_dir = os.path.join(config["paths"]["data_raw"], "ur_fall_detection")
    keypoint_dir = os.path.join(config["paths"]["data_processed"], "keypoints")

    video_path = os.path.join(data_dir, f"{video_name}.mp4")
    json_path = os.path.join(keypoint_dir, f"{video_name}.json")

    if not os.path.exists(video_path):
        print(f"  [ERR] Video không tồn tại: {video_path}")
        return
    if not os.path.exists(json_path):
        print(f"  [ERR] JSON không tồn tại: {json_path}")
        return

    # Load keypoints JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_frames = data["total_frames"]
    label = "FALL" if data["label"] == 1 else "ADL"

    # Tự chọn frame đại diện nếu không chỉ định
    if frame_indices is None:
        frame_indices = [
            0,                          # Đầu video
            total_frames // 4,          # 25%
            total_frames // 2,          # Giữa
            total_frames * 3 // 4,      # 75%
            total_frames - 1,           # Cuối
        ]

    # Mở video
    cap = cv2.VideoCapture(video_path)
    preprocess_config = config.get("preprocessing", {})
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  [*] {video_name} ({label}, {total_frames} frames)")
    print(f"  Visualize frames: {frame_indices}")

    for target_idx in frame_indices:
        if target_idx < 0 or target_idx >= total_frames:
            print(f"    [SKIP] Frame #{target_idx} ngoài phạm vi (0-{total_frames-1})")
            continue

        # Seek tới frame cần thiết
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, original = cap.read()
        if not ret:
            print(f"    [ERR] Không đọc được frame #{target_idx}")
            continue

        # Preprocessing
        preprocessed = preprocess_frame(original, preprocess_config)

        # Vẽ skeleton
        skeleton_frame = preprocessed.copy()
        frame_data = data["frames"][target_idx]
        persons = frame_data.get("persons", [])

        if persons:
            for person in persons:
                skeleton_frame = draw_bbox(skeleton_frame, person["bbox"], person["bbox_conf"], label)
                skeleton_frame = draw_skeleton(skeleton_frame, person["keypoints"])
            status = f"{len(persons)} person(s)"
        else:
            # Vẽ text "No person detected"
            h, w = skeleton_frame.shape[:2]
            cv2.putText(skeleton_frame, "No person detected", (w // 4, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            status = "no person"

        # Tạo ảnh so sánh
        comparison = create_comparison(original, preprocessed, skeleton_frame,
                                       video_name, target_idx, label)

        # Lưu
        out_path = os.path.join(output_dir, f"{video_name}_frame{target_idx:04d}.png")
        cv2.imwrite(out_path, comparison)
        print(f"    [OK] Frame #{target_idx:>4d} -> {status} -> {os.path.basename(out_path)}")

    cap.release()


def main():
    parser = argparse.ArgumentParser(description="Visualize skeleton keypoints trên video frames")
    parser.add_argument("--video", default=None, help="Tên video cụ thể (mặc định: tất cả 5 video mẫu)")
    parser.add_argument("--frames", nargs="+", type=int, default=None, help="Các frame cần visualize")
    parser.add_argument("--config", default="configs/default.yaml", help="Đường dẫn config")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = os.path.join(config["paths"]["outputs"], "skeleton_viz")

    print("=" * 60)
    print("  SKELETON VISUALIZATION - Verify Keypoints")
    print("=" * 60)

    if args.video:
        videos = [args.video]
    else:
        # Visualize tất cả 5 video mẫu đã extract
        keypoint_dir = os.path.join(config["paths"]["data_processed"], "keypoints")
        videos = sorted([
            os.path.splitext(f)[0]
            for f in os.listdir(keypoint_dir)
            if f.endswith(".json")
        ])

    for video_name in videos:
        visualize_video(video_name, args.frames, config, output_dir)

    print(f"\n  Output saved to: {os.path.abspath(output_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
