"""
Script visualize skeleton trên dataset Multiple Cameras Fall Dataset (MCFD)
Render toàn bộ video và lưu thành file .mp4 để bạn dễ dàng xem lại toàn bộ quá trình di chuyển.

Usage:
    python scripts/visualize_mcfd.py --chute 1 --cam 1
"""

import argparse
import json
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.visualize_skeleton import draw_skeleton, draw_bbox

def get_frame_label(frame_idx: int, segments: list) -> str:
    """Xác định xem frame hiện tại có nằm trong đoạn Fall hay Non-Fall không."""
    for seg in segments:
        if seg["start"] <= frame_idx <= seg["end"]:
            return "FALL" if seg["label"] == 1 else "NON-FALL"
    return "NORMAL"

def main():
    parser = argparse.ArgumentParser(description="Visualize skeleton keypoints tren video MCFD")
    parser.add_argument("--chute", type=int, default=1, help="Chute number (vd: 1)")
    parser.add_argument("--cam", type=int, default=1, help="Camera number (vd: 1)")
    parser.add_argument("--json_dir", default="data/processed/keypoints/multiple_cameras_fall", help="Thu muc chua file json")
    parser.add_argument("--data_dir", default="data/raw/multiple_cameras_fall/dataset/dataset", help="Thu muc raw dataset MCFD")
    parser.add_argument("--out_dir", default="outputs/skeleton_viz_mcfd", help="Thu muc luu video output")
    args = parser.parse_args()

    video_name = f"chute{args.chute:02d}-cam{args.cam}"
    json_path = os.path.join(args.json_dir, f"{video_name}.json")
    video_path = os.path.join(args.data_dir, f"chute{args.chute:02d}", f"cam{args.cam}.avi")

    if not os.path.exists(json_path):
        print(f"[Loi] Khong tim thay file JSON: {json_path}")
        return
    
    if not os.path.exists(video_path):
        print(f"[Loi] Khong tim thay video gốc: {video_path}")
        return

    # Load JSON
    print(f"[*] Dang load du lieu tu {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames_data = data["frames"]
    segments = data.get("segments", [])

    # Mo video doc
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Chuan bi video writer
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{video_name}_skeleton.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print(f"[*] Dang render video output vao {out_path}...")
    print(f"[*] Tong so frames: {total_frames}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Tinh trang frame
        label_text = get_frame_label(frame_idx, segments)

        # Tim data cho frame nay
        person_data = None
        if frame_idx < len(frames_data):
            # Dam bao map dung frame index
            fd = frames_data[frame_idx]
            if fd["frame_idx"] == frame_idx:
                person_data = fd["persons"]

        overlay = frame.copy()

        if person_data:
            for person in person_data:
                overlay = draw_bbox(overlay, person["bbox"], person["bbox_conf"], label_text)
                overlay = draw_skeleton(overlay, person["keypoints"])

        # Ve Label cua Segments goc trai phia tren
        color = (0, 0, 255) if label_text == "FALL" else ((0, 255, 0) if label_text == "NON-FALL" else (255, 255, 255))
        cv2.putText(overlay, f"Frame: {frame_idx}/{total_frames} | State: {label_text}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        out.write(overlay)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  -> Da render {frame_idx}/{total_frames} frames...")

    cap.release()
    out.release()
    print(f"[OK] Render thanh cong: {out_path}")

if __name__ == "__main__":
    main()
