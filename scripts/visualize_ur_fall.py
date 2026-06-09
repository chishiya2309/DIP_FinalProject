"""
Script visualize skeleton trên dataset UR Fall Detection.
Render toàn bộ video và lưu thành file .mp4 để dễ theo dõi sự di chuyển.

Usage:
    python scripts/visualize_ur_fall.py --video fall-01-cam0
"""

import argparse
import json
import os
import sys
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.visualize_skeleton import draw_skeleton, draw_bbox

def main():
    parser = argparse.ArgumentParser(description="Visualize skeleton keypoints tren video UR Fall (xuat ra file .mp4)")
    parser.add_argument("--video", default="fall-01-cam0", help="Ten video (vd: fall-01-cam0)")
    parser.add_argument("--json_dir", default="data/processed/keypoints", help="Thu muc chua file json")
    parser.add_argument("--data_dir", default="data/raw/ur_fall_detection", help="Thu muc raw dataset UR Fall")
    parser.add_argument("--out_dir", default="outputs/skeleton_viz_ur_fall", help="Thu muc luu video output")
    args = parser.parse_args()

    video_name = args.video
    # Chấp nhận cả việc JSON có thể nằm trong thư mục con ur_fall_detection hoặc trực tiếp ở keypoints
    json_path = os.path.join(args.json_dir, f"{video_name}.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(args.json_dir, "ur_fall_detection", f"{video_name}.json")

    video_path = os.path.join(args.data_dir, f"{video_name}.mp4")

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
    # Ở UR Fall, label áp dụng cho toàn bộ video
    video_label_num = data.get("label", -1)
    video_label = "FALL" if video_label_num == 1 else ("NON-FALL" if video_label_num == 0 else "UNKNOWN")

    # Mo video doc
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
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

        # Tim data cho frame nay
        person_data = None
        if frame_idx < len(frames_data):
            fd = frames_data[frame_idx]
            if fd["frame_idx"] == frame_idx:
                person_data = fd["persons"]

        overlay = frame.copy()

        if person_data:
            for person in person_data:
                overlay = draw_bbox(overlay, person["bbox"], person["bbox_conf"], video_label)
                overlay = draw_skeleton(overlay, person["keypoints"])

        # Ve Label goc trai phia tren
        color = (0, 0, 255) if video_label == "FALL" else ((0, 255, 0) if video_label == "NON-FALL" else (255, 255, 255))
        cv2.putText(overlay, f"Frame: {frame_idx}/{total_frames} | Video Type: {video_label}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        out.write(overlay)
        frame_idx += 1

        if frame_idx % 50 == 0:
            print(f"  -> Da render {frame_idx}/{total_frames} frames...")

    cap.release()
    out.release()
    print(f"[OK] Render thanh cong: {out_path}")

if __name__ == "__main__":
    main()
