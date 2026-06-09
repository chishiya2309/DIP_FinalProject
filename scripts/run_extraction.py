"""
Script chạy Pipeline trích xuất keypoints end-to-end.

Luồng xử lý: Video → Gaussian Blur → CLAHE → YOLOv8-Pose → JSON output

Mặc định chạy trên 5 video đại diện để verify pipeline:
  - fall-01-cam0, fall-01-cam1, fall-10-cam0 (fall samples)
  - adl-01-cam0 (sáng), adl-10-cam0 (tối) (ADL samples)

Usage:
    python scripts/run_extraction.py                    # 5 video mẫu
    python scripts/run_extraction.py --all               # Toàn bộ UR Fall
    python scripts/run_extraction.py --videos fall-01-cam0 fall-02-cam0
"""

import argparse
import json
import os
import sys
import time

import cv2
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.filters import preprocess_frame
from src.pose_estimation.keypoint_extractor import KeypointExtractor


# 5 video đại diện cho verify pipeline
DEFAULT_SAMPLE_VIDEOS = [
    "fall-01-cam0",
    "fall-01-cam1",
    "fall-10-cam0",
    "adl-01-cam0",
    "adl-10-cam0",
]


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load cấu hình từ file YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_video_list(data_dir: str, video_names: list[str] | None = None, run_all: bool = False) -> list[str]:
    """Xác định danh sách video cần xử lý.

    Args:
        data_dir: Thư mục chứa video (.mp4).
        video_names: Danh sách tên video cụ thể (không cần .mp4).
        run_all: Nếu True, xử lý toàn bộ video trong thư mục.

    Returns:
        List đường dẫn đầy đủ tới các file video.
    """
    if run_all:
        videos = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".mp4")
        ])
    elif video_names:
        videos = []
        for name in video_names:
            path = os.path.join(data_dir, f"{name}.mp4")
            if os.path.exists(path):
                videos.append(path)
            else:
                print(f"  [WARN] Không tìm thấy: {path}")
    else:
        videos = []
        for name in DEFAULT_SAMPLE_VIDEOS:
            path = os.path.join(data_dir, f"{name}.mp4")
            if os.path.exists(path):
                videos.append(path)
            else:
                print(f"  [WARN] Không tìm thấy video mẫu: {name}.mp4")

    return videos


def process_video(video_path: str, extractor: KeypointExtractor, preprocess_config: dict) -> dict:
    """Xử lý một video qua toàn bộ pipeline.

    Args:
        video_path: Đường dẫn tới file video.
        extractor: Instance của KeypointExtractor.
        preprocess_config: Dict chứa tham số preprocessing.

    Returns:
        Dict chứa toàn bộ kết quả trích xuất, tương thích với
        format chuyển đổi sang MMAction2 .pkl sau này.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"  [ERR] Không mở được video: {video_path}")
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_data = []
    frame_idx = 0
    detected_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = preprocess_frame(frame, preprocess_config)
        persons = extractor.extract(processed_frame)

        if persons:
            detected_count += 1

        frames_data.append({
            "frame_idx": frame_idx,
            "persons": persons,
        })

        frame_idx += 1

    cap.release()

    # Xác định label và source từ tên video
    if video_name.startswith("fall"):
        label = 1
    elif video_name.startswith("adl"):
        label = 0
    else:
        label = -1

    source = "ur_fall_detection"

    return {
        "video_name": video_name,
        "source": source,
        "label": label,
        "img_shape": [height, width],
        "total_frames": frame_idx,
        "fps": round(fps, 2),
        "frames_with_person": detected_count,
        "frames": frames_data,
    }


def save_result(result: dict, output_dir: str) -> str:
    """Lưu kết quả trích xuất ra file JSON.

    Args:
        result: Dict kết quả từ process_video().
        output_dir: Thư mục output.

    Returns:
        Đường dẫn file JSON đã lưu.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{result['video_name']}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Pipeline trích xuất keypoints từ video")
    parser.add_argument("--all", action="store_true", help="Xử lý toàn bộ video UR Fall")
    parser.add_argument("--videos", nargs="+", help="Danh sách tên video cụ thể (không cần .mp4)")
    parser.add_argument("--config", default="configs/default.yaml", help="Đường dẫn file config")
    parser.add_argument("--device", default=None, help="Device: cpu hoặc cuda (mặc định đọc từ config)")
    args = parser.parse_args()

    config = load_config(args.config)
    device = args.device or config.get("system", {}).get("device", "cpu")

    # Fallback sang CPU nếu CUDA không khả dụng
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print("[INFO] CUDA không khả dụng, chuyển sang CPU")
                device = "cpu"
        except ImportError:
            print("[INFO] PyTorch chưa cài đặt, chuyển sang CPU")
            device = "cpu"

    data_dir = os.path.join(config["paths"]["data_raw"], "ur_fall_detection")
    output_dir = os.path.join(config["paths"]["data_processed"], "keypoints")

    print("=" * 60)
    print("  PIPELINE TRÍCH XUẤT KEYPOINTS - Tuần 3-4")
    print("=" * 60)
    print(f"  Device       : {device}")
    print(f"  Data dir     : {data_dir}")
    print(f"  Output dir   : {output_dir}")
    print()

    # Load model YOLOv8-Pose
    pose_cfg = config.get("pose_estimation", {})
    print(f"[1/3] Đang load model {pose_cfg.get('model', 'yolov8n-pose.pt')}...")
    extractor = KeypointExtractor(
        model_name=pose_cfg.get("model", "yolov8n-pose.pt"),
        conf_threshold=pose_cfg.get("confidence_threshold", 0.5),
        device=device,
    )
    print("  → Model loaded thành công!\n")

    # Xác định danh sách video
    videos = get_video_list(data_dir, args.videos, args.all)
    if not videos:
        print("[ERR] Không tìm thấy video nào để xử lý!")
        return

    mode = "TOÀN BỘ" if args.all else f"{len(videos)} video"
    print(f"[2/3] Đang xử lý {mode}...")
    print("-" * 60)

    preprocess_config = config.get("preprocessing", {})
    total_start = time.time()
    summary = []

    for i, video_path in enumerate(videos, 1):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"  [{i}/{len(videos)}] {video_name}...", end=" ", flush=True)

        start = time.time()
        result = process_video(video_path, extractor, preprocess_config)

        if not result:
            print("FAILED")
            continue

        output_path = save_result(result, output_dir)
        elapsed = time.time() - start

        label_str = "FALL" if result["label"] == 1 else "ADL"
        detect_rate = result["frames_with_person"] / max(result["total_frames"], 1) * 100

        print(f"OK ({result['total_frames']} frames, {detect_rate:.0f}% detected, {elapsed:.1f}s)")

        summary.append({
            "video": video_name,
            "label": label_str,
            "frames": result["total_frames"],
            "detected": result["frames_with_person"],
            "detect_rate": f"{detect_rate:.1f}%",
            "time": f"{elapsed:.1f}s",
            "output": output_path,
        })

    total_elapsed = time.time() - total_start

    # Báo cáo tóm tắt
    print()
    print("=" * 60)
    print(f"[3/3] BÁO CÁO TÓM TẮT")
    print("=" * 60)
    print(f"  Tổng video    : {len(summary)}")
    print(f"  Tổng thời gian: {total_elapsed:.1f}s")
    print()
    print(f"  {'Video':<20} {'Label':>5} {'Frames':>7} {'Detect':>8} {'Rate':>6} {'Time':>6}")
    print(f"  {'-'*20} {'-'*5} {'-'*7} {'-'*8} {'-'*6} {'-'*6}")
    for s in summary:
        print(f"  {s['video']:<20} {s['label']:>5} {s['frames']:>7} {s['detected']:>8} {s['detect_rate']:>6} {s['time']:>6}")

    print(f"\n  Output saved to: {os.path.abspath(output_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
