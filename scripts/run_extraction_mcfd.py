"""
Script trích xuất keypoints cho Multiple Cameras Fall Dataset (MCFD).

Cấu trúc MCFD:
  - 24 kịch bản (chute01..chute24), mỗi kịch bản 8 camera (cam1..cam8)
  - Video: data/raw/multiple_cameras_fall/dataset/dataset/chute{NN}/cam{N}.avi
  - Labels: data/raw/multiple_cameras_fall/data_tuple3.csv
    Columns: chute, cam, start, end, label (1=fall, 0=non-fall)
    Labels áp dụng cho frame-span, không phải toàn bộ video

Khác biệt so với UR Fall:
  - Video dài hơn (hàng ngàn frames), chỉ một vài đoạn ngắn (~30 frames) chứa fall
  - Mỗi video extract toàn bộ keypoints, labels được lưu riêng trong JSON
    để sau này Sliding Window sẽ dùng labels cắt đúng đoạn fall/non-fall

Usage:
    python scripts/run_extraction_mcfd.py                  # Chạy 3 chute mẫu (chute01-03)
    python scripts/run_extraction_mcfd.py --all             # Toàn bộ 24 chute × 8 cam
    python scripts/run_extraction_mcfd.py --chutes 1 5 10   # Chute cụ thể
    python scripts/run_extraction_mcfd.py --cam 1            # Chỉ camera 1
"""

import argparse
import csv
import json
import os
import sys
import time

import cv2
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.filters import preprocess_frame
from src.pose_estimation.keypoint_extractor import KeypointExtractor


def load_config(config_path: str = "configs/default.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_mcfd_labels(csv_path: str) -> dict:
    """Load và tổ chức labels theo (chute, cam).

    Returns:
        Dict với key = (chute_num, cam_num), value = list of segments:
        {
            (1, 1): [
                {"start": 1052, "end": 1082, "label": 0},
                {"start": 1083, "end": 1113, "label": 1},
                ...
            ],
            ...
        }
    """
    labels = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chute = int(float(row["chute"]))
            cam = int(float(row["cam"]))

            # Bỏ qua dòng lỗi trong CSV (chute23, cam55)
            if cam > 8:
                continue

            key = (chute, cam)
            if key not in labels:
                labels[key] = []

            labels[key].append({
                "start": int(float(row["start"])),
                "end": int(float(row["end"])),
                "label": int(float(row["label"])),
            })

    return labels


def get_video_list(dataset_dir: str, chutes: list[int] | None, cam_filter: int | None, run_all: bool) -> list[dict]:
    """Xác định danh sách video cần xử lý.

    Returns:
        List các dict: {"path": ..., "chute": N, "cam": N}
    """
    if run_all:
        chute_nums = list(range(1, 25))
    elif chutes:
        chute_nums = chutes
    else:
        # Mặc định: 3 chute mẫu
        chute_nums = [1, 2, 3]

    cam_nums = [cam_filter] if cam_filter else list(range(1, 9))

    videos = []
    for chute in chute_nums:
        chute_dir = os.path.join(dataset_dir, f"chute{chute:02d}")
        if not os.path.isdir(chute_dir):
            print(f"  [WARN] Thu muc khong ton tai: {chute_dir}")
            continue

        for cam in cam_nums:
            video_path = os.path.join(chute_dir, f"cam{cam}.avi")
            if os.path.exists(video_path):
                videos.append({"path": video_path, "chute": chute, "cam": cam})
            else:
                print(f"  [WARN] Video khong ton tai: cam{cam}.avi trong chute{chute:02d}")

    return videos


def process_mcfd_video(video_info: dict, extractor: KeypointExtractor,
                       preprocess_config: dict, labels: dict) -> dict:
    """Xử lý một video MCFD qua pipeline.

    Args:
        video_info: Dict {"path", "chute", "cam"}.
        extractor: KeypointExtractor instance.
        preprocess_config: Preprocessing config.
        labels: Dict labels từ CSV.

    Returns:
        Dict kết quả trích xuất với segment annotations.
    """
    video_path = video_info["path"]
    chute = video_info["chute"]
    cam = video_info["cam"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERR] Khong mo duoc video: {video_path}")
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Lấy label segments cho video này
    key = (chute, cam)
    segments = labels.get(key, [])

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

    video_name = f"chute{chute:02d}-cam{cam}"

    return {
        "video_name": video_name,
        "source": "multiple_cameras_fall",
        "chute": chute,
        "cam": cam,
        "img_shape": [height, width],
        "total_frames": frame_idx,
        "fps": round(fps, 2),
        "frames_with_person": detected_count,
        "segments": segments,
        "frames": frames_data,
    }


def save_result(result: dict, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{result['video_name']}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Pipeline trich xuat keypoints - MCFD")
    parser.add_argument("--all", action="store_true", help="Xu ly toan bo 24 chute x 8 cam")
    parser.add_argument("--chutes", nargs="+", type=int, help="Danh sach chute cu the (vd: 1 5 10)")
    parser.add_argument("--cam", type=int, default=None, help="Chi xu ly 1 camera cu the (1-8)")
    parser.add_argument("--config", default="configs/default.yaml", help="Duong dan config")
    parser.add_argument("--device", default=None, help="Device: cpu hoac cuda")
    args = parser.parse_args()

    config = load_config(args.config)
    device = args.device or config.get("system", {}).get("device", "cpu")

    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print("[INFO] CUDA khong kha dung, chuyen sang CPU")
                device = "cpu"
        except ImportError:
            print("[INFO] PyTorch chua cai dat, chuyen sang CPU")
            device = "cpu"

    mcfd_dir = os.path.join(config["paths"]["data_raw"], "multiple_cameras_fall")
    dataset_dir = os.path.join(mcfd_dir, "dataset", "dataset")
    csv_path = os.path.join(mcfd_dir, "data_tuple3.csv")
    output_dir = os.path.join(config["paths"]["data_processed"], "keypoints_mcfd")

    print("=" * 60)
    print("  PIPELINE TRICH XUAT KEYPOINTS - MCFD")
    print("=" * 60)
    print(f"  Device       : {device}")
    print(f"  Dataset dir  : {dataset_dir}")
    print(f"  CSV labels   : {csv_path}")
    print(f"  Output dir   : {output_dir}")
    print()

    # Load labels
    print("[1/3] Dang load labels tu CSV...")
    labels = load_mcfd_labels(csv_path)
    total_segments = sum(len(v) for v in labels.values())
    print(f"  -> {len(labels)} video co label, {total_segments} segments tong cong")
    print()

    # Load model
    pose_cfg = config.get("pose_estimation", {})
    print(f"[2/3] Dang load model {pose_cfg.get('model', 'yolo11s-pose.pt')}...")
    extractor = KeypointExtractor(
        model_name=pose_cfg.get("model", "yolo11s-pose.pt"),
        conf_threshold=pose_cfg.get("confidence_threshold", 0.5),
        device=device,
    )
    print("  -> Model loaded thanh cong!")
    print()

    # Xác định video list
    videos = get_video_list(dataset_dir, args.chutes, args.cam, args.all)
    if not videos:
        print("[ERR] Khong tim thay video nao!")
        return

    cam_info = f"cam{args.cam}" if args.cam else "all cams"
    if args.all:
        mode_str = f"TOAN BO 24 chute x {cam_info}"
    elif args.chutes:
        mode_str = f"chute {args.chutes} x {cam_info}"
    else:
        mode_str = f"3 chute mau x {cam_info}"

    print(f"[3/3] Dang xu ly {mode_str} ({len(videos)} video)...")
    print("-" * 60)

    preprocess_config = config.get("preprocessing", {})
    total_start = time.time()
    summary = []

    for i, video_info in enumerate(videos, 1):
        video_name = f"chute{video_info['chute']:02d}-cam{video_info['cam']}"
        print(f"  [{i}/{len(videos)}] {video_name}...", end=" ", flush=True)

        start = time.time()
        result = process_mcfd_video(video_info, extractor, preprocess_config, labels)

        if not result:
            print("FAILED")
            continue

        output_path = save_result(result, output_dir)
        elapsed = time.time() - start

        detect_rate = result["frames_with_person"] / max(result["total_frames"], 1) * 100
        n_fall = sum(1 for s in result["segments"] if s["label"] == 1)
        n_nonfal = sum(1 for s in result["segments"] if s["label"] == 0)

        print(f"OK ({result['total_frames']} fr, {detect_rate:.0f}% det, {n_fall}F/{n_nonfal}NF, {elapsed:.1f}s)")

        summary.append({
            "video": video_name,
            "frames": result["total_frames"],
            "detected": result["frames_with_person"],
            "detect_rate": f"{detect_rate:.1f}%",
            "fall_seg": n_fall,
            "nf_seg": n_nonfal,
            "time": f"{elapsed:.1f}s",
        })

    total_elapsed = time.time() - total_start

    # Báo cáo
    print()
    print("=" * 60)
    print("BAO CAO TOM TAT - MCFD")
    print("=" * 60)
    print(f"  Tong video     : {len(summary)}")
    print(f"  Tong thoi gian : {total_elapsed:.1f}s ({total_elapsed/60:.1f} phut)")
    print()
    print(f"  {'Video':<18} {'Frames':>7} {'Detect':>7} {'Rate':>6} {'Fall':>5} {'NF':>4} {'Time':>7}")
    print(f"  {'-'*18} {'-'*7} {'-'*7} {'-'*6} {'-'*5} {'-'*4} {'-'*7}")
    for s in summary:
        print(f"  {s['video']:<18} {s['frames']:>7} {s['detected']:>7} {s['detect_rate']:>6} {s['fall_seg']:>5} {s['nf_seg']:>4} {s['time']:>7}")

    print(f"\n  Output saved to: {os.path.abspath(output_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
