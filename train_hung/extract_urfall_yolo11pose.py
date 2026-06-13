"""
Extract COCO-17 keypoints from UR Fall Detection dataset using YOLO11-pose.

Output format is compatible with CustomFallDataset in train.py:
    Each sample = {
        "keypoint":       np.ndarray shape (M, T, 17, 2),  # M=num_persons, T=num_frames
        "keypoint_score": np.ndarray shape (M, T, 17),
        "label":          int (0=non-fall, 1=fall),
        "video_name":     str,
        "total_frames":   int,
        "fps":            float,
    }

Usage on Kaggle:
    1. Upload UR Fall Detection dataset as Kaggle dataset
    2. pip install ultralytics
    3. python extract_urfall_yolo11pose.py \
         --data-dir /kaggle/input/ur-fall-detection \
         --falls-csv /kaggle/input/ur-fall-detection/urfall-cam0-falls.csv \
         --adls-csv /kaggle/input/ur-fall-detection/urfall-cam0-adls.csv \
         --output-dir /kaggle/working/urfall_pose \
         --model yolo11s-pose.pt \
         --val-ratio 0.2
"""
from __future__ import annotations

import argparse
import csv
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract COCO-17 keypoints from UR Fall Detection using YOLO11-pose"
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw/ur_fall_detection",
        help="Directory containing .mp4 video files",
    )
    parser.add_argument(
        "--falls-csv",
        default="data/raw/ur_fall_detection/urfall-cam0-falls.csv",
        help="Path to urfall-cam0-falls.csv",
    )
    parser.add_argument(
        "--adls-csv",
        default="data/raw/ur_fall_detection/urfall-cam0-adls.csv",
        help="Path to urfall-cam0-adls.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/urfall_pose",
        help="Output directory for .pkl files",
    )
    parser.add_argument(
        "--model",
        default="yolo11s-pose.pt",
        help="YOLO pose model weights (e.g. yolo11s-pose.pt, yolo11m-pose.pt)",
    )
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, 0, cuda:0")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO input image size")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    parser.add_argument(
        "--cam-filter",
        default="cam0",
        help="Camera filter for video files (default: cam0 only)",
    )
    return parser.parse_args()


def resolve_device(device: str) -> str:
    """Resolve device string for YOLO."""
    if device != "auto":
        return device

    import torch

    if torch.cuda.is_available():
        return "0"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def parse_label_csv(csv_path: str | Path) -> dict[str, dict[int, int]]:
    """Parse UR Fall label CSV into {video_name: {frame_num: label}}.

    CSV format (no header):
        sequence_name, frame_number, label, feature1, feature2, ...

    Labels: -1 = not lying, 0 = transitioning (falling), 1 = lying on ground.
    """
    labels: dict[str, dict[int, int]] = defaultdict(dict)
    path = Path(csv_path)

    if not path.is_file():
        print(f"  [WARN] Label CSV not found: {csv_path}")
        return labels

    with path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            seq_name = row[0].strip()
            frame_num = int(row[1].strip())
            label = int(row[2].strip())
            labels[seq_name][frame_num] = label

    return labels


def determine_video_label(
    video_name: str,
    frame_labels: dict[str, dict[int, int]],
) -> int:
    """Determine binary label for entire video.

    A video is labeled as fall (1) if it has any frame with label=1 (lying).
    Otherwise non-fall (0).

    The CSV uses sequence names like 'fall-01', 'adl-01' while video files
    are named 'fall-01-cam0.mp4', 'adl-01-cam0.mp4'.
    """
    # Extract sequence name: "fall-01-cam0" -> "fall-01"
    parts = video_name.split("-")
    if len(parts) >= 3:
        seq_name = f"{parts[0]}-{parts[1]}"
    else:
        seq_name = video_name

    if seq_name in frame_labels:
        has_lying = any(label == 1 for label in frame_labels[seq_name].values())
        if has_lying:
            return 1
        return 0

    # Fallback: infer from video name
    if video_name.startswith("fall"):
        return 1
    return 0


def get_per_frame_labels(
    video_name: str,
    total_frames: int,
    frame_labels: dict[str, dict[int, int]],
) -> list[int]:
    """Get per-frame labels for a video. Returns list of length total_frames.

    Labels: -1 = not lying, 0 = transitioning, 1 = lying.
    Default to -1 if not found in CSV.
    """
    parts = video_name.split("-")
    if len(parts) >= 3:
        seq_name = f"{parts[0]}-{parts[1]}"
    else:
        seq_name = video_name

    per_frame = []
    video_frame_labels = frame_labels.get(seq_name, {})

    for frame_idx in range(total_frames):
        # CSV uses 1-based frame numbering
        csv_frame_num = frame_idx + 1
        label = video_frame_labels.get(csv_frame_num, -1)
        per_frame.append(label)

    return per_frame


def select_person_from_result(result: Any) -> tuple[np.ndarray, np.ndarray]:
    """Select the most prominent person from a YOLO pose result.

    Returns:
        xy: shape (17, 2) - keypoint coordinates
        confidence: shape (17,) - keypoint confidence scores
    """
    keypoints = getattr(result, "keypoints", None)
    boxes = getattr(result, "boxes", None)

    if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
        return np.zeros((17, 2), dtype=np.float32), np.zeros(17, dtype=np.float32)

    xy = keypoints.xy.cpu().numpy().astype(np.float32)

    if keypoints.conf is None:
        conf = np.ones(xy.shape[:2], dtype=np.float32)
    else:
        conf = keypoints.conf.cpu().numpy().astype(np.float32)

    # Score each person: mean keypoint confidence * sqrt(box area) * box confidence
    scores = conf.mean(axis=1)

    if boxes is not None and boxes.xyxy is not None and len(boxes.xyxy) == xy.shape[0]:
        box_array = boxes.xyxy.cpu().numpy().astype(np.float32)
        widths = np.clip(box_array[:, 2] - box_array[:, 0], 0.0, None)
        heights = np.clip(box_array[:, 3] - box_array[:, 1], 0.0, None)
        areas = widths * heights

        box_conf = np.ones(xy.shape[0], dtype=np.float32)
        if boxes.conf is not None:
            box_conf = boxes.conf.cpu().numpy().astype(np.float32)

        scores = scores * np.sqrt(areas + 1.0) * box_conf

    person_idx = int(np.argmax(scores))
    return xy[person_idx], conf[person_idx]


def extract_video_keypoints(
    video_path: str | Path,
    model: Any,
    imgsz: int = 640,
    conf: float = 0.25,
    device: str = "0",
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Extract keypoints from all frames of a video.

    Returns:
        all_xy: shape (T, 17, 2)
        all_scores: shape (T, 17)
        fps: float
        total_frames: int
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_xy = []
    all_scores = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO pose on single frame
        results = model.predict(
            source=frame,
            imgsz=imgsz,
            conf=conf,
            device=device,
            verbose=False,
        )

        if results and len(results) > 0:
            xy, score = select_person_from_result(results[0])
        else:
            xy = np.zeros((17, 2), dtype=np.float32)
            score = np.zeros(17, dtype=np.float32)

        all_xy.append(xy)
        all_scores.append(score)
        frame_idx += 1

    cap.release()

    if not all_xy:
        return (
            np.zeros((0, 17, 2), dtype=np.float32),
            np.zeros((0, 17), dtype=np.float32),
            fps,
            0,
        )

    return (
        np.stack(all_xy).astype(np.float32),
        np.stack(all_scores).astype(np.float32),
        fps,
        frame_idx,
    )


def build_samples(
    data_dir: Path,
    model: Any,
    frame_labels: dict[str, dict[int, int]],
    cam_filter: str,
    imgsz: int,
    conf: float,
    device: str,
) -> list[dict[str, Any]]:
    """Process all videos and build sample list."""
    video_files = sorted(data_dir.glob("*.mp4"))

    if cam_filter:
        video_files = [v for v in video_files if cam_filter in v.stem]

    if not video_files:
        raise RuntimeError(f"No .mp4 files found in {data_dir}")

    print(f"Found {len(video_files)} videos (filter: {cam_filter})")

    samples = []
    stats = {"fall": 0, "non_fall": 0, "total_frames": 0, "detected_frames": 0}

    for video_path in tqdm(video_files, desc="Extracting keypoints"):
        video_name = video_path.stem

        try:
            all_xy, all_scores, fps, total_frames = extract_video_keypoints(
                video_path, model, imgsz=imgsz, conf=conf, device=device
            )
        except Exception as e:
            print(f"  [ERR] {video_name}: {e}")
            continue

        if total_frames == 0:
            print(f"  [WARN] {video_name}: 0 frames extracted")
            continue

        # Determine video-level label
        label = determine_video_label(video_name, frame_labels)

        # Get per-frame labels for reference
        per_frame = get_per_frame_labels(video_name, total_frames, frame_labels)

        # Count frames with at least one detected keypoint
        detected_mask = all_scores.max(axis=1) > 0.1
        detected_count = int(detected_mask.sum())

        # Build sample in CustomFallDataset format:
        # keypoint: (M, T, 17, 2) where M=1 (single person)
        # keypoint_score: (M, T, 17)
        sample = {
            "keypoint": all_xy[np.newaxis, ...],        # (1, T, 17, 2)
            "keypoint_score": all_scores[np.newaxis, ...],  # (1, T, 17)
            "label": label,
            "video_name": video_name,
            "total_frames": total_frames,
            "fps": float(fps),
            "per_frame_labels": per_frame,
        }
        samples.append(sample)

        if label == 1:
            stats["fall"] += 1
        else:
            stats["non_fall"] += 1
        stats["total_frames"] += total_frames
        stats["detected_frames"] += detected_count

    print(f"\n{'='*60}")
    print(f"Extraction Summary:")
    print(f"  Videos:  {len(samples)} ({stats['fall']} fall, {stats['non_fall']} non-fall)")
    print(f"  Frames:  {stats['total_frames']} total, {stats['detected_frames']} with person detected")
    if stats["total_frames"] > 0:
        rate = stats["detected_frames"] / stats["total_frames"] * 100
        print(f"  Detect:  {rate:.1f}%")
    print(f"{'='*60}\n")

    return samples


def split_train_val(
    samples: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split samples into train/val, stratified by label."""
    rng = random.Random(seed)

    falls = [s for s in samples if s["label"] == 1]
    non_falls = [s for s in samples if s["label"] == 0]

    rng.shuffle(falls)
    rng.shuffle(non_falls)

    n_val_falls = max(1, int(len(falls) * val_ratio))
    n_val_non_falls = max(1, int(len(non_falls) * val_ratio))

    val_samples = falls[:n_val_falls] + non_falls[:n_val_non_falls]
    train_samples = falls[n_val_falls:] + non_falls[n_val_non_falls:]

    rng.shuffle(val_samples)
    rng.shuffle(train_samples)

    return train_samples, val_samples


def save_pkl(samples: list[dict[str, Any]], output_path: Path) -> None:
    """Save samples as pickle file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(samples, f)
    print(f"Saved {len(samples)} samples to {output_path}")


def main() -> None:
    args = parse_args()

    # Parse per-frame labels from CSV
    print("[1/4] Parsing label CSVs...")
    falls_labels = parse_label_csv(args.falls_csv)
    adls_labels = parse_label_csv(args.adls_csv)

    # Merge labels
    frame_labels: dict[str, dict[int, int]] = {}
    frame_labels.update(falls_labels)
    frame_labels.update(adls_labels)
    print(f"  Falls sequences: {len(falls_labels)}")
    print(f"  ADL sequences:   {len(adls_labels)}")

    # Load YOLO model
    print(f"\n[2/4] Loading YOLO model: {args.model}...")
    from ultralytics import YOLO

    device = resolve_device(args.device)
    print(f"  Device: {device}")
    model = YOLO(args.model)

    # Extract keypoints from all videos
    print(f"\n[3/4] Extracting keypoints from videos...")
    data_dir = Path(args.data_dir)
    samples = build_samples(
        data_dir=data_dir,
        model=model,
        frame_labels=frame_labels,
        cam_filter=args.cam_filter,
        imgsz=args.imgsz,
        conf=args.conf,
        device=device,
    )

    if not samples:
        print("[ERR] No samples extracted!")
        return

    # Split train/val
    print(f"[4/4] Splitting train/val (ratio={args.val_ratio})...")
    train_samples, val_samples = split_train_val(samples, args.val_ratio, args.seed)

    train_falls = sum(1 for s in train_samples if s["label"] == 1)
    train_non = len(train_samples) - train_falls
    val_falls = sum(1 for s in val_samples if s["label"] == 1)
    val_non = len(val_samples) - val_falls

    print(f"  Train: {len(train_samples)} ({train_falls} fall, {train_non} non-fall)")
    print(f"  Val:   {len(val_samples)} ({val_falls} fall, {val_non} non-fall)")

    # Save
    output_dir = Path(args.output_dir)
    save_pkl(train_samples, output_dir / "train_data.pkl")
    save_pkl(val_samples, output_dir / "val_data.pkl")
    save_pkl(samples, output_dir / "all_data.pkl")

    print(f"\nDone! Output saved to: {output_dir}")
    print(f"\nTo train, run:")
    print(f"  python train.py \\")
    print(f"    --dataset-source custom \\")
    print(f"    --custom-train-pkl {output_dir / 'train_data.pkl'} \\")
    print(f"    --custom-val-pkl {output_dir / 'val_data.pkl'} \\")
    print(f"    --sequence-length 72 \\")
    print(f"    --epochs 50")


if __name__ == "__main__":
    main()
