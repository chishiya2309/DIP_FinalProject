# ============================================================================
# Kaggle Notebook: Extract COCO-17 Keypoints from UR Fall Detection
# Using YOLO11-pose → Output .pkl for CustomFallDataset (train.py)
# ============================================================================
# Strategy:
#   1. Extract keypoints from ALL frames of each video using YOLO11-pose
#   2. Use per-frame CSV labels to identify contiguous segments:
#      - label -1 = not lying (non-fall)
#      - label  0 = transitioning (SKIP — ambiguous)
#      - label  1 = lying on ground (fall)
#   3. Apply sliding window with different strides for fall vs non-fall
#   4. Output .pkl compatible with CustomFallDataset in train.py
# ============================================================================
# Setup:
#   1. Upload UR Fall Detection dataset as Kaggle dataset
#      (all .mp4 + urfall-cam0-falls.csv + urfall-cam0-adls.csv)
#   2. Enable GPU accelerator (Settings → Accelerator → GPU T4 x2 or P100)
#   3. Run all cells
# ============================================================================

# %% [markdown]
# ## 1. Install Dependencies

# %%
# !pip install -q ultralytics

# %% [markdown]
# ## 2. Configuration

# %%
import csv
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm.auto import tqdm

# ---- CONFIG ----
DATA_DIR = Path("/kaggle/input/ur-fall-detection")
FALLS_CSV = DATA_DIR / "urfall-cam0-falls.csv"
ADLS_CSV = DATA_DIR / "urfall-cam0-adls.csv"
OUTPUT_DIR = Path("/kaggle/working/urfall_pose")
MODEL_NAME = "yolo11s-pose.pt"
IMGSZ = 640
CONF = 0.25
VAL_RATIO = 0.2
SEED = 42
CAM_FILTER = "cam0"

# Sliding window augmentation settings
WINDOW_SIZE = 30          # frames per window
STRIDE_FALL = 10          # stride for fall (lying) segments → more overlap = more samples
STRIDE_NONFALL = 30       # stride for non-fall segments → less overlap
MIN_WINDOW_FRAMES = 15    # minimum frames to form a valid window

# %% [markdown]
# ## 3. Parse Per-Frame Labels from CSV

# %%
def parse_label_csv(csv_path: Path) -> dict[str, dict[int, int]]:
    """Parse UR Fall label CSV → {sequence_name: {frame_num: label}}.

    Labels: -1 = not lying, 0 = transitioning, 1 = lying.
    """
    labels: dict[str, dict[int, int]] = defaultdict(dict)
    if not csv_path.is_file():
        print(f"  [WARN] CSV not found: {csv_path}")
        return labels
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            seq_name = row[0].strip()
            frame_num = int(row[1].strip())
            label = int(row[2].strip())
            labels[seq_name][frame_num] = label
    return labels

falls_labels = parse_label_csv(FALLS_CSV)
adls_labels = parse_label_csv(ADLS_CSV)

frame_labels: dict[str, dict[int, int]] = {}
frame_labels.update(falls_labels)
frame_labels.update(adls_labels)

print(f"Falls sequences: {len(falls_labels)}")
print(f"ADL sequences:   {len(adls_labels)}")
print(f"Total sequences: {len(frame_labels)}")

# %% [markdown]
# ## 4. Load YOLO11-pose Model

# %%
from ultralytics import YOLO
import torch

device = "0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device} (CUDA available: {torch.cuda.is_available()})")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

model = YOLO(MODEL_NAME)
print(f"Model loaded: {MODEL_NAME}")

# %% [markdown]
# ## 5. Helper Functions

# %%
def select_person(result) -> tuple[np.ndarray, np.ndarray]:
    """Select most prominent person from YOLO result."""
    kpts = getattr(result, "keypoints", None)
    boxes = getattr(result, "boxes", None)

    if kpts is None or kpts.xy is None or len(kpts.xy) == 0:
        return np.zeros((17, 2), dtype=np.float32), np.zeros(17, dtype=np.float32)

    xy = kpts.xy.cpu().numpy().astype(np.float32)
    conf = (
        kpts.conf.cpu().numpy().astype(np.float32)
        if kpts.conf is not None
        else np.ones(xy.shape[:2], dtype=np.float32)
    )

    scores = conf.mean(axis=1)
    if boxes is not None and boxes.xyxy is not None and len(boxes.xyxy) == xy.shape[0]:
        box_arr = boxes.xyxy.cpu().numpy().astype(np.float32)
        areas = (
            np.clip(box_arr[:, 2] - box_arr[:, 0], 0, None)
            * np.clip(box_arr[:, 3] - box_arr[:, 1], 0, None)
        )
        box_conf = (
            boxes.conf.cpu().numpy().astype(np.float32)
            if boxes.conf is not None
            else np.ones(xy.shape[0], dtype=np.float32)
        )
        scores = scores * np.sqrt(areas + 1.0) * box_conf

    idx = int(np.argmax(scores))
    return xy[idx], conf[idx]


def get_per_frame_labels(video_name: str, n_frames: int, frame_labels: dict) -> list[int]:
    """Per-frame labels: -1=not lying, 0=transitioning, 1=lying.

    Default to -1 if frame not found in CSV (ADL videos typically all -1).
    """
    parts = video_name.split("-")
    seq = f"{parts[0]}-{parts[1]}" if len(parts) >= 3 else video_name
    seq_labels = frame_labels.get(seq, {})
    return [seq_labels.get(i + 1, -1) for i in range(n_frames)]


def find_contiguous_segments(
    per_frame: list[int],
    is_fall_video: bool,
) -> list[dict[str, int]]:
    """Find contiguous runs of same label, skipping label=0 (transitioning).

    For fall-* videos:
        - CSV label -1 → binary 0 (non-fall: person standing/walking before fall)
        - CSV label  0 → SKIP (ambiguous transitioning frames)
        - CSV label  1 → binary 1 (fall: person lying on ground after falling)

    For adl-* videos:
        - ALL frames → binary 0 (non-fall), regardless of CSV label.
          ADL videos may have label=1 (lying) but it's normal resting,
          NOT a fall event.
    """
    segments = []
    if not per_frame:
        return segments

    current_label = None
    current_start = 0

    for i, csv_label in enumerate(per_frame):
        # For ADL videos: skip transitioning frames, treat everything else as non-fall
        # For fall videos: skip transitioning frames, map -1→0, 1→1
        if csv_label == 0:
            if current_label is not None:
                segments.append({
                    "start": current_start,
                    "end": i - 1,
                    "label": current_label,
                })
                current_label = None
            continue

        # Map CSV label to binary
        if is_fall_video:
            binary_label = 1 if csv_label == 1 else 0
        else:
            # ADL: always non-fall (even if CSV says lying=1)
            binary_label = 0

        if binary_label != current_label:
            if current_label is not None:
                segments.append({
                    "start": current_start,
                    "end": i - 1,
                    "label": current_label,
                })
            current_label = binary_label
            current_start = i

    if current_label is not None:
        segments.append({
            "start": current_start,
            "end": len(per_frame) - 1,
            "label": current_label,
        })

    return segments


def sliding_window_samples(
    xy: np.ndarray,
    scores: np.ndarray,
    start: int,
    end: int,
    label: int,
    window_size: int,
    stride: int,
    min_frames: int,
    video_name: str,
    fps: float,
) -> list[dict[str, Any]]:
    """Create sliding window samples from a segment.

    Args:
        xy: (T_total, 17, 2) full video keypoints
        scores: (T_total, 17) full video scores
        start: segment start frame (0-indexed, inclusive)
        end: segment end frame (0-indexed, inclusive)
        label: 0 or 1
        window_size: target frames per window
        stride: step between windows
        min_frames: minimum frames for a valid window
        video_name: for metadata
        fps: video fps
    """
    seg_length = end - start + 1
    samples = []

    if seg_length < min_frames:
        return samples

    # If segment is shorter than window_size, use full segment as one sample
    if seg_length <= window_size:
        seg_xy = xy[start:end + 1]
        seg_sc = scores[start:end + 1]
        samples.append({
            "keypoint": seg_xy[np.newaxis, ...],          # (1, T, 17, 2)
            "keypoint_score": seg_sc[np.newaxis, ...],    # (1, T, 17)
            "label": label,
            "video_name": video_name,
            "total_frames": seg_xy.shape[0],
            "fps": float(fps),
            "segment_start": start,
            "segment_end": end,
        })
        return samples

    # Slide through the segment
    for w_start in range(0, seg_length - min_frames + 1, stride):
        w_end = min(w_start + window_size, seg_length)
        actual_start = start + w_start
        actual_end = start + w_end

        seg_xy = xy[actual_start:actual_end]
        seg_sc = scores[actual_start:actual_end]

        if seg_xy.shape[0] < min_frames:
            continue

        samples.append({
            "keypoint": seg_xy[np.newaxis, ...],
            "keypoint_score": seg_sc[np.newaxis, ...],
            "label": label,
            "video_name": f"{video_name}_w{w_start}",
            "total_frames": seg_xy.shape[0],
            "fps": float(fps),
            "segment_start": actual_start,
            "segment_end": actual_end,
        })

    return samples

# %% [markdown]
# ## 6. Extract Keypoints & Build Sliding Window Samples

# %%
video_files = sorted(DATA_DIR.glob("*.mp4"))
if CAM_FILTER:
    video_files = [v for v in video_files if CAM_FILTER in v.stem]

print(f"Found {len(video_files)} videos")

all_samples = []
stats = {
    "videos": 0,
    "total_frames": 0,
    "detected_frames": 0,
    "fall_samples": 0,
    "nonfall_samples": 0,
    "skipped_transitioning_frames": 0,
}

for video_path in tqdm(video_files, desc="Extracting"):
    video_name = video_path.stem
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"  [ERR] Cannot open: {video_name}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    all_xy = []
    all_scores = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame, imgsz=IMGSZ, conf=CONF, device=device, verbose=False)
        if results and len(results) > 0:
            xy, sc = select_person(results[0])
        else:
            xy, sc = np.zeros((17, 2), dtype=np.float32), np.zeros(17, dtype=np.float32)
        all_xy.append(xy)
        all_scores.append(sc)

    cap.release()
    n_frames = len(all_xy)

    if n_frames == 0:
        print(f"  [WARN] {video_name}: 0 frames")
        continue

    xy_arr = np.stack(all_xy).astype(np.float32)      # (T, 17, 2)
    sc_arr = np.stack(all_scores).astype(np.float32)   # (T, 17)

    stats["videos"] += 1
    stats["total_frames"] += n_frames
    stats["detected_frames"] += int((sc_arr.max(axis=1) > 0.1).sum())

    # Get per-frame labels and find contiguous segments
    per_frame = get_per_frame_labels(video_name, n_frames, frame_labels)
    stats["skipped_transitioning_frames"] += sum(1 for l in per_frame if l == 0)

    is_fall_video = video_name.startswith("fall")
    segments = find_contiguous_segments(per_frame, is_fall_video=is_fall_video)

    # Apply sliding window on each segment
    for seg in segments:
        stride = STRIDE_FALL if seg["label"] == 1 else STRIDE_NONFALL

        window_samples = sliding_window_samples(
            xy=xy_arr,
            scores=sc_arr,
            start=seg["start"],
            end=seg["end"],
            label=seg["label"],
            window_size=WINDOW_SIZE,
            stride=stride,
            min_frames=MIN_WINDOW_FRAMES,
            video_name=video_name,
            fps=fps,
        )

        for ws in window_samples:
            all_samples.append(ws)
            if ws["label"] == 1:
                stats["fall_samples"] += 1
            else:
                stats["nonfall_samples"] += 1

print(f"\n{'='*60}")
print(f"Extraction Summary:")
print(f"  Videos processed:       {stats['videos']}")
print(f"  Total frames:           {stats['total_frames']}")
print(f"  Detected frames:        {stats['detected_frames']}")
print(f"  Transitioning skipped:  {stats['skipped_transitioning_frames']}")
print(f"  Fall samples:           {stats['fall_samples']}")
print(f"  Non-fall samples:       {stats['nonfall_samples']}")
print(f"  Total samples:          {len(all_samples)}")
if len(all_samples) > 0:
    ratio = stats['fall_samples'] / len(all_samples) * 100
    print(f"  Fall ratio:             {ratio:.1f}%")
print(f"{'='*60}")

# %% [markdown]
# ## 7. Train/Val Split & Save
# Split by **video** (sequence name) to avoid data leakage

# %%
rng = random.Random(SEED)

# Group samples by base video name (without _wN suffix)
def get_base_video(name: str) -> str:
    """'fall-01-cam0_w10' → 'fall-01-cam0'"""
    return name.split("_w")[0] if "_w" in name else name

all_videos = sorted(set(get_base_video(s["video_name"]) for s in all_samples))
fall_videos = [v for v in all_videos if v.startswith("fall")]
adl_videos = [v for v in all_videos if v.startswith("adl")]

rng.shuffle(fall_videos)
rng.shuffle(adl_videos)

# Stratified split by video
n_val_fall = max(1, int(len(fall_videos) * VAL_RATIO))
n_val_adl = max(1, int(len(adl_videos) * VAL_RATIO))

val_videos = set(fall_videos[:n_val_fall] + adl_videos[:n_val_adl])
train_videos = set(fall_videos[n_val_fall:] + adl_videos[n_val_adl:])

print(f"Train videos ({len(train_videos)}): {sorted(train_videos)}")
print(f"Val videos ({len(val_videos)}):   {sorted(val_videos)}")

train_samples = [s for s in all_samples if get_base_video(s["video_name"]) in train_videos]
val_samples = [s for s in all_samples if get_base_video(s["video_name"]) in val_videos]

rng.shuffle(train_samples)
rng.shuffle(val_samples)

train_falls = sum(1 for s in train_samples if s["label"] == 1)
train_nonfalls = len(train_samples) - train_falls
val_falls = sum(1 for s in val_samples if s["label"] == 1)
val_nonfalls = len(val_samples) - val_falls

print(f"\nTrain: {len(train_samples)} ({train_falls} fall, {train_nonfalls} non-fall)")
print(f"Val:   {len(val_samples)} ({val_falls} fall, {val_nonfalls} non-fall)")

# Save
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for name, data in [
    ("train_data.pkl", train_samples),
    ("val_data.pkl", val_samples),
    ("all_data.pkl", all_samples),
]:
    path = OUTPUT_DIR / name
    with path.open("wb") as f:
        pickle.dump(data, f)
    print(f"Saved {len(data)} samples → {path}")

# %% [markdown]
# ## 8. Verify Output

# %%
with (OUTPUT_DIR / "train_data.pkl").open("rb") as f:
    loaded = pickle.load(f)

s0 = loaded[0]
print(f"Sample 0: {s0['video_name']}")
print(f"  keypoint shape:       {s0['keypoint'].shape}")
print(f"  keypoint_score shape: {s0['keypoint_score'].shape}")
print(f"  label: {s0['label']}")
print(f"  total_frames: {s0['total_frames']}")
print(f"  fps: {s0['fps']}")

assert s0["keypoint"].ndim == 4, "keypoint must be 4D: (M, T, 17, 2)"
assert s0["keypoint_score"].ndim == 3, "keypoint_score must be 3D: (M, T, 17)"
assert s0["keypoint"].shape[2] == 17, "Must have 17 joints"
assert s0["keypoint"].shape[3] == 2, "Must have 2 coords (x, y)"
print("\n✅ All shapes verified — compatible with CustomFallDataset!")

# Show distribution
from collections import Counter
print(f"\nLabel distribution in train_data.pkl:")
label_dist = Counter(s["label"] for s in loaded)
print(f"  {dict(label_dist)}")

# %% [markdown]
# ## 9. Usage with train.py
# ```bash
# python train.py \
#     --dataset-source custom \
#     --custom-train-pkl data/processed/urfall_pose/train_data.pkl \
#     --custom-val-pkl data/processed/urfall_pose/val_data.pkl \
#     --sequence-length 30 \
#     --epochs 50 \
#     --batch-size 128
# ```
