# ============================================================================
# Kaggle Notebook: Extract COCO-17 Keypoints from Multiple Cameras Fall Dataset
# Using YOLO11-pose → Output .pkl for CustomFallDataset (train.py)
# ============================================================================
# Dataset structure:
#   dataset/dataset/chute01/cam1.avi ... cam8.avi
#   dataset/dataset/chute02/cam1.avi ... cam8.avi
#   ...
#   data_tuple3.csv: chute, cam, start, end, label (segments)
#
# Strategy:
#   1. Extract keypoints from ALL frames of each video using YOLO11-pose
#   2. Use CSV segments to slice labeled windows (fall=1, non-fall=0)
#   3. Apply sliding window for data augmentation on fall segments
#   4. Skip chute24 (no labels in CSV)
#   5. Output .pkl compatible with CustomFallDataset in train.py
# ============================================================================
# Setup:
#   1. Upload Multiple Cameras Fall dataset as Kaggle dataset
#   2. Enable GPU accelerator (Settings → Accelerator → GPU T4 x2)
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
DATA_DIR = Path("/kaggle/input/multiple-cameras-fall/dataset/dataset")
LABELS_CSV = Path("/kaggle/input/multiple-cameras-fall/data_tuple3.csv")
OUTPUT_DIR = Path("/kaggle/working/multicam_pose")
MODEL_NAME = "yolo11s-pose.pt"
IMGSZ = 640
CONF = 0.25
VAL_RATIO = 0.2
SEED = 42

# Sliding window augmentation settings
WINDOW_SIZE = 30          # frames per window (matches ~1 sec at 30fps)
STRIDE_FALL = 10          # stride for fall segments (more overlap → more fall samples)
STRIDE_NONFALL = 30       # stride for non-fall segments (less overlap)
MIN_WINDOW_FRAMES = 15    # minimum frames to form a valid window

# Skip chutes without labels
SKIP_CHUTES = {24}

# %% [markdown]
# ## 3. Parse Segment Labels from CSV

# %%
def parse_segments_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Parse data_tuple3.csv → list of segment dicts.

    Each segment: {chute, cam, start, end, label}
    """
    segments = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chute = int(float(row["chute"]))
            cam = int(float(row["cam"]))
            start = int(float(row["start"]))
            end = int(float(row["end"]))
            label = int(float(row["label"]))

            # Skip invalid cam numbers (CSV has cam=55 which is a typo)
            if cam > 8:
                continue

            segments.append({
                "chute": chute,
                "cam": cam,
                "start": start,
                "end": end,
                "label": label,
            })
    return segments

segments = parse_segments_csv(LABELS_CSV)
print(f"Total segments: {len(segments)}")

# Group by (chute, cam) for quick lookup
seg_by_video: dict[tuple[int, int], list[dict]] = defaultdict(list)
for seg in segments:
    seg_by_video[(seg["chute"], seg["cam"])].append(seg)

# Stats
fall_segs = sum(1 for s in segments if s["label"] == 1)
nonfall_segs = sum(1 for s in segments if s["label"] == 0)
print(f"Fall segments: {fall_segs}, Non-fall segments: {nonfall_segs}")

chutes_in_csv = sorted(set(s["chute"] for s in segments))
print(f"Chutes in CSV: {chutes_in_csv}")
print(f"Skipping chutes: {SKIP_CHUTES}")

# %% [markdown]
# ## 4. Load YOLO11-pose Model

# %%
from ultralytics import YOLO
import torch

device = "0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device} (CUDA: {torch.cuda.is_available()})")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

model = YOLO(MODEL_NAME)
print(f"Model loaded: {MODEL_NAME}")

# %% [markdown]
# ## 5. Helper Functions

# %%
def select_person(result) -> tuple[np.ndarray, np.ndarray]:
    """Select most prominent person from YOLO result.

    Returns:
        xy: (17, 2) keypoint coords
        conf: (17,) confidence scores
    """
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


def extract_video_keypoints(
    video_path: Path,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Extract keypoints from all frames of a video.

    Returns:
        all_xy: (T, 17, 2)
        all_scores: (T, 17)
        fps: float
        total_frames: int
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    all_xy, all_scores = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(
            source=frame, imgsz=IMGSZ, conf=CONF, device=device, verbose=False
        )
        if results and len(results) > 0:
            xy, sc = select_person(results[0])
        else:
            xy = np.zeros((17, 2), dtype=np.float32)
            sc = np.zeros(17, dtype=np.float32)
        all_xy.append(xy)
        all_scores.append(sc)

    cap.release()
    n = len(all_xy)

    if n == 0:
        return np.zeros((0, 17, 2), np.float32), np.zeros((0, 17), np.float32), fps, 0

    return np.stack(all_xy).astype(np.float32), np.stack(all_scores).astype(np.float32), fps, n


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
        start: segment start frame (0-indexed)
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

    # If segment is shorter than window_size, use the full segment as one sample
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
# ## 6. Extract Keypoints & Build Samples

# %%
all_samples = []
stats = {
    "videos_processed": 0,
    "videos_skipped": 0,
    "fall_samples": 0,
    "nonfall_samples": 0,
    "total_frames": 0,
}

chutes_to_process = sorted(set(s["chute"] for s in segments) - SKIP_CHUTES)

for chute_id in tqdm(chutes_to_process, desc="Chutes"):
    chute_dir = DATA_DIR / f"chute{chute_id:02d}"

    if not chute_dir.is_dir():
        print(f"  [WARN] Missing dir: {chute_dir}")
        continue

    for cam_id in range(1, 9):
        video_path = chute_dir / f"cam{cam_id}.avi"
        if not video_path.is_file():
            stats["videos_skipped"] += 1
            continue

        video_key = (chute_id, cam_id)
        video_segments = seg_by_video.get(video_key, [])

        if not video_segments:
            stats["videos_skipped"] += 1
            continue

        video_name = f"chute{chute_id:02d}_cam{cam_id}"

        try:
            xy, sc, fps, n_frames = extract_video_keypoints(video_path)
        except Exception as e:
            print(f"  [ERR] {video_name}: {e}")
            stats["videos_skipped"] += 1
            continue

        if n_frames == 0:
            stats["videos_skipped"] += 1
            continue

        stats["videos_processed"] += 1
        stats["total_frames"] += n_frames

        # Process each labeled segment
        for seg in video_segments:
            # Convert to 0-indexed
            seg_start = max(0, seg["start"] - 1)
            seg_end = min(n_frames - 1, seg["end"] - 1)
            seg_label = seg["label"]

            # Choose stride based on label for augmentation
            stride = STRIDE_FALL if seg_label == 1 else STRIDE_NONFALL

            window_samples = sliding_window_samples(
                xy=xy,
                scores=sc,
                start=seg_start,
                end=seg_end,
                label=seg_label,
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
print(f"  Videos processed: {stats['videos_processed']}")
print(f"  Videos skipped:   {stats['videos_skipped']}")
print(f"  Total frames:     {stats['total_frames']}")
print(f"  Fall samples:     {stats['fall_samples']}")
print(f"  Non-fall samples: {stats['nonfall_samples']}")
print(f"  Total samples:    {len(all_samples)}")
if stats['fall_samples'] + stats['nonfall_samples'] > 0:
    ratio = stats['fall_samples'] / (stats['fall_samples'] + stats['nonfall_samples']) * 100
    print(f"  Fall ratio:       {ratio:.1f}%")
print(f"{'='*60}")

# %% [markdown]
# ## 7. Train/Val Split & Save
# Split by **chute** (not random sample) to avoid data leakage across cameras

# %%
rng = random.Random(SEED)

# Split at chute level to prevent data leakage
all_chutes = sorted(set(s["chute"] for s in segments) - SKIP_CHUTES)
rng.shuffle(all_chutes)

n_val_chutes = max(1, int(len(all_chutes) * VAL_RATIO))
val_chutes = set(all_chutes[:n_val_chutes])
train_chutes = set(all_chutes[n_val_chutes:])

print(f"Train chutes ({len(train_chutes)}): {sorted(train_chutes)}")
print(f"Val chutes ({len(val_chutes)}):   {sorted(val_chutes)}")

def get_chute_from_name(name: str) -> int:
    """Extract chute number from video_name like 'chute01_cam1_w0'."""
    return int(name.split("_")[0].replace("chute", ""))

train_samples = [s for s in all_samples if get_chute_from_name(s["video_name"]) in train_chutes]
val_samples = [s for s in all_samples if get_chute_from_name(s["video_name"]) in val_chutes]

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
print(f"\nLabel distribution in train_data.pkl:")
from collections import Counter
label_dist = Counter(s["label"] for s in loaded)
print(f"  {dict(label_dist)}")

# %% [markdown]
# ## 9. Usage with train.py
# ```bash
# python train.py \
#     --dataset-source custom \
#     --custom-train-pkl data/processed/multicam_pose/train_data.pkl \
#     --custom-val-pkl data/processed/multicam_pose/val_data.pkl \
#     --sequence-length 30 \
#     --epochs 50 \
#     --batch-size 128
# ```
