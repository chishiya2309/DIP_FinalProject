from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
ManifestRow = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", default="data/video_frames/of-sta-cs")
    parser.add_argument("--output-dir", default="data/omnifall_pose")
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(device: str | None) -> str | None:
    if device in {None, "", "none"}:
        return None

    if device != "auto":
        return device

    import torch

    if torch.cuda.is_available():
        return "0"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def read_manifest(path: Path) -> list[ManifestRow]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def write_manifest(path: Path, rows: list[ManifestRow]) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        return

    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def list_frames(window_path: Path) -> list[Path]:
    return sorted(
        path
        for path in window_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_frame_timestamps(window_path: Path, frame_paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    timestamp_path = window_path / "timestamps.csv"
    target_by_name = {}
    source_by_name = {}

    if timestamp_path.is_file():
        with timestamp_path.open(newline="") as file:
            for row in csv.DictReader(file):
                frame_name = str(row.get("frame_name", ""))
                target_by_name[frame_name] = float(row.get("target_timestamp", 0.0))
                source_by_name[frame_name] = float(row.get("source_timestamp", target_by_name[frame_name]))

    target_timestamps = []
    source_timestamps = []

    for index, frame_path in enumerate(frame_paths):
        fallback = float(index)
        target_timestamps.append(target_by_name.get(frame_path.name, fallback))
        source_timestamps.append(source_by_name.get(frame_path.name, target_timestamps[-1]))

    return (
        np.asarray(target_timestamps, dtype=np.float32),
        np.asarray(source_timestamps, dtype=np.float32),
    )


def select_person(result: Any) -> tuple[np.ndarray, np.ndarray, float]:
    keypoints = getattr(result, "keypoints", None)
    boxes = getattr(result, "boxes", None)

    if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
        return np.zeros((17, 3), dtype=np.float32), np.zeros(4, dtype=np.float32), 0.0

    xy = keypoints.xy.cpu().numpy().astype(np.float32)

    if keypoints.conf is None:
        confidence = np.ones(xy.shape[:2], dtype=np.float32)
    else:
        confidence = keypoints.conf.cpu().numpy().astype(np.float32)

    scores = confidence.mean(axis=1)
    box_array = np.zeros((xy.shape[0], 4), dtype=np.float32)
    box_confidence = np.ones(xy.shape[0], dtype=np.float32)

    if boxes is not None and boxes.xyxy is not None and len(boxes.xyxy) == xy.shape[0]:
        box_array = boxes.xyxy.cpu().numpy().astype(np.float32)
        widths = np.clip(box_array[:, 2] - box_array[:, 0], 0.0, None)
        heights = np.clip(box_array[:, 3] - box_array[:, 1], 0.0, None)
        areas = widths * heights

        if boxes.conf is not None:
            box_confidence = boxes.conf.cpu().numpy().astype(np.float32)

        scores = scores * np.sqrt(areas + 1.0) * box_confidence

    person_index = int(np.argmax(scores))
    pose = np.concatenate((xy[person_index], confidence[person_index, :, None]), axis=1)

    return pose.astype(np.float32), box_array[person_index], float(box_confidence[person_index])


def run_pose_on_window(
    model: Any,
    frame_paths: list[Path],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    results = model.predict(
        source=[str(path) for path in frame_paths],
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        batch=args.batch,
        stream=False,
        verbose=False,
    )

    keypoints = []
    boxes = []
    person_confidence = []

    for result in results:
        pose, box, score = select_person(result)
        keypoints.append(pose)
        boxes.append(box)
        person_confidence.append(score)

    return (
        np.stack(keypoints).astype(np.float32),
        np.stack(boxes).astype(np.float32),
        np.asarray(person_confidence, dtype=np.float32),
    )


def process_split(args: argparse.Namespace, split: str, model: Any) -> None:
    frames_root = Path(args.frames_dir) / split
    input_manifest = frames_root / "manifest.csv"
    output_root = Path(args.output_dir) / Path(args.frames_dir).name / split

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} already exists. Use --overwrite to replace it.")
        shutil.rmtree(output_root)

    pose_root = output_root / "poses"
    pose_root.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(input_manifest)
    output_rows = []

    for row in tqdm(rows, desc=f"pose {split}"):
        output_row = dict(row)
        window_id = str(row.get("window_id", ""))
        window_path = Path(str(row.get("window_path", "")))
        pose_path = pose_root / f"{window_id}.npz"

        if row.get("status") != "ok":
            output_row.update({"pose_path": "", "pose_status": "skipped_frame_status"})
            output_rows.append(output_row)
            continue

        if not window_path.is_dir():
            output_row.update({"pose_path": "", "pose_status": "missing_window_path"})
            output_rows.append(output_row)
            continue

        frame_paths = list_frames(window_path)

        if not frame_paths:
            output_row.update({"pose_path": "", "pose_status": "missing_frames"})
            output_rows.append(output_row)
            continue

        try:
            target_timestamps, source_timestamps = read_frame_timestamps(window_path, frame_paths)
            keypoints, boxes, person_confidence = run_pose_on_window(model, frame_paths, args)
            np.savez_compressed(
                pose_path,
                keypoints=keypoints,
                boxes=boxes,
                person_confidence=person_confidence,
                frame_paths=np.asarray([str(path) for path in frame_paths]),
                target_timestamps=target_timestamps,
                source_timestamps=source_timestamps,
                timestamps=source_timestamps,
            )
            status = "ok"
        except Exception as error:
            pose_path = Path("")
            status = str(error)

        output_row.update({"pose_path": str(pose_path), "pose_status": status})
        output_rows.append(output_row)

    write_manifest(output_root / "manifest.csv", output_rows)
    print(f"saved manifest to {output_root / 'manifest.csv'}")


def main() -> None:
    args = parse_args()

    if args.conf < 0.0 or args.conf > 1.0:
        raise ValueError("--conf must be between 0 and 1.")

    if args.batch <= 0:
        raise ValueError("--batch must be greater than 0.")

    if not Path(args.weights).is_file():
        raise FileNotFoundError(f"YOLO weights not found: {args.weights}")

    from ultralytics import YOLO

    args.device = resolve_device(args.device)
    print(f"using device: {args.device}")

    model = YOLO(args.weights)

    for split in args.splits:
        process_split(args, split, model)


if __name__ == "__main__":
    main()
