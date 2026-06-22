from __future__ import annotations

import argparse
import csv
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, cast

import cv2
import numpy as np
from datasets import load_from_disk
from tqdm import tqdm


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg")
Row = Mapping[str, Any]
ManifestRow = dict[str, Any]
WindowJob = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/omnifall/of-sta-cs")
    parser.add_argument("--video-root", default="data/omnifall/videos")
    parser.add_argument("--output-dir", default="data/video_frames")
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument("--image-ext", default=".jpg")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--num-workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_video_index(video_root: Path) -> dict[str, Path]:
    index = {}

    for path in video_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        relative_stem = path.relative_to(video_root).with_suffix("").as_posix()
        index.setdefault(relative_stem, path)
        index.setdefault(path.stem, path)

    return index


def resolve_video_path(row: Row, video_root: Path, video_index: dict[str, Path]) -> Path | None:
    raw_path = str(row["path"]).strip()
    dataset = str(row.get("dataset", "")).strip()
    candidates = [raw_path]

    if dataset:
        candidates.append(f"{dataset}/{raw_path}")

    for candidate in candidates:
        candidate_path = video_root / candidate

        if candidate_path.is_file():
            return candidate_path

        for extension in VIDEO_EXTENSIONS:
            path_with_extension = candidate_path.with_suffix(extension)
            if path_with_extension.is_file():
                return path_with_extension

        if candidate in video_index:
            return video_index[candidate]

    return None


def get_video_info(video_path: Path) -> tuple[float, int, float]:
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / source_fps if source_fps > 0 else 0.0
    capture.release()

    return source_fps, frame_count, duration


def make_windows(
    start: float,
    end: float,
    video_duration: float,
    window_seconds: float,
    stride_seconds: float,
) -> list[tuple[float, float]]:
    start = max(0.0, start)
    end = max(start, end)
    video_duration = max(video_duration, window_seconds)

    if end - start <= window_seconds:
        center = (start + end) / 2.0
        window_start = center - window_seconds / 2.0
        window_start = min(max(0.0, window_start), max(0.0, video_duration - window_seconds))
        return [(window_start, window_start + window_seconds)]

    windows = []
    last_start = min(end - window_seconds, video_duration - window_seconds)
    current = start

    while current <= last_start:
        windows.append((current, current + window_seconds))
        current += stride_seconds

    if windows and abs(windows[-1][0] - last_start) > 1e-6:
        windows.append((last_start, last_start + window_seconds))

    if not windows:
        windows.append((last_start, last_start + window_seconds))

    return windows


def read_frame_at_timestamp(
    capture: cv2.VideoCapture,
    timestamp: float,
    source_fps: float,
    frame_count: int,
) -> tuple[np.ndarray, float] | None:
    timestamp = max(0.0, timestamp)
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
    success, frame = capture.read()

    if success:
        frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if source_fps > 0 and frame_index >= 0:
            return frame, frame_index / source_fps
        return frame, timestamp

    if source_fps > 0 and frame_count > 0:
        frame_index = int(round(timestamp * source_fps))
        frame_index = min(max(0, frame_index), frame_count - 1)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()

        if success:
            return frame, frame_index / source_fps

    return None


def save_window_frames(
    video_path: Path,
    output_path: Path,
    window_start: float,
    fps: float,
    frame_total: int,
    image_ext: str,
    jpeg_quality: int,
) -> int:
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if source_fps <= 0:
        capture.release()
        raise RuntimeError(f"Invalid FPS for video: {video_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    saved = 0
    previous_frame = None
    previous_source_timestamp = 0.0
    timestamp_rows = []
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]

    for frame_number in range(frame_total):
        timestamp = window_start + frame_number / fps
        result = read_frame_at_timestamp(capture, timestamp, source_fps, frame_count)

        if result is None:
            frame = previous_frame
            source_timestamp = previous_source_timestamp
        else:
            frame, source_timestamp = result

        if frame is None:
            continue

        previous_frame = frame
        previous_source_timestamp = source_timestamp
        frame_name = f"frame_{frame_number:04d}{image_ext}"
        frame_path = output_path / frame_name

        if image_ext.lower() in {".jpg", ".jpeg"}:
            cv2.imwrite(str(frame_path), frame, encode_params)
        else:
            cv2.imwrite(str(frame_path), frame)

        timestamp_rows.append(
            {
                "frame_name": frame_name,
                "target_timestamp": timestamp,
                "source_timestamp": source_timestamp,
            }
        )
        saved += 1

    capture.release()

    if timestamp_rows:
        timestamp_path = output_path / "timestamps.csv"
        with timestamp_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["frame_name", "target_timestamp", "source_timestamp"])
            writer.writeheader()
            writer.writerows(timestamp_rows)

    return saved


def save_window_job(job: WindowJob) -> ManifestRow:
    try:
        saved_frames = save_window_frames(
            Path(job["video_path"]),
            Path(job["window_path"]),
            job["window_start"],
            job["fps"],
            job["frame_total"],
            job["image_ext"],
            job["jpeg_quality"],
        )
        status = "ok" if saved_frames == job["frame_total"] else "partial"
    except Exception as error:
        saved_frames = 0
        status = str(error)

    return make_manifest_row(
        job["row_index"],
        job["row"],
        job["video_path"],
        job["window_path"],
        job["window_id"],
        saved_frames,
        status,
        job["window_start"],
        job["window_end"],
    )


def split_frames_for_split(args: argparse.Namespace, split: str, video_index: dict[str, Path]) -> None:
    dataset_path = Path(args.dataset_dir) / split
    output_root = Path(args.output_dir) / Path(args.dataset_dir).name / split

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} already exists. Use --overwrite to replace it.")
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    dataset = load_from_disk(dataset_path)
    manifest_path = output_root / "manifest.csv"
    frame_total = int(round(args.fps * args.window_seconds))
    rows: list[ManifestRow] = []
    jobs: list[WindowJob] = []

    for row_index, raw_row in enumerate(tqdm(dataset, desc=f"scan {split}")):
        row = cast(dict[str, Any], raw_row)
        video_path = resolve_video_path(row, Path(args.video_root), video_index)

        if video_path is None:
            rows.append(make_manifest_row(row_index, row, "", "", "", 0, "missing_video"))
            continue

        try:
            _, _, duration = get_video_info(video_path)
            windows = make_windows(
                float(row["start"]),
                float(row["end"]),
                duration,
                args.window_seconds,
                args.stride_seconds,
            )
        except Exception as error:
            rows.append(make_manifest_row(row_index, row, video_path, "", "", 0, str(error)))
            continue

        for window_index, (window_start, window_end) in enumerate(windows):
            window_id = f"{row_index:06d}_{window_index:03d}"
            window_path = output_root / "frames" / window_id
            jobs.append(
                {
                    "row_index": row_index,
                    "row": row,
                    "video_path": str(video_path),
                    "window_path": str(window_path),
                    "window_id": window_id,
                    "window_start": window_start,
                    "window_end": window_end,
                    "fps": args.fps,
                    "frame_total": frame_total,
                    "image_ext": args.image_ext,
                    "jpeg_quality": args.jpeg_quality,
                }
            )

    if args.num_workers == 1:
        for job in tqdm(jobs, desc=f"split {split}"):
            rows.append(save_window_job(job))
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(save_window_job, job) for job in jobs]

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"split {split}"):
                rows.append(future.result())

    rows.sort(key=lambda item: (item["row_index"], str(item["window_id"])))
    write_manifest(manifest_path, rows)
    print(f"saved manifest to {manifest_path}")


def make_manifest_row(
    row_index: int,
    row: Row,
    video_path: Path | str,
    window_path: Path | str,
    window_id: str,
    saved_frames: int,
    status: str,
    window_start: float | str = "",
    window_end: float | str = "",
) -> ManifestRow:
    return {
        "row_index": row_index,
        "window_id": window_id,
        "video_path": str(video_path),
        "window_path": str(window_path),
        "source_path": row.get("path", ""),
        "label": row.get("label", ""),
        "start": row.get("start", ""),
        "end": row.get("end", ""),
        "window_start": window_start,
        "window_end": window_end,
        "subject": row.get("subject", ""),
        "cam": row.get("cam", ""),
        "dataset": row.get("dataset", ""),
        "saved_frames": saved_frames,
        "status": status,
    }


def write_manifest(manifest_path: Path, rows: list[ManifestRow]) -> None:
    fieldnames = [
        "row_index",
        "window_id",
        "video_path",
        "window_path",
        "source_path",
        "label",
        "start",
        "end",
        "window_start",
        "window_end",
        "subject",
        "cam",
        "dataset",
        "saved_frames",
        "status",
    ]

    with manifest_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be greater than 0.")

    if args.window_seconds <= 0:
        raise ValueError("--window-seconds must be greater than 0.")

    if args.stride_seconds <= 0:
        raise ValueError("--stride-seconds must be greater than 0.")

    if args.num_workers <= 0:
        raise ValueError("--num-workers must be greater than 0.")

    video_root = Path(args.video_root)
    video_index = build_video_index(video_root) if video_root.exists() else {}

    for split in args.splits:
        split_frames_for_split(args, split, video_index)


if __name__ == "__main__":
    main()
