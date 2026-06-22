from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))
from model import build_model


ManifestRow = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-source", choices=["manifest", "ntu-hrnet", "custom", "combined"], default="manifest")
    parser.add_argument("--pose-dir", default="data/omnifall_pose/of-sta-cs")
    parser.add_argument("--ntu-pkl", default="data/ntu60_hrnet.pkl")
    parser.add_argument("--custom-train-pkl", default="data/processed/mmaction/train_data.pkl")
    parser.add_argument("--custom-val-pkl", default="data/processed/mmaction/val_data.pkl")
    parser.add_argument("--ntu-protocol", choices=["xsub", "xview"], default="xsub")
    parser.add_argument("--output-dir", default="runs/fall_detection")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--fall-labels", nargs="+", type=int)
    parser.add_argument("--sequence-length", type=int, default=72)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--fall-weight", type=float, default=1.25)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--beta", type=float, default=1.25)
    parser.add_argument("--threshold-min", type=float, default=0.30)
    parser.add_argument("--threshold-max", type=float, default=0.80)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--disable-balanced-sampler", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_manifest(path: Path) -> list[ManifestRow]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def binary_label(label: str | int, fall_labels: set[int]) -> int:
    return int(int(label) in fall_labels)


def resolve_fps(args: argparse.Namespace) -> float:
    if args.fps is not None:
        return float(args.fps)

    if args.dataset_source == "ntu-hrnet":
        return 30.0

    return 24.0


def resolve_fall_labels(args: argparse.Namespace) -> set[int]:
    if args.fall_labels is not None:
        return set(args.fall_labels)

    if args.dataset_source == "ntu-hrnet":
        return {42}

    return {1, 2}


class PoseWindowDataset(Dataset):
    def __init__(self, manifest_path: Path, fall_labels: set[int], sequence_length: int, fps: float) -> None:
        self.sequence_length = sequence_length
        self.fps = fps
        self.samples = []

        for row in read_manifest(manifest_path):
            if row.get("pose_status") != "ok":
                continue

            pose_path = resolve_pose_path(Path(str(row.get("pose_path", ""))), manifest_path)

            if not pose_path.is_file():
                continue

            self.samples.append(
                {
                    "pose_path": pose_path,
                    "label": binary_label(row["label"], fall_labels),
                    "window_id": row.get("window_id", ""),
                }
            )

        if not self.samples:
            raise RuntimeError(f"No valid pose samples found in {manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        data = np.load(sample["pose_path"], allow_pickle=True)
        keypoints = data["keypoints"].astype(np.float32)
        keypoints = keypoints[:, :, :3]

        if "timestamps" in data:
            timestamps = data["timestamps"].astype(np.float32)
        elif "source_timestamps" in data:
            timestamps = data["source_timestamps"].astype(np.float32)
        else:
            timestamps = np.arange(keypoints.shape[0], dtype=np.float32) / np.float32(self.fps)

        keypoints, timestamps, mask = self.fit_sequence(keypoints, timestamps)

        return (
            torch.from_numpy(keypoints),
            torch.from_numpy(timestamps),
            torch.from_numpy(mask),
            torch.tensor(sample["label"], dtype=torch.long),
        )

    def fit_sequence(self, keypoints: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        length = keypoints.shape[0]
        output = np.zeros((self.sequence_length, 17, 3), dtype=np.float32)
        output_timestamps = np.zeros(self.sequence_length, dtype=np.float32)
        mask = np.zeros(self.sequence_length, dtype=bool)
        timestamps = timestamps[:length].astype(np.float32)

        if length >= self.sequence_length:
            output[:] = keypoints[: self.sequence_length]
            output_timestamps[:] = timestamps[: self.sequence_length]
            mask[:] = True
        else:
            output[:length] = keypoints
            output_timestamps[:length] = timestamps

            if length > 0:
                start = output_timestamps[length - 1]
            else:
                start = 0.0

            for index in range(length, self.sequence_length):
                output_timestamps[index] = start + (index - length + 1) / self.fps

            mask[:length] = True

        output[:, :, 2] = np.clip(output[:, :, 2], 0.0, 1.0)

        return output, output_timestamps, mask

    def labels(self) -> list[int]:
        return [int(sample["label"]) for sample in self.samples]


class CustomFallDataset(Dataset):
    def __init__(
        self,
        pkl_path: Path,
        sequence_length: int,
        fps: float,
        training: bool,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        self.sequence_length = sequence_length
        self.fps = fps
        self.training = training

        if not pkl_path.is_file():
            raise FileNotFoundError(f"Pickle not found: {pkl_path}")

        with pkl_path.open("rb") as file:
            samples = pickle.load(file)
            
        self.samples = self.limit_samples(samples, max_samples, seed)

    def limit_samples(self, samples: list[dict[str, Any]], max_samples: int | None, seed: int) -> list[dict[str, Any]]:
        if max_samples is None or len(samples) <= max_samples:
            return samples
        rng = random.Random(seed)
        positives = [sample for sample in samples if sample["label"] == 1]
        negatives = [sample for sample in samples if sample["label"] == 0]
        rng.shuffle(positives)
        rng.shuffle(negatives)
        positive_count = min(len(positives), max_samples // 2)
        negative_count = max_samples - positive_count
        selected = positives[:positive_count] + negatives[:negative_count]
        rng.shuffle(selected)
        return selected

    def __len__(self) -> int:
        return len(self.samples)

    def select_person(self, scores: np.ndarray) -> int:
        if scores.shape[0] == 1:
            return 0
        return int(scores.mean(axis=(1, 2)).argmax())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        keypoints = sample["keypoint"].astype(np.float32)
        scores = sample["keypoint_score"].astype(np.float32)
        
        person_index = self.select_person(scores)
        xy = keypoints[person_index]
        confidence = scores[person_index]
        pose = np.concatenate((xy, confidence[..., None]), axis=-1).astype(np.float32)
        timestamps = np.arange(pose.shape[0], dtype=np.float32) / np.float32(self.fps)
        
        pose, timestamps, mask = self.fit_sequence(pose, timestamps)

        return (
            torch.from_numpy(pose),
            torch.from_numpy(timestamps),
            torch.from_numpy(mask),
            torch.tensor(sample["label"], dtype=torch.long),
        )

    def fit_sequence(self, keypoints: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        length = keypoints.shape[0]
        output = np.zeros((self.sequence_length, 17, 3), dtype=np.float32)
        output_timestamps = np.zeros(self.sequence_length, dtype=np.float32)
        mask = np.zeros(self.sequence_length, dtype=bool)

        if length >= self.sequence_length:
            max_start = length - self.sequence_length
            if self.training and max_start > 0:
                start = random.randint(0, max_start)
            else:
                start = max_start // 2

            end = start + self.sequence_length
            output[:] = keypoints[start:end]
            output_timestamps[:] = timestamps[start:end]
            mask[:] = True
        else:
            output[:length] = keypoints
            output_timestamps[:length] = timestamps

            if length > 0:
                start_time = output_timestamps[length - 1]
            else:
                start_time = 0.0

            for index in range(length, self.sequence_length):
                output_timestamps[index] = start_time + (index - length + 1) / self.fps

            mask[:length] = True

        output[:, :, 2] = np.clip(output[:, :, 2], 0.0, 1.0)
        return output, output_timestamps, mask

    def labels(self) -> list[int]:
        return [int(sample["label"]) for sample in self.samples]


class NtuHrnetPoseDataset(Dataset):
    def __init__(
        self,
        pkl_path: Path,
        split_name: str,
        fall_labels: set[int],
        sequence_length: int,
        fps: float,
        training: bool,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        self.sequence_length = sequence_length
        self.fps = fps
        self.training = training

        if not pkl_path.is_file():
            raise FileNotFoundError(f"NTU HRNet pickle not found: {pkl_path}")

        with pkl_path.open("rb") as file:
            data = pickle.load(file)

        split = data.get("split", {})

        if split_name not in split:
            raise ValueError(f"Split {split_name!r} not found. Available splits: {sorted(split)}")

        frame_dirs = set(split[split_name])
        samples = []

        for annotation in data["annotations"]:
            if annotation["frame_dir"] not in frame_dirs:
                continue

            samples.append(
                {
                    "annotation": annotation,
                    "label": binary_label(annotation["label"], fall_labels),
                    "frame_dir": annotation["frame_dir"],
                    "source_label": int(annotation["label"]),
                }
            )

        self.samples = self.limit_samples(samples, max_samples, seed)

        if not self.samples:
            raise RuntimeError(f"No NTU HRNet samples found for split {split_name}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        annotation = sample["annotation"]
        keypoints = annotation["keypoint"].astype(np.float32)
        scores = annotation["keypoint_score"].astype(np.float32)
        person_index = self.select_person(scores)
        xy = keypoints[person_index]
        confidence = scores[person_index]
        pose = np.concatenate((xy, confidence[..., None]), axis=-1).astype(np.float32)
        timestamps = np.arange(pose.shape[0], dtype=np.float32) / np.float32(self.fps)
        pose, timestamps, mask = self.fit_sequence(pose, timestamps)

        return (
            torch.from_numpy(pose),
            torch.from_numpy(timestamps),
            torch.from_numpy(mask),
            torch.tensor(sample["label"], dtype=torch.long),
        )

    def fit_sequence(self, keypoints: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        length = keypoints.shape[0]
        output = np.zeros((self.sequence_length, 17, 3), dtype=np.float32)
        output_timestamps = np.zeros(self.sequence_length, dtype=np.float32)
        mask = np.zeros(self.sequence_length, dtype=bool)

        if length >= self.sequence_length:
            max_start = length - self.sequence_length
            if self.training and max_start > 0:
                start = random.randint(0, max_start)
            else:
                start = max_start // 2

            end = start + self.sequence_length
            output[:] = keypoints[start:end]
            output_timestamps[:] = timestamps[start:end]
            mask[:] = True
        else:
            output[:length] = keypoints
            output_timestamps[:length] = timestamps

            if length > 0:
                start_time = output_timestamps[length - 1]
            else:
                start_time = 0.0

            for index in range(length, self.sequence_length):
                output_timestamps[index] = start_time + (index - length + 1) / self.fps

            mask[:length] = True

        output[:, :, 2] = np.clip(output[:, :, 2], 0.0, 1.0)
        return output, output_timestamps, mask

    def select_person(self, scores: np.ndarray) -> int:
        if scores.shape[0] == 1:
            return 0

        return int(scores.mean(axis=(1, 2)).argmax())

    def limit_samples(self, samples: list[dict[str, Any]], max_samples: int | None, seed: int) -> list[dict[str, Any]]:
        if max_samples is None or len(samples) <= max_samples:
            return samples

        rng = random.Random(seed)
        positives = [sample for sample in samples if sample["label"] == 1]
        negatives = [sample for sample in samples if sample["label"] == 0]
        rng.shuffle(positives)
        rng.shuffle(negatives)
        positive_count = min(len(positives), max_samples // 2)
        negative_count = max_samples - positive_count
        selected = positives[:positive_count] + negatives[:negative_count]
        rng.shuffle(selected)
        return selected

    def labels(self) -> list[int]:
        return [int(sample["label"]) for sample in self.samples]

    def source_labels(self) -> list[int]:
        return [int(sample["source_label"]) for sample in self.samples]


def resolve_pose_path(pose_path: Path, manifest_path: Path) -> Path:
    if pose_path.is_file() or pose_path.is_absolute():
        return pose_path

    candidate = manifest_path.parent / pose_path

    if candidate.is_file():
        return candidate

    return pose_path


def make_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    balanced_sampler: bool,
) -> DataLoader:
    sampler = None

    if balanced_sampler:
        labels = np.asarray(dataset.labels(), dtype=np.int64)
        class_counts = np.bincount(labels, minlength=2).astype(np.float64)
        class_weights = 1.0 / np.clip(class_counts, 1.0, None)
        sample_weights = class_weights[labels].astype(np.float64)
        sampler = WeightedRandomSampler(
            weights=sample_weights.tolist(),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any | None,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    use_amp = scaler is not None

    for keypoints, timestamps, mask, labels in tqdm(loader, desc="train", leave=False):
        keypoints = keypoints.to(device, non_blocking=True)
        timestamps = timestamps.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with amp_context(use_amp):
            logits = model(keypoints, mask=mask, timestamps=timestamps)
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.detach()) * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


def make_grad_scaler(device: torch.device) -> Any | None:
    if device.type != "cuda":
        return None

    grad_scaler = getattr(torch, "GradScaler", None)

    if grad_scaler is not None:
        return grad_scaler("cuda")

    cuda_amp = getattr(torch.cuda, "amp", None)

    if cuda_amp is None:
        return None

    return getattr(cuda_amp, "GradScaler")()


def amp_context(enabled: bool) -> Any:
    if not enabled:
        return nullcontext()

    autocast = getattr(torch, "autocast", None)

    if autocast is not None:
        return autocast(device_type="cuda")

    cuda_amp = getattr(torch.cuda, "amp", None)

    if cuda_amp is None:
        return nullcontext()

    return getattr(cuda_amp, "autocast")()


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    probabilities = []
    labels_all = []

    for keypoints, timestamps, mask, labels in tqdm(loader, desc="eval", leave=False):
        keypoints = keypoints.to(device, non_blocking=True)
        timestamps = timestamps.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(keypoints, mask=mask, timestamps=timestamps)
        loss = criterion(logits, labels)
        prob_fall = torch.softmax(logits, dim=1)[:, 1]

        batch_size = labels.size(0)
        total_loss += float(loss.detach()) * batch_size
        total_items += batch_size
        probabilities.append(prob_fall.cpu().numpy())
        labels_all.append(labels.cpu().numpy())

    return (
        total_loss / max(total_items, 1),
        np.concatenate(probabilities),
        np.concatenate(labels_all),
    )


def threshold_report(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
    beta: float,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    rows = []
    best = None
    thresholds = np.arange(threshold_min, threshold_max + threshold_step / 2.0, threshold_step)

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(np.int64)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0,
        )
        beta2 = beta * beta
        fbeta = (1 + beta2) * precision * recall / max(beta2 * precision + recall, 1e-12)
        false_positives = int(((predictions == 1) & (labels == 0)).sum())
        false_negatives = int(((predictions == 0) & (labels == 1)).sum())
        true_positives = int(((predictions == 1) & (labels == 1)).sum())
        true_negatives = int(((predictions == 0) & (labels == 0)).sum())
        row = {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "fbeta": float(fbeta),
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_positives": true_positives,
            "true_negatives": true_negatives,
        }
        rows.append(row)

        if best is None or row["fbeta"] > best["fbeta"]:
            best = row

    return rows, best or rows[0]


def write_threshold_report(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    fps = resolve_fps(args)
    args.resolved_fps = fps
    output_dir = Path(args.output_dir)

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    fall_labels = resolve_fall_labels(args)

    if args.dataset_source == "ntu-hrnet":
        train_split = f"{args.ntu_protocol}_train"
        val_split = f"{args.ntu_protocol}_val"
        train_dataset = NtuHrnetPoseDataset(
            Path(args.ntu_pkl),
            train_split,
            fall_labels,
            args.sequence_length,
            fps,
            training=True,
            max_samples=args.max_train_samples,
            seed=args.seed,
        )
        val_dataset = NtuHrnetPoseDataset(
            Path(args.ntu_pkl),
            val_split,
            fall_labels,
            args.sequence_length,
            fps,
            training=False,
            max_samples=args.max_val_samples,
            seed=args.seed,
        )
    elif args.dataset_source == "custom":
        train_dataset = CustomFallDataset(
            Path(args.custom_train_pkl),
            args.sequence_length,
            fps,
            training=True,
            max_samples=args.max_train_samples,
            seed=args.seed,
        )
        val_dataset = CustomFallDataset(
            Path(args.custom_val_pkl),
            args.sequence_length,
            fps,
            training=False,
            max_samples=args.max_val_samples,
            seed=args.seed,
        )
    elif args.dataset_source == "combined":
        train_split = f"{args.ntu_protocol}_train"
        val_split = f"{args.ntu_protocol}_val"
        
        ntu_train = NtuHrnetPoseDataset(Path(args.ntu_pkl), train_split, {42}, args.sequence_length, fps, True, args.max_train_samples, args.seed)
        custom_train = CustomFallDataset(Path(args.custom_train_pkl), args.sequence_length, fps, True, args.max_train_samples, args.seed)
        train_dataset = torch.utils.data.ConcatDataset([ntu_train, custom_train])
        train_dataset.labels = lambda: ntu_train.labels() + custom_train.labels()
        
        ntu_val = NtuHrnetPoseDataset(Path(args.ntu_pkl), val_split, {42}, args.sequence_length, fps, False, args.max_val_samples, args.seed)
        custom_val = CustomFallDataset(Path(args.custom_val_pkl), args.sequence_length, fps, False, args.max_val_samples, args.seed)
        val_dataset = torch.utils.data.ConcatDataset([ntu_val, custom_val])
        val_dataset.labels = lambda: ntu_val.labels() + custom_val.labels()
    else:
        train_dataset = PoseWindowDataset(
            Path(args.pose_dir) / args.train_split / "manifest.csv",
            fall_labels,
            args.sequence_length,
            fps,
        )
        val_dataset = PoseWindowDataset(
            Path(args.pose_dir) / args.val_split / "manifest.csv",
            fall_labels,
            args.sequence_length,
            fps,
        )

    use_balanced_sampler = args.balanced_sampler or args.dataset_source in ["ntu-hrnet", "custom", "combined"]

    if args.disable_balanced_sampler:
        use_balanced_sampler = False

    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        args.num_workers,
        shuffle=True,
        balanced_sampler=use_balanced_sampler,
    )
    val_loader = make_loader(
        val_dataset,
        args.batch_size,
        args.num_workers,
        shuffle=False,
        balanced_sampler=False,
    )
    model = build_model(
        fps=fps,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    class_weights = torch.tensor([1.0, args.fall_weight], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = make_grad_scaler(device)
    best_score = -1.0
    best_epoch = 0
    history = []

    print(f"device: {device}")
    print(f"dataset source: {args.dataset_source}")
    print(f"fps: {fps}")
    print(f"fall labels: {sorted(fall_labels)}")
    print(f"balanced sampler: {use_balanced_sampler}")
    print(f"train samples: {len(train_dataset)}")
    print(f"val samples: {len(val_dataset)}")
    print(f"train labels: {np.bincount(np.asarray(train_dataset.labels()), minlength=2).tolist()}")
    print(f"val labels: {np.bincount(np.asarray(val_dataset.labels()), minlength=2).tolist()}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            args.grad_clip,
        )
        val_loss, probabilities, labels = predict(model, val_loader, criterion, device)
        threshold_rows, best_threshold = threshold_report(
            probabilities,
            labels,
            args.threshold_min,
            args.threshold_max,
            args.threshold_step,
            args.beta,
        )
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "threshold": best_threshold,
        }
        history.append(metrics)
        print(
            "epoch "
            f"{epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"thr={best_threshold['threshold']:.2f} | "
            f"precision={best_threshold['precision']:.4f} | "
            f"recall={best_threshold['recall']:.4f} | "
            f"fbeta={best_threshold['fbeta']:.4f}"
        )

        save_checkpoint(output_dir / "last.pt", model, optimizer, args, epoch, metrics)

        if best_threshold["fbeta"] > best_score:
            best_score = best_threshold["fbeta"]
            best_epoch = epoch
            save_checkpoint(output_dir / "best.pt", model, optimizer, args, epoch, metrics)
            write_threshold_report(output_dir / "threshold_report.csv", threshold_rows)

        if epoch - best_epoch >= args.patience:
            print(f"early stopping at epoch {epoch}")
            break

    with (output_dir / "history.json").open("w") as file:
        json.dump(history, file, indent=2)

    print(f"best epoch: {best_epoch}")
    print(f"saved best checkpoint to {output_dir / 'best.pt'}")
    print(f"saved threshold report to {output_dir / 'threshold_report.csv'}")


if __name__ == "__main__":
    main()
