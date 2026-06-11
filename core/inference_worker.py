import collections
import time
from typing import Dict, Optional, Tuple, Any

import numpy as np
import torch
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from .model import build_model


class FallDetectorWorker:
    def __init__(
        self,
        model_path: str = "best.pt",
        yolo_model: str = "yolov8n-pose.pt",
        sequence_length: int = 72,
        fps: float = 24.0,
        device: str = "auto",
        conf_threshold: float = 0.5,
    ):
        self.sequence_length = sequence_length
        self.fps = fps
        self.conf_threshold = conf_threshold
        
        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        print(f"[Worker] Using device: {self.device}")

        # Load YOLO
        if YOLO is None:
            raise RuntimeError("Ultralytics is not installed. Please pip install ultralytics")
        self.yolo = YOLO(yolo_model)

        # Load PoseBiGRU
        self.model = build_model(fps=fps).to(self.device)
        self.model.eval()
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if "model_state" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state"])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"[Worker] Loaded PoseBiGRU from {model_path}")
        except FileNotFoundError:
            print(f"[Worker] WARNING: Model path {model_path} not found. Using untrained weights for testing!")
            
        # State tracking
        self.person_history: Dict[int, collections.deque] = {}
        self.person_last_seen: Dict[int, float] = {}
        
        # Output state
        self.fall_state: Dict[int, bool] = {}

    def fit_sequence(self, keypoints: np.ndarray, timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Copied from train_hung/train.py PoseWindowDataset.fit_sequence
        Ensures input tensor is exactly shape (sequence_length, 17, 3)
        """
        length = keypoints.shape[0]
        output = np.zeros((self.sequence_length, 17, 3), dtype=np.float32)
        output_timestamps = np.zeros(self.sequence_length, dtype=np.float32)
        mask = np.zeros(self.sequence_length, dtype=bool)

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

    def process_frame(self, frame: np.ndarray, timestamp: float) -> Tuple[np.ndarray, Dict[int, float]]:
        """
        Process a single frame.
        Returns:
            - frame: annotated frame
            - fall_probs: Dictionary mapping track_id to fall_probability
        """
        # Run YOLO tracking
        results = self.yolo.track(frame, persist=True, verbose=False, classes=[0])  # class 0 is person
        
        fall_probs = {}
        
        if len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().numpy()
            
            # Important: YOLO might return keypoints even if id is None, but here we check id is not None
            if result.keypoints is not None:
                keypoints_all = result.keypoints.data.cpu().numpy() # shape: (N, 17, 3)
                
                for i, track_id in enumerate(track_ids):
                    track_id = int(track_id)
                    self.person_last_seen[track_id] = timestamp
                    
                    if track_id not in self.person_history:
                        self.person_history[track_id] = collections.deque(maxlen=self.sequence_length)
                        self.fall_state[track_id] = False
                        
                    kp = keypoints_all[i] # shape (17, 3)
                    self.person_history[track_id].append((kp, timestamp))
                    
                    # Inference ONLY if we have full sequence (72 frames)
                    if len(self.person_history[track_id]) == self.sequence_length:
                        history_list = list(self.person_history[track_id])
                        kps = np.stack([item[0] for item in history_list]) # (72, 17, 3)
                        ts = np.array([item[1] for item in history_list], dtype=np.float32) # (72,)
                        
                        # Normalize
                        kps_seq, ts_seq, mask_seq = self.fit_sequence(kps, ts)
                        
                        kps_tensor = torch.from_numpy(kps_seq).unsqueeze(0).to(self.device) # (1, 72, 17, 3)
                        ts_tensor = torch.from_numpy(ts_seq).unsqueeze(0).to(self.device) # (1, 72)
                        mask_tensor = torch.from_numpy(mask_seq).unsqueeze(0).to(self.device) # (1, 72)
                        
                        # Infer
                        with torch.no_grad():
                            logits = self.model(kps_tensor, mask=mask_tensor, timestamps=ts_tensor)
                            prob = torch.softmax(logits, dim=1)[0, 1].item()
                            
                        fall_probs[track_id] = prob
                        
                        if prob >= self.conf_threshold:
                            self.fall_state[track_id] = True
                        else:
                            self.fall_state[track_id] = False
                    else:
                        fall_probs[track_id] = 0.0

        # Cleanup disappeared tracks
        cleanup_threshold = 2.0 # 2 seconds
        to_delete = []
        for tid, last_ts in self.person_last_seen.items():
            if timestamp - last_ts > cleanup_threshold:
                to_delete.append(tid)
        
        for tid in to_delete:
            del self.person_last_seen[tid]
            if tid in self.person_history:
                del self.person_history[tid]
            if tid in self.fall_state:
                del self.fall_state[tid]
                
        # Draw on frame (Visualization)
        annotated_frame = frame.copy()
        if len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().numpy()
            
            for i, track_id in enumerate(track_ids):
                track_id = int(track_id)
                x1, y1, x2, y2 = map(int, boxes[i])
                
                is_fall = self.fall_state.get(track_id, False)
                color = (0, 0, 255) if is_fall else (0, 255, 0)
                
                label = f"ID: {track_id} {'FALL' if is_fall else 'OK'}"
                prob = fall_probs.get(track_id, 0.0)
                
                if track_id in self.person_history and len(self.person_history[track_id]) == self.sequence_length:
                     label += f" ({prob:.2f})"
                else:
                     buf_len = len(self.person_history.get(track_id, []))
                     label += f" (buf: {buf_len}/{self.sequence_length})"
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return annotated_frame, fall_probs
