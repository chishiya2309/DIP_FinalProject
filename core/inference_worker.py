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
        yolo_model: str = "yolov8s-pose.pt",
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
        # Run YOLO tracking with conf=0.25 (lower to catch lying poses), but rely on kp quality filter for false positives
        # Run YOLO tracking with conf=0.25 and use ByteTrack which is much better at keeping IDs during fast motion
        results = self.yolo.track(frame, persist=True, verbose=False, classes=[0], conf=0.25, tracker="bytetrack.yaml")
        
        fall_probs = {}
        self.valid_track_ids = set() # Store valid IDs for this frame
        
        if len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().numpy()
            
            if result.keypoints is not None:
                keypoints_all = result.keypoints.data.cpu().numpy()
                
                for i, track_id in enumerate(track_ids):
                    track_id = int(track_id)
                    x1, y1, x2, y2 = map(int, boxes[i])
                    
                    w, h = x2 - x1, y2 - y1
                    if w < 50 or h < 50:
                        continue
                        
                    kp = keypoints_all[i]
                    avg_kp_conf = np.mean(kp[:, 2])
                    if avg_kp_conf < 0.3:
                        continue
                        
                    self.valid_track_ids.add(track_id)
                    self.person_last_seen[track_id] = timestamp
                    
                    if track_id not in self.person_history:
                        # --- CƠ CHẾ KẾ THỪA ID (ID INHERITANCE) ---
                        # Nếu YOLO bị mất dấu và cấp ID mới, ta sẽ tìm ID vừa biến mất gần nhất (< 1.5s) để kế thừa lịch sử
                        inherited = False
                        for old_tid, old_last_seen in list(self.person_last_seen.items()):
                            if old_tid != track_id and timestamp - old_last_seen < 1.5:
                                # Kế thừa buffer của người cũ
                                self.person_history[track_id] = self.person_history.get(old_tid, collections.deque(maxlen=self.sequence_length))
                                if hasattr(self, "last_probs") and old_tid in self.last_probs:
                                    self.last_probs[track_id] = self.last_probs[old_tid]
                                inherited = True
                                break
                                
                        if not inherited:
                            self.person_history[track_id] = collections.deque(maxlen=self.sequence_length)
                            self.fall_state[track_id] = False
                            
                    self.person_history[track_id].append((kp, timestamp))
                    
                    # Inference as long as we have a few frames (e.g., >= 5 frames)
                    # fit_sequence handles padding automatically
                    if len(self.person_history[track_id]) >= 5:
                        history_list = list(self.person_history[track_id])
                        kps = np.stack([item[0] for item in history_list]) # (len, 17, 3)
                        ts = np.array([item[1] for item in history_list], dtype=np.float32) # (len,)
                        
                        # Normalize (fit_sequence handles padding to 72)
                        kps_seq, ts_seq, mask_seq = self.fit_sequence(kps, ts)
                        
                        kps_tensor = torch.from_numpy(kps_seq).unsqueeze(0).to(self.device) # (1, 72, 17, 3)
                        ts_tensor = torch.from_numpy(ts_seq).unsqueeze(0).to(self.device) # (1, 72)
                        mask_tensor = torch.from_numpy(mask_seq).unsqueeze(0).to(self.device) # (1, 72)
                        
                        # Infer
                        with torch.no_grad():
                            logits = self.model(kps_tensor, mask=mask_tensor, timestamps=ts_tensor)
                            prob = torch.softmax(logits, dim=1)[0, 1].item()
                            
                        # 3. Làm mượt xác suất FALL (EMA Smoothing)
                        # Lấy xác suất cũ để mượt hóa (tránh nhảy số giật cục)
                        old_prob = getattr(self, "last_probs", {}).get(track_id, prob)
                        smoothed_prob = 0.6 * prob + 0.4 * old_prob
                        if not hasattr(self, "last_probs"): self.last_probs = {}
                        self.last_probs[track_id] = smoothed_prob
                            
                        fall_probs[track_id] = smoothed_prob
                        
                        # Khởi tạo counter và timer nếu chưa có
                        if not hasattr(self, "fall_counter"): self.fall_counter = {}
                        if not hasattr(self, "fall_latch_timer"): self.fall_latch_timer = {}
                        
                        # Chỉ cần 4 frame liên tục (~0.16s) là đủ để xác nhận một cú ngã thực sự (vì ngã xảy ra rất nhanh)
                        if smoothed_prob >= self.conf_threshold:
                            self.fall_counter[track_id] = self.fall_counter.get(track_id, 0) + 1
                        else:
                            self.fall_counter[track_id] = 0
                            
                        # Nếu phát hiện ngã, kích hoạt LATCH timer (giữ báo động trong 120 frames ~ 5 giây)
                        if self.fall_counter[track_id] >= 4:
                            self.fall_latch_timer[track_id] = 120
                            
                        # Cập nhật trạng thái dựa trên LATCH timer
                        current_latch = self.fall_latch_timer.get(track_id, 0)
                        if current_latch > 0:
                            self.fall_state[track_id] = True
                            self.fall_latch_timer[track_id] = current_latch - 1
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
            if hasattr(self, "fall_counter") and tid in self.fall_counter:
                del self.fall_counter[tid]
            if hasattr(self, "fall_latch_timer") and tid in self.fall_latch_timer:
                del self.fall_latch_timer[tid]
            if hasattr(self, "last_probs") and tid in self.last_probs:
                del self.last_probs[tid]
                
        # Draw on frame (Visualization)
        annotated_frame = frame.copy()
        if len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().numpy()
            
            for i, track_id in enumerate(track_ids):
                track_id = int(track_id)
                if track_id not in getattr(self, 'valid_track_ids', set()):
                    continue
                    
                x1, y1, x2, y2 = map(int, boxes[i])
                
                is_fall = self.fall_state.get(track_id, False)
                color = (0, 0, 255) if is_fall else (0, 255, 0)
                
                label = f"ID: {track_id} {'FALL' if is_fall else 'OK'}"
                prob = fall_probs.get(track_id, 0.0)
                
                if track_id in self.person_history and len(self.person_history[track_id]) >= 5:
                     label += f" ({prob:.2f})"
                else:
                     buf_len = len(self.person_history.get(track_id, []))
                     label += f" (buf: {buf_len}/5)"
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw the 17 keypoints for visual debugging
                if result.keypoints is not None and len(result.keypoints.data) > i:
                    kp = result.keypoints.data[i].cpu().numpy() # (17, 3) where [x, y, conf]
                    for j in range(17):
                        kx, ky, kconf = kp[j]
                        if kconf > 0.3: # Only draw confident keypoints
                            cv2.circle(annotated_frame, (int(kx), int(ky)), 4, (0, 255, 255), -1) # Yellow dots

        return annotated_frame, fall_probs
