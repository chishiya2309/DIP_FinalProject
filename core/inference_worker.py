import collections
import math
import time
import os
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
        model_path: str = os.path.join("train_hung", "runs", "fall_detection", "best.pt"),
        yolo_model: str = "yolo11s-pose.pt", # Chuyển sang bản Small (s) để nhận diện người nằm tốt hơn bản Nano
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
        self.last_probs: Dict[int, float] = {}
        self.fall_reasons: Dict[int, str] = {}
        
        # Rule-based fallback states
        self.rule_states = collections.defaultdict(lambda: {
            "suspect_count": 0,
            "lying_count": 0,
            "last_center": None,
            "last_time": None,
            "last_bbox": None,
            "last_seen_time": None,
            "alarm_until": 0.0,
            "last_reason": "OK",
        })
        self.rule_lie_frames = 3              # Giảm từ 6 → 3: phản ứng nhanh hơn khi nằm
        self.rule_suspect_frames = 2           # Giảm từ 4 → 2: chỉ cần 2 frame nghi ngờ liên tục
        self.rule_alarm_hold = 2.0
        self.rule_model_soft_threshold = 0.25

        # Cấu hình lọc người thật (chống detect nhầm ghế, vật thể)
        self.min_visible_keypoints = 3         # Giảm từ 5 → 3: người nằm thường bị che nhiều keypoint
        self.min_kp_conf_for_visible = 0.3     # Ngưỡng conf để tính keypoint là "thấy được"
        self.min_box_conf = 0.25               # Ngưỡng YOLO detection confidence tối thiểu
        # Keypoint indices: 5,6=shoulders, 11,12=hips — cần ít nhất 1 vai HOẶC 1 hông
        self.core_upper_body_indices = [5, 6]   # Vai trái, vai phải
        self.core_lower_body_indices = [11, 12]  # Hông trái, hông phải

    def _is_valid_person(self, kp: np.ndarray, box_conf: float) -> bool:
        """
        Kiểm tra detection có phải là người thật hay không.
        Lọc 3 tầng:
          1. YOLO box confidence >= min_box_conf
          2. Số keypoint visible >= min_visible_keypoints
          3. Có ít nhất 1 vai HOẶC 1 hông visible (cấu trúc skeleton người)
        """
        # Tầng 1: YOLO detection confidence
        if box_conf < self.min_box_conf:
            return False

        # Tầng 2: Đếm keypoint visible
        visible_count = int(np.sum(kp[:, 2] > self.min_kp_conf_for_visible))
        if visible_count < self.min_visible_keypoints:
            return False

        # Tầng 3: Kiểm tra cấu trúc skeleton — phải có ít nhất 1 vai hoặc 1 hông
        has_shoulder = any(
            kp[idx, 2] > self.min_kp_conf_for_visible
            for idx in self.core_upper_body_indices
            if idx < len(kp)
        )
        has_hip = any(
            kp[idx, 2] > self.min_kp_conf_for_visible
            for idx in self.core_lower_body_indices
            if idx < len(kp)
        )
        if not has_shoulder and not has_hip:
            return False

        return True

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

        output_timestamps[length:] = timestamps[-1]
        
        return output, output_timestamps, mask

    def _visible_point(self, kpts, idx, min_conf=0.25):
        if kpts is None or len(kpts) <= idx:
            return None
        p = kpts[idx]
        if len(p) >= 3:
            x, y, c = float(p[0]), float(p[1]), float(p[2])
            if c < min_conf:
                return None
            return np.array([x, y], dtype=np.float32)
        return np.array([float(p[0]), float(p[1])], dtype=np.float32)

    def _mean_visible(self, kpts, indices, min_conf=0.25):
        pts = []
        for idx in indices:
            p = self._visible_point(kpts, idx, min_conf)
            if p is not None:
                pts.append(p)
        if not pts:
            return None
        return np.mean(pts, axis=0)

    def _angle_to_horizontal(self, p1, p2):
        dx = float(p2[0] - p1[0])
        dy = float(p2[1] - p1[1])
        angle = abs(math.degrees(math.atan2(dy, dx)))
        if angle > 90:
            angle = 180 - angle
        return angle

    def _rule_based_fall_check(self, track_id, bbox, kpts, frame_shape, model_prob, current_time):
        H, W = frame_shape[:2]
        x1, y1, x2, y2 = map(float, bbox)
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        aspect = bw / bh
        center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
        state = self.rule_states[track_id]

        shoulder = self._mean_visible(kpts, [5, 6])
        hip = self._mean_visible(kpts, [11, 12])
        
        torso_horizontal = False
        torso_vertical = False
        if shoulder is not None and hip is not None:
            torso_angle = self._angle_to_horizontal(shoulder, hip)
            torso_horizontal = torso_angle <= 35
            torso_vertical = torso_angle >= 60

        # Đếm số keypoint phần dưới cơ thể (hips, knees, ankles: 11-16)
        lower_body_kpts = 0
        if kpts is not None:
            lower_body_kpts = sum(1 for i in range(11, 17) if i < len(kpts) and kpts[i][2] > 0.3)

        lying_by_box = aspect >= 1.1  # Giảm từ 1.25 → 1.1: bắt cả tư thế co mình khi ngã
        touching_bottom = y2 >= H * 0.95
        
        # Nếu không thấy phần dưới cơ thể và box bị cắt dưới (ngồi gần webcam),
        # bounding box chỉ chứa thân trên nên sẽ bị rộng ngang -> Không đánh giá là nằm
        if lower_body_kpts == 0 and y2 > H * 0.6:
            lying_by_box = False
            
        # Nếu bị cắt ngang dưới cùng và không thấy hông
        if touching_bottom and hip is None:
            lying_by_box = False
            
        # Nếu tính được góc thân và người đang thẳng đứng thì chắc chắn không nằm
        if torso_vertical:
            lying_by_box = False

        lower_body_area = y2 >= H * 0.45
        
        max_h = state.get("max_h", bh)
        # Decay nhẹ mỗi frame, nhưng vẫn cập nhật ngay nếu bh tăng
        max_h = max(bh, max_h * 0.97)
        state["max_h"] = max_h
            
        collapsed = False
        if max_h > 50 and bh < max_h * 0.6 and lower_body_area:  # 0.5 → 0.6: nhạy hơn khi chiều cao giảm
            collapsed = True
            
        sudden_motion = False
        if state["last_center"] is not None and state["last_time"] is not None:
            dt = max(1e-3, current_time - state["last_time"])
            dist = float(np.linalg.norm(center - state["last_center"]))
            speed = dist / dt
            sudden_motion = speed >= H * 0.80

        # Tính tỉ lệ chiều cao hiện tại so với max → nếu giảm mạnh = ngã
        height_ratio_drop = False
        if max_h > 50:
            ratio = bh / max_h
            height_ratio_drop = ratio < 0.55  # Chiều cao giảm > 45% so với cao nhất

        model_suspicious = model_prob >= self.rule_model_soft_threshold
        score = 0
        if lying_by_box: score += 2
        if torso_horizontal: score += 2
        if collapsed: score += 2
        if height_ratio_drop: score += 1  # Tín hiệu phụ: chiều cao giảm đột ngột
        if lower_body_area: score += 1
        if sudden_motion: score += 1
        if model_suspicious: score += 1

        suspicious = score >= 3
        if suspicious:
            state["suspect_count"] += 1
        else:
            state["suspect_count"] = max(0, state["suspect_count"] - 1)

        if lying_by_box or torso_horizontal or collapsed:
            state["lying_count"] += 1
        else:
            state["lying_count"] = max(0, state["lying_count"] - 1)

        is_fall_by_rule = False
        reason = "OK"

        if state["lying_count"] >= self.rule_lie_frames and suspicious:
            is_fall_by_rule = True
            reason = "LYING_POSTURE"
        if state["suspect_count"] >= self.rule_suspect_frames and (lying_by_box or torso_horizontal):
            is_fall_by_rule = True
            reason = "RULE_FALL"
        if model_prob >= self.conf_threshold:
            is_fall_by_rule = True
            reason = "MODEL_FALL"

        if is_fall_by_rule:
            state["alarm_until"] = current_time + self.rule_alarm_hold

        if current_time <= state["alarm_until"]:
            is_fall_by_rule = True
            if reason == "OK":
                reason = state["last_reason"]

        state["last_center"] = center
        state["last_time"] = current_time
        state["last_bbox"] = bbox
        state["last_seen_time"] = current_time
        state["last_reason"] = reason

        return is_fall_by_rule, reason

    def _check_lost_tracks(self, current_time):
        lost_falls = {}
        for track_id, state in self.rule_states.items():
            last_seen = state.get("last_seen_time")
            if last_seen is None:
                continue
            time_lost = current_time - last_seen
            recently_suspicious = (
                state.get("suspect_count", 0) >= self.rule_suspect_frames
                or state.get("lying_count", 0) >= self.rule_lie_frames
            )
            # Người đang đứng (box cao) đột ngột biến mất → rất có thể đã ngã
            was_standing = False
            last_bbox = state.get("last_bbox")
            if last_bbox is not None:
                _, _, bx2, by2 = map(float, last_bbox)
                bx1, by1 = float(last_bbox[0]), float(last_bbox[1])
                last_bh = by2 - by1
                last_bw = bx2 - bx1
                was_standing = last_bh > last_bw * 1.2 and last_bh > 80  # Box cao hơn rộng = đang đứng

            if 0.2 <= time_lost <= 2.0 and (recently_suspicious or was_standing):
                lost_falls[track_id] = "LOST_AFTER_SUSPECTED_FALL"
                state["alarm_until"] = current_time + self.rule_alarm_hold
        return lost_falls

    def process_frame(self, frame: np.ndarray, timestamp: float) -> Tuple[np.ndarray, Dict[int, float]]:
        """
        Process a single frame.
        Returns:
            - frame: annotated frame
            - fall_probs: Dictionary mapping track_id to fall_probability
        """
        # YOLO tracking: dùng conf thấp để YOLO không bỏ sót người nằm,
        # nhưng sau đó lọc bằng _is_valid_person() (3 tầng) để loại ghế/vật thể
        results = self.yolo.track(
            frame, 
            persist=True, 
            verbose=False, 
            classes=[0], 
            conf=0.10,  # Giữ conf thấp cho YOLO để không miss người nằm
            iou=0.45,
            imgsz=960,
            tracker="bytetrack.yaml"  # ByteTrack không dùng GMC nên tránh crash optical flow khi frame size thay đổi
        )
        
        fall_probs = {}
        self.valid_track_ids = set()
        
        if len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().numpy()
            box_confs = result.boxes.conf.cpu().numpy()  # Lấy confidence của từng box
            
            if result.keypoints is not None:
                keypoints_all = result.keypoints.data.cpu().numpy()
                
                for i, track_id in enumerate(track_ids):
                    track_id = int(track_id)
                    x1, y1, x2, y2 = map(int, boxes[i])
                    
                    w, h = x2 - x1, y2 - y1
                    if w < 30 or h < 30:  # 50→30: bắt người nằm xa camera có box nhỏ
                        continue
                        
                    kp = keypoints_all[i]
                    box_conf = float(box_confs[i])
                    
                    # === LỌC 3 TẦNG: Chỉ chấp nhận detection là người thật ===
                    if not self._is_valid_person(kp, box_conf):
                        continue
                    
                    avg_kp_conf = np.mean(kp[:, 2])
                    kp_reliable = avg_kp_conf >= 0.20
                        
                    self.valid_track_ids.add(track_id)
                    self.person_last_seen[track_id] = timestamp
                    
                    if track_id not in self.person_history:
                        # --- CƠ CHẾ KẾ THỪA ID (ID INHERITANCE) ---
                        # Nếu YOLO bị mất dấu và cấp ID mới, tìm ID vừa biến mất gần nhất (<1.5s) để kế thừa lịch sử
                        inherited = False
                        consumed_tid = None
                        for old_tid, old_last_seen in list(self.person_last_seen.items()):
                            if old_tid != track_id and timestamp - old_last_seen < 1.5:
                                # Kế thừa buffer keypoints
                                old_buf = self.person_history.get(old_tid, collections.deque(maxlen=self.sequence_length))
                                self.person_history[track_id] = collections.deque(old_buf, maxlen=self.sequence_length)
                                
                                if old_tid in self.last_probs:
                                    self.last_probs[track_id] = self.last_probs[old_tid]

                                # Kế thừa rule_states (lying_count, suspect_count, alarm_until, max_h)
                                # → ID mới KHÔNG phải tích lũy lại từ 0
                                if old_tid in self.rule_states:
                                    old_rule = self.rule_states[old_tid]
                                    self.rule_states[track_id].update({
                                        "suspect_count": old_rule.get("suspect_count", 0),
                                        "lying_count": old_rule.get("lying_count", 0),
                                        "max_h": old_rule.get("max_h", 0),
                                        "alarm_until": old_rule.get("alarm_until", 0.0),
                                        "last_reason": old_rule.get("last_reason", "OK"),
                                    })
                                if old_tid in self.fall_state:
                                    self.fall_state[track_id] = self.fall_state[old_tid]
                                    
                                inherited = True
                                consumed_tid = old_tid
                                break
                                
                        if not inherited:
                            self.person_history[track_id] = collections.deque(maxlen=self.sequence_length)
                            self.fall_state[track_id] = False
                        else:
                            # Xóa ngay để tránh track khác kế thừa trùng trong cùng frame
                            if consumed_tid in self.person_last_seen:
                                del self.person_last_seen[consumed_tid]
                                
                    # 1. Chạy Rule-Based Fallback trước (kể cả khi kp yếu)
                    current_model_prob = self.last_probs.get(track_id, 0.0)
                    rule_fall, rule_reason = self._rule_based_fall_check(
                        track_id=track_id,
                        bbox=boxes[i],
                        kpts=kp if kp_reliable else None,
                        frame_shape=frame.shape,
                        model_prob=current_model_prob,
                        current_time=timestamp,
                    )
                    
                    if rule_fall:
                        fall_probs[track_id] = max(current_model_prob, self.conf_threshold)
                        self.fall_state[track_id] = True
                        self.fall_reasons[track_id] = rule_reason
                    else:
                        fall_probs[track_id] = current_model_prob
                        self.fall_state[track_id] = False
                        self.fall_reasons[track_id] = "OK"
                            
                    # 2. Chạy Model Inference nếu kp đủ tin cậy
                    if kp_reliable:
                        self.person_history[track_id].append((kp, timestamp))
                        
                        if len(self.person_history[track_id]) >= 5:
                            history_list = list(self.person_history[track_id])
                            kps = np.stack([item[0] for item in history_list]) # (len, 17, 3)
                            ts = np.array([item[1] for item in history_list], dtype=np.float32) # (len,)
                            
                            kps_seq, ts_seq, mask_seq = self.fit_sequence(kps, ts)
                            
                            kps_tensor = torch.from_numpy(kps_seq).unsqueeze(0).to(self.device)
                            ts_tensor = torch.from_numpy(ts_seq).unsqueeze(0).to(self.device)
                            mask_tensor = torch.from_numpy(mask_seq).unsqueeze(0).to(self.device)
                            
                            with torch.no_grad():
                                logits = self.model(kps_tensor, mask=mask_tensor, timestamps=ts_tensor)
                                prob = torch.softmax(logits, dim=1)[0, 1].item()
                                
                            # EMA Smoothing
                            old_prob = self.last_probs.get(track_id, prob)
                            smoothed_prob = 0.6 * prob + 0.4 * old_prob
                            self.last_probs[track_id] = smoothed_prob
                            
                            # Cập nhật state nếu model báo ngã (chỉ ghi đè nếu model cao hơn threshold và rule_reason đang là OK)
                            fall_probs[track_id] = max(fall_probs[track_id], smoothed_prob)
                            if smoothed_prob >= self.conf_threshold:
                                self.fall_state[track_id] = True
                                if not rule_fall:
                                    self.fall_reasons[track_id] = "MODEL_FALL"
                        
            # Bổ sung _check_lost_tracks sau khi đã lặp xong keypoints
            lost_falls = self._check_lost_tracks(timestamp)
            for tid, reason in lost_falls.items():
                fall_probs[tid] = max(fall_probs.get(tid, 0.0), self.conf_threshold)
                self.fall_state[tid] = True
                self.fall_reasons[tid] = reason
                
        else:
            # Nếu không có detection nào, kiểm tra track bị mất
            lost_falls = self._check_lost_tracks(timestamp)
            for tid, reason in lost_falls.items():
                fall_probs[tid] = max(fall_probs.get(tid, 0.0), self.conf_threshold)
                self.fall_state[tid] = True
                self.fall_reasons[tid] = reason

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
            if tid in self.last_probs:
                del self.last_probs[tid]
            if tid in self.fall_reasons:
                del self.fall_reasons[tid]
            if tid in self.rule_states:
                del self.rule_states[tid]
                
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
                
                reason = self.fall_reasons.get(track_id, "OK")
                prob = fall_probs.get(track_id, 0.0)
                
                if is_fall:
                    label = f"ID: {track_id} FALL ({prob:.2f}) {reason}"
                else:
                    label = f"ID: {track_id} OK ({prob:.2f})"
                    if track_id not in self.person_history or len(self.person_history[track_id]) < 5:
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

        for tid, reason in lost_falls.items():
            if getattr(self, "fall_state", {}).get(tid, False) and tid not in getattr(self, 'valid_track_ids', set()):
                cv2.putText(
                    annotated_frame,
                    f"ID: {tid} FALL {reason} (LOST)",
                    (30, 50 + 30 * int(tid % 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                
        return annotated_frame, fall_probs
