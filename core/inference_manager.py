import threading
import time
import cv2
import numpy as np
from core.inference_worker import FallDetectorWorker

class InferenceManager:
    def __init__(self, camera_manager):
        self.camera_manager = camera_manager
        
        # Cấu hình worker
        self.worker = FallDetectorWorker()
        
        self.running = False
        self.lock = threading.Lock()
        
        # Shared state
        self.latest_annotated_frame = None
        self.has_fall = False
        self.falling_track_ids = []
        self.new_falling_ids = []
        
        self.cooldown_time = 3.0
        self.cooldowns = {}
        
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()

    def _inference_loop(self):
        last_frame_time = 0
        while self.running:
            # Lấy frame mới nhất từ camera
            frame_rgb = self.camera_manager.get_frame()
            if frame_rgb is None:
                time.sleep(0.01)
                continue
                
            # Đảm bảo không xử lý lặp lại cùng 1 frame quá nhanh nếu camera chậm
            current_time = time.time()
            if current_time - last_frame_time < 0.01: # max 100fps
                time.sleep(0.01)
                continue
            last_frame_time = current_time

            # Chuyển đổi RGB sang BGR để OpenCV / YOLO xử lý
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            try:
                # Inference
                annotated_bgr, fall_probs = self.worker.process_frame(frame_bgr, timestamp=current_time)
                
                # Update trạng thái
                falling_ids = []
                new_falling_ids = []
                # FallDetectorWorker tự cập nhật self.worker.fall_state
                for tid, is_fall in self.worker.fall_state.items():
                    if is_fall:
                        falling_ids.append(tid)
                        if tid not in self.cooldowns or current_time - self.cooldowns[tid] > self.cooldown_time:
                            self.cooldowns[tid] = current_time
                            new_falling_ids.append(tid)
                            
                # Cleanup cooldowns cho các track_id không còn tồn tại
                active_tids = self.worker.valid_track_ids if hasattr(self.worker, 'valid_track_ids') else []
                for tid in list(self.cooldowns.keys()):
                    if tid not in active_tids and current_time - self.cooldowns[tid] > self.cooldown_time * 2:
                        del self.cooldowns[tid]
                        
                # Chuyển lại sang RGB để UI (CustomTkinter/Pillow) hiển thị
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                
                with self.lock:
                    self.latest_annotated_frame = annotated_rgb
                    self.has_fall = len(falling_ids) > 0
                    self.falling_track_ids = falling_ids
                    self.new_falling_ids.extend(new_falling_ids)
                    
            except Exception as e:
                print(f"[InferenceManager] Lỗi xử lý frame: {e}")
                time.sleep(0.1)

    def get_annotated_frame(self):
        with self.lock:
            if self.latest_annotated_frame is not None:
                return self.latest_annotated_frame.copy()
            return self.camera_manager.get_frame() # Fallback to raw frame
            
    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            
    def update_threshold(self, threshold):
        self.worker.conf_threshold = threshold
        
    def update_cooldown(self, cooldown):
        self.cooldown_time = float(cooldown)
        
    def pop_new_falls(self):
        with self.lock:
            falls = self.new_falling_ids.copy()
            self.new_falling_ids.clear()
            return falls
