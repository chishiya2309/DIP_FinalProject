"""
Bộ quản lý luồng suy diễn phát hiện té ngã (Inference Manager).

Module này chịu trách nhiệm:
  - Quản lý một luồng riêng (threading) chạy vòng lặp suy diễn phát hiện ngã (YOLO + PoseBiGRU).
  - Nhận luồng khung hình từ CameraManager, chuyển đổi màu sắc, và chuyển tới InferenceWorker xử lý.
  - Cập nhật và lưu trữ kết quả suy diễn (bounding box vẽ trên hình, xác suất ngã, trạng thái ngã của các ID).
  - Quản lý cơ chế Cooldown (tránh gửi cảnh báo trùng lặp liên tục cho một ID).
"""

import threading
import time
import cv2
import numpy as np
from core.inference_worker import FallDetectorWorker

class InferenceManager:
    """Quản lý luồng xử lý và đồng bộ hóa kết quả phát hiện té ngã.

    Lớp này đóng vai trò điều phối giữa luồng lấy ảnh từ camera và bộ xử lý thuật toán phát hiện ngã,
    đảm bảo luồng chạy ổn định thời gian thực, đồng thời cập nhật trạng thái cảnh báo trên GUI.

    Attributes:
        camera_manager: Instance quản lý camera đầu vào.
        worker: Bộ xử lý thuật toán cốt lõi FallDetectorWorker (YOLOv11-Pose + PoseBiGRU).
        running: Cờ hoạt động của luồng suy diễn.
        lock: Khóa dùng để đồng bộ hóa truy cập tài nguyên dùng chung giữa luồng xử lý và luồng GUI.
        latest_annotated_frame: Khung hình RGB mới nhất đã được vẽ các bounding box và khung xương.
        has_fall: Trạng thái có phát hiện té ngã trên khung hình hiện tại hay không.
        falling_track_ids: Danh sách các ID đối tượng đang trong trạng thái ngã.
        new_falling_ids: Danh sách các ID đối tượng mới phát hiện ngã (chưa báo động).
        cooldown_time: Thời gian debounce/cooldown (giây) trước khi xóa trạng thái ngã khi đối tượng đứng dậy.
        cooldowns: Từ điển lưu dấu thời gian thấy đối tượng ngã lần cuối.
    """

    def __init__(self, camera_manager):
        """Khởi tạo InferenceManager kết nối với CameraManager cụ thể.

        Args:
            camera_manager: Trình quản lý camera để lấy khung hình.
        """
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
        """Khởi động luồng chạy ngầm xử lý suy diễn (Inference Loop Thread)."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()

    def _inference_loop(self):
        """Vòng lặp chạy ngầm thực hiện suy diễn phát hiện té ngã.
        Lấy khung hình từ camera, chuyển đổi định dạng BGR, đưa qua FallDetectorWorker,
        phân tích kết quả trả về, quản lý cooldown/debounce cho các ID ngã,
        và cập nhật kết quả đầu ra (khung hình đã vẽ bbox, trạng thái ngã) một cách an toàn.
        """
        last_frame_time = 0
        # Task 3: Throttle – inference target = 12 FPS (không cần nhiều hơn trên CPU-only)
        _TARGET_INTERVAL = 1.0 / 12.0
        # Task 0: FPS counter
        _fps_frame_count = 0
        _fps_window_start = time.time()
        while self.running:
            # Lấy frame mới nhất từ camera
            frame_rgb = self.camera_manager.get_frame()
            if frame_rgb is None:
                time.sleep(0.01)
                continue
            current_time = time.time()
            elapsed = current_time - last_frame_time
            if elapsed < _TARGET_INTERVAL:
                time.sleep(_TARGET_INTERVAL - elapsed)
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
                active_tids = self.worker.valid_track_ids if hasattr(self.worker, 'valid_track_ids') else []
                
                # FallDetectorWorker tự cập nhật self.worker.fall_state
                for tid, is_fall in self.worker.fall_state.items():
                    if is_fall:
                        falling_ids.append(tid)
                        if tid not in self.cooldowns:
                            # Lần đầu tiên thấy ngã -> Gửi cảnh báo
                            self.cooldowns[tid] = current_time
                            new_falling_ids.append(tid)
                        else:
                            # Đang nằm -> Cập nhật thời gian cuối cùng thấy nằm (debounce)
                            self.cooldowns[tid] = current_time
                    else:
                        # Hết ngã (Đứng lên). Xóa khỏi cooldown nếu đã đứng lên đủ lâu (debounce)
                        if tid in self.cooldowns and current_time - self.cooldowns[tid] > self.cooldown_time:
                            del self.cooldowns[tid]
                            
                # Cleanup cooldowns cho các track_id không còn tồn tại trên màn hình
                for tid in list(self.cooldowns.keys()):
                    if tid not in active_tids and current_time - self.cooldowns.get(tid, current_time) > self.cooldown_time * 2:
                        del self.cooldowns[tid]
                        
                # Chuyển lại sang RGB để UI (CustomTkinter/Pillow) hiển thị
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                
                with self.lock:
                    self.latest_annotated_frame = annotated_rgb
                    self.has_fall = len(falling_ids) > 0
                    self.falling_track_ids = falling_ids
                    self.new_falling_ids.extend(new_falling_ids)
                # Task 0: FPS log mỗi 5 giây
                _fps_frame_count += 1
                if current_time - _fps_window_start >= 5.0:
                    infer_fps = _fps_frame_count / (current_time - _fps_window_start)
                    print(f"[PERF] infer_fps={infer_fps:.1f}")
                    _fps_frame_count = 0
                    _fps_window_start = current_time
            except Exception as e:
                print(f"[InferenceManager] Lỗi xử lý frame: {e}")
                time.sleep(0.1)

    def get_annotated_frame(self):
        """Lấy bản sao của khung hình đã được vẽ kết quả phát hiện ngã (BBox + Skeleton).

        Returns:
            np.ndarray: Khung hình RGB đã được chú thích trực quan, hoặc khung hình gốc nếu chưa có kết quả.
        """
        with self.lock:
            if self.latest_annotated_frame is not None:
                return self.latest_annotated_frame.copy()
            return self.camera_manager.get_frame() # Fallback to raw frame
            
    def stop(self):
        """Dừng luồng xử lý suy diễn."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            
    def update_threshold(self, threshold):
        self.worker.conf_threshold = threshold
        
    def update_cooldown(self, cooldown):
        self.cooldown_time = float(cooldown)
        
    def pop_new_falls(self):
        """Cập nhật ngưỡng quyết định té ngã cho mô hình PoseBiGRU.

        Args:
            threshold: Ngưỡng xác suất (0.0 -> 1.0) để kích hoạt trạng thái ngã.
        """
        self.worker.conf_threshold = threshold
        
    def update_cooldown(self, cooldown):
        """Cập nhật thời gian cooldown của trạng thái ngã.

        Args:
            cooldown: Thời gian cooldown (giây).
        """
        self.cooldown_time = float(cooldown)
        
    def pop_new_falls(self):
        """Lấy và xóa danh sách các sự kiện té ngã mới phát hiện ra khỏi hàng đợi.

        Được UI gọi để phát tiếng beep cảnh báo hoặc lưu log duy nhất một lần khi bắt đầu ngã.

        Returns:
            list[int]: Danh sách các track_id vừa té ngã.
        """
        with self.lock:
            falls = self.new_falling_ids.copy()
            self.new_falling_ids.clear()
            return falls