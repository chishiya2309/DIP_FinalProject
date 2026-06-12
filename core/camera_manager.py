import cv2
import threading
from src.preprocessing import EnhancementConfigManager, enhance_frame

class CameraManager:
    def __init__(self, source=0):
        self.source = source
        self.cap = cv2.VideoCapture(self.source)
        self.running = False
        self.lock = threading.Lock()
        self.latest_frame = None
        
        # Khởi tạo bộ quản lý cấu hình và flag kích hoạt xử lý ảnh
        self.enhance_config = EnhancementConfigManager()
        self.enhance_enabled = True

    def start(self):
        if self.running:
            return
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.source)
            
        self.reset_calibration()
        self.running = True
        self.thread = threading.Thread(target=self._update_frames, daemon=True)
        self.thread.start()

    def reset_calibration(self):
        """Reset trạng thái hiệu chuẩn tự động khi đổi camera hoặc khởi động lại."""
        if hasattr(self.enhance_config, "calibration_history"):
            self.enhance_config.calibration_history = []
            self.enhance_config.is_calibrated = False

    def _update_frames(self):
        import time
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:
            fps = 30.0
        frame_delay = 1.0 / fps
        
        while self.running:
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                # Nếu là video file, loop lại
                if isinstance(self.source, str) and not self.source.startswith('rtsp'):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.reset_calibration()
                    continue
                else:
                    break
            
            # Áp dụng bộ lọc thích nghi tăng cường chất lượng video nếu được bật
            if self.enhance_enabled:
                try:
                    frame = enhance_frame(frame, self.enhance_config)
                except Exception as e:
                    print(f"[CameraManager] Lỗi xử lý tăng cường video: {e}")
            
            # Chuyển đổi màu từ BGR sang RGB cho Pillow
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            with self.lock:
                self.latest_frame = frame_rgb
                
            elapsed = time.time() - start_time
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)

    def get_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

    def change_source(self, new_source):
        self.stop()
        self.source = new_source
        self.cap = cv2.VideoCapture(self.source)
        self.reset_calibration()
        self.start()

