import cv2
import threading

class CameraManager:
    def __init__(self, source=0):
        self.source = source
        self.cap = cv2.VideoCapture(self.source)
        self.running = False
        self.lock = threading.Lock()
        self.latest_frame = None

    def start(self):
        if self.running:
            return
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.source)
            
        self.running = True
        self.thread = threading.Thread(target=self._update_frames, daemon=True)
        self.thread.start()

    def _update_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                # Nếu là video file, loop lại
                if isinstance(self.source, str) and not self.source.startswith('rtsp'):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            
            # Chuyển đổi màu từ BGR sang RGB cho Pillow
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            with self.lock:
                self.latest_frame = frame_rgb

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
        self.start()
