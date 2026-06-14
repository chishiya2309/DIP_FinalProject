import csv
import threading
import queue
import time
import os
from datetime import datetime
import cv2
import numpy as np

class FallLogger:
    def __init__(self, log_file="fall_history.csv"):
        self.log_file = log_file
        self.log_queue = queue.Queue()
        self.running = False
        self.thread = None
        
        # Tạo thư mục lưu ảnh
        self.snapshots_dir = "logs/snapshots"
        os.makedirs(self.snapshots_dir, exist_ok=True)
        
        # Tạo file và header nếu chưa có
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Camera_Source", "Track_ID", "Status", "Snapshot_Path"])

    def start(self):
        if self.running: return
        self.running = True
        self.thread = threading.Thread(target=self._log_worker, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)

    def log_fall(self, source, track_id, frame_image=None, status="DETECTED"):
        # Put vào queue thay vì ghi trực tiếp để tránh block UI
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        snapshot_path = ""
        if frame_image is not None:
            snapshot_filename = f"fall_cam_{timestamp_file}_id{track_id}.jpg"
            snapshot_path = os.path.join(self.snapshots_dir, snapshot_filename)
            # frame_image từ UI thường là RGB, chuyển lại BGR để cv2 lưu đúng màu
            bgr_frame = cv2.cvtColor(frame_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(snapshot_path, bgr_frame)
            
            # Gửi cảnh báo qua Telegram (chạy thread ẩn bên dưới)
            from core.telegram_notifier import send_telegram_alert
            msg = f"🚨 PHÁT HIỆN TÉ NGÃ 🚨\n- Camera: {source}\n- Thời gian: {timestamp_str}\n- ID Nạn nhân: {track_id}"
            send_telegram_alert(msg, snapshot_path)
            
        self.log_queue.put([timestamp_str, source, track_id, status, snapshot_path])

    def _log_worker(self):
        while self.running:
            try:
                # Chờ lấy data từ queue (timeout 1s để check vòng lặp running)
                row = self.log_queue.get(timeout=1.0)
                
                # Ghi tuần tự, an toàn
                with open(self.log_file, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                    
                self.log_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                print(f"[FallLogger] Lỗi ghi file log: {e}")
                time.sleep(1.0)

# Khởi tạo logger dùng chung
fall_logger = FallLogger()
fall_logger.start()
