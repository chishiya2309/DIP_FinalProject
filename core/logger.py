import csv
import threading
import queue
import time
import os
from datetime import datetime

class FallLogger:
    def __init__(self, log_file="fall_history.csv"):
        self.log_file = log_file
        self.log_queue = queue.Queue()
        self.running = False
        self.thread = None
        
        # Tạo file và header nếu chưa có
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Camera_Source", "Track_ID", "Status"])

    def start(self):
        if self.running: return
        self.running = True
        self.thread = threading.Thread(target=self._log_worker, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)

    def log_fall(self, source, track_id, status="DETECTED"):
        # Put vào queue thay vì ghi trực tiếp để tránh block UI
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_queue.put([timestamp, source, track_id, status])

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
