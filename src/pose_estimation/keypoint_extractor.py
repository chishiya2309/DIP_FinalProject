"""Module ước lượng tư thế người (Pose Estimation) bằng YOLO11-Pose.

Wrapper class cho mô hình YOLO11-Pose pre-trained trên COCO dataset.
Sử dụng chiến lược Zero-shot (đóng băng toàn bộ weights) để trích xuất
17 COCO keypoints từ mỗi frame video.

Pipeline position: Preprocessing → [THIS] → Sliding Window
"""

import numpy as np
from ultralytics import YOLO


# COCO-17 Keypoints reference
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


class KeypointExtractor:
    """Wrapper cho YOLO11-Pose pre-trained.

    Chức năng:
      - Load model yolo11s-pose.pt (tự động download lần đầu)
      - Nhận frame đã qua tiền xử lý (BGR)
      - Trả về danh sách người phát hiện được, mỗi người gồm:
        + bbox: [x1, y1, x2, y2] và confidence
        + keypoints: mảng (17, 3) gồm (x, y, confidence) cho mỗi joint
      - Xử lý edge case: không phát hiện người → trả về list rỗng

    Attributes:
        model: Instance của YOLO model.
        conf_threshold: Ngưỡng confidence tối thiểu để giữ detection.
    """

    def __init__(self, model_name: str = "yolo11s-pose.pt", conf_threshold: float = 0.5, device: str = "cpu"):
        """Khởi tạo và load model YOLOv8-Pose.

        Args:
            model_name: Tên model (sẽ tự download nếu chưa có).
            conf_threshold: Ngưỡng confidence cho detection.
            device: Device để chạy inference ("cpu" hoặc "cuda").
        """
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.device = device

    def extract(self, frame: np.ndarray) -> list[dict]:
        """Trích xuất keypoints từ một frame.

        Args:
            frame: Ảnh BGR đã qua tiền xử lý (H, W, 3).

        Returns:
            List các dict, mỗi dict chứa thông tin một người:
            [
                {
                    "bbox": [x1, y1, x2, y2],   # float
                    "bbox_conf": 0.95,            # float
                    "keypoints": [[x, y, conf], ...] # 17 joints, mỗi joint 3 giá trị
                },
                ...
            ]
            Trả về list rỗng nếu không phát hiện được người nào.
        """
        results = self.model(frame, conf=self.conf_threshold, device=self.device, verbose=False)
        persons = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes
            keypoints_data = result.keypoints

            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                bbox_conf = float(boxes.conf[i].cpu().numpy())

                if keypoints_data is not None and keypoints_data.xy is not None:
                    kp_xy = keypoints_data.xy[i].cpu().numpy()
                    kp_conf = keypoints_data.conf[i].cpu().numpy()
                    kpts = np.column_stack([kp_xy, kp_conf]).tolist()
                else:
                    kpts = [[0.0, 0.0, 0.0]] * 17

                persons.append({
                    "bbox": [round(v, 2) for v in bbox],
                    "bbox_conf": round(bbox_conf, 4),
                    "keypoints": [[round(v, 2) for v in kp] for kp in kpts],
                })

        return persons
