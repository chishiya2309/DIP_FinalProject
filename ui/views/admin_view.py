import customtkinter as ctk
from ui.components.video_panel import VideoPanel
from core.logger import fall_logger

class AdminView(ctk.CTkFrame):
    """Màn hình giám sát của Admin (Admin Monitor View).

    Hiển thị luồng video trực tiếp từ camera, vẽ khung xương người di chuyển,
    và cập nhật banner cảnh báo trạng thái té ngã theo thời gian thực (phát âm thanh beep,
    lưu vết lịch sử té ngã khi phát hiện sự cố mới).
    """

    def __init__(self, master, controller, **kwargs):
        """Khởi tạo giao diện màn hình giám sát dành cho quản trị viên.

        Args:
            master: Widget cha chứa view này (thường là view_container của MainApp).
            controller: Bộ điều khiển chính MainApp để liên kết camera và mô hình suy diễn.
            **kwargs: Các đối số bổ sung chuyển qua cho CTkFrame.
        """
        super().__init__(master, fg_color="transparent", **kwargs)
        self.controller = controller
        self.video_panel = None  # Khởi tạo lazy cùng với InferenceManager

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Top bar
        self.topbar = ctk.CTkFrame(self, height=50)
        self.topbar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.lbl_role = ctk.CTkLabel(self.topbar, text="Role: ADMIN | FPS: -- | Pipeline: YOLO11-Pose + PoseBiGRU", font=ctk.CTkFont(weight="bold"))

        self.lbl_role.pack(side="left", padx=20, pady=10)

        # Video Area
        self.video_frame = ctk.CTkFrame(self)
        self.video_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 5), pady=(0, 10))
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)

        # Loading overlay (hiển thị khi model đang load)
        self.loading_label = ctk.CTkLabel(
            self.video_frame,
            text="⏳ Đang tải model AI...\n(YOLO + PoseBiGRU)",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#f0a500",
        )
        self.loading_label.grid(row=0, column=0)

        # Status Area (dưới video)
        self.status_frame = ctk.CTkFrame(self, height=100)
        self.status_frame.grid(row=2, column=0, sticky="ew", padx=(0, 5))
        self.status_frame.grid_columnconfigure(0, weight=1)

        self.lbl_status = ctk.CTkLabel(
            self.status_frame,
            text="HỆ THỐNG AN TOÀN",
            font=ctk.CTkFont(size=36, weight="bold"),
            text_color="#2ecc71",
        )
        self.lbl_status.grid(row=0, column=0, pady=20)

    def start_video(self):
        """Yêu cầu InferenceManager (lazy-load) rồi khởi động camera và inference."""
        # Hiện loading label trong lúc chờ model load
        self.loading_label.grid(row=0, column=0)
        if self.video_panel is not None:
            self.video_panel.stop()

        self.controller.request_inference_manager(self._on_model_ready)

    def _on_model_ready(self, inference_manager):
        """Callback sau khi InferenceManager đã load xong model."""
        # Ẩn loading label
        self.loading_label.grid_forget()

        # Tạo VideoPanel nếu chưa có
        if self.video_panel is None:
            self.video_panel = VideoPanel(
                self.video_frame,
                camera_manager=self.controller.camera_manager,
                inference_manager=inference_manager,
            )
            self.video_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.controller.camera_manager.start()
        inference_manager.start()
        self.video_panel.start()
        self._check_fall_status()

    def stop_video(self):
        """Dừng việc cập nhật hình ảnh camera trên giao diện GUI."""
        if self.video_panel is not None:
            self.video_panel.stop()

    def _check_fall_status(self):
        """Chu kỳ thăm dò (polling) kiểm tra trạng thái ngã của người giám sát.

        Nếu phát hiện té ngã, cập nhật chữ cảnh báo màu đỏ, phát âm thanh cảnh báo (Beep)
        và ghi nhận sự kiện ngã này vào file nhật ký lịch sử cùng ảnh chụp lúc ngã.
        """
        if not self.winfo_ismapped():
            self.after(500, self._check_fall_status)
            return

        mgr = self.controller.inference_manager
        if mgr is None:
            self.after(500, self._check_fall_status)
            return

        with mgr.lock:
            has_fall = mgr.has_fall
            falling_ids = mgr.falling_track_ids

        new_falls = mgr.pop_new_falls()
        if new_falls:
            import winsound
            import threading
            threading.Thread(target=lambda: winsound.Beep(1000, 500), daemon=True).start()
            print(f"[ALERT] Phát hiện ngã mới từ các ID: {new_falls}")

            # Log history
            source_str = str(self.controller.camera_manager.source)
            frame = mgr.get_annotated_frame()
            for tid in new_falls:
                fall_logger.log_fall(source_str, tid, frame_image=frame)

        if has_fall:
            self.lbl_status.configure(
                text=f"⚠ PHÁT HIỆN TÉ NGÃ (ID: {falling_ids}) ⚠", text_color="red"
            )
        else:
            self.lbl_status.configure(text="HỆ THỐNG AN TOÀN", text_color="#2ecc71")

        self.after(500, self._check_fall_status)
