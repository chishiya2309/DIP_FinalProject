import customtkinter as ctk
from ui.components.video_panel import VideoPanel
from core.logger import fall_logger

class ClientView(ctk.CTkFrame):
    """Màn hình giám sát dành cho người dùng thông thường (Client View - User mode).

    Hiển thị camera giám sát nhưng ẩn đi các cài đặt cấu hình và thông tin chi tiết
    các xương khớp ID cụ thể để giao diện đơn giản nhất, chỉ tập trung vào hiển thị
    trạng thái AN TOÀN hoặc CẢNH BÁO TÉ NGÃ.
    """

    def __init__(self, master, controller, **kwargs):
        """Khởi tạo màn hình giám sát rút gọn dành cho User.

        Args:
            master: Widget cha chứa view này.
            controller: Bộ điều khiển chính MainApp.
            **kwargs: Các đối số bổ sung chuyển sang cho CTkFrame.
        """
        super().__init__(master, fg_color="transparent", **kwargs)
        self.controller = controller
        self.video_panel = None  # Khởi tạo lazy cùng với InferenceManager

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Top bar
        self.topbar = ctk.CTkFrame(self, height=50)
        self.topbar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.lbl_role = ctk.CTkLabel(self.topbar, text="Role: USER", font=ctk.CTkFont(weight="bold"))
        self.lbl_role.pack(side="left", padx=20, pady=10)

        # Video Area
        self.video_frame = ctk.CTkFrame(self)
        self.video_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)

        # Loading overlay
        self.loading_label = ctk.CTkLabel(
            self.video_frame,
            text="⏳ Đang tải model AI...\n(YOLO + PoseBiGRU)",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#f0a500",
        )
        self.loading_label.grid(row=0, column=0)

        # Status Area
        self.status_frame = ctk.CTkFrame(self, height=120)
        self.status_frame.grid(row=2, column=0, sticky="ew")
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
        self.loading_label.grid(row=0, column=0)
        if self.video_panel is not None:
            self.video_panel.stop()

        self.controller.request_inference_manager(self._on_model_ready)

    def _on_model_ready(self, inference_manager):
        """Callback sau khi InferenceManager đã load xong model."""
        self.loading_label.grid_forget()

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
        """Dừng cập nhật khung hình camera trên giao diện GUI."""
        if self.video_panel is not None:
            self.video_panel.stop()

    def _check_fall_status(self):
        """Chu kỳ thăm dò (polling) kiểm tra trạng thái té ngã, tương tự AdminView nhưng rút gọn."""
        if not self.winfo_ismapped():
            self.after(500, self._check_fall_status)
            return

        mgr = self.controller.inference_manager
        if mgr is None:
            self.after(500, self._check_fall_status)
            return

        with mgr.lock:
            has_fall = mgr.has_fall

        new_falls = mgr.pop_new_falls()
        if new_falls:
            import winsound
            import threading
            threading.Thread(target=lambda: winsound.Beep(1000, 500), daemon=True).start()

            source_str = str(self.controller.camera_manager.source)
            frame = mgr.get_annotated_frame()
            for tid in new_falls:
                fall_logger.log_fall(source_str, tid, frame_image=frame)

        if has_fall:
            self.lbl_status.configure(text="⚠ PHÁT HIỆN TÉ NGÃ ⚠", text_color="red")
        else:
            self.lbl_status.configure(text="HỆ THỐNG AN TOÀN", text_color="#2ecc71")

        self.after(500, self._check_fall_status)
