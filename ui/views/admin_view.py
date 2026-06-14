import customtkinter as ctk
from ui.components.video_panel import VideoPanel
from core.logger import fall_logger

class AdminView(ctk.CTkFrame):
    def __init__(self, master, controller, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.controller = controller
        
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Top bar
        self.topbar = ctk.CTkFrame(self, height=50)
        self.topbar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.lbl_role = ctk.CTkLabel(self.topbar, text="Role: ADMIN | FPS: -- | Pipeline: YOLOv8-Pose + PoseC3D", font=ctk.CTkFont(weight="bold"))
        self.lbl_role.pack(side="left", padx=20, pady=10)
        
        # Video Area
        self.video_frame = ctk.CTkFrame(self)
        self.video_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 5), pady=(0, 10))
        
        self.video_panel = VideoPanel(self.video_frame, camera_manager=controller.camera_manager, inference_manager=controller.inference_manager)
        self.video_panel.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Status Area (dưới video)
        self.status_frame = ctk.CTkFrame(self, height=100)
        self.status_frame.grid(row=2, column=0, sticky="ew", padx=(0, 5))
        self.status_frame.grid_columnconfigure(0, weight=1)
        
        self.lbl_status = ctk.CTkLabel(self.status_frame, text="HỆ THỐNG AN TOÀN", 
                                       font=ctk.CTkFont(size=36, weight="bold"), text_color="#2ecc71")
        self.lbl_status.grid(row=0, column=0, pady=20)

    def start_video(self):
        self.controller.camera_manager.start()
        self.controller.inference_manager.start()
        self.video_panel.start()
        self._check_fall_status()

    def stop_video(self):
        self.video_panel.stop()

    def _check_fall_status(self):
        if not self.winfo_ismapped():
            self.after(500, self._check_fall_status)
            return
            
        with self.controller.inference_manager.lock:
            has_fall = self.controller.inference_manager.has_fall
            falling_ids = self.controller.inference_manager.falling_track_ids
            
        new_falls = self.controller.inference_manager.pop_new_falls()
        if new_falls:
            import winsound
            import threading
            threading.Thread(target=lambda: winsound.Beep(1000, 500), daemon=True).start()
            print(f"[ALERT] Phát hiện ngã mới từ các ID: {new_falls}")
            
            # Log history
            source_str = str(self.controller.camera_manager.source)
            frame = self.controller.inference_manager.get_annotated_frame()
            for tid in new_falls:
                fall_logger.log_fall(source_str, tid, frame_image=frame)
            
        if has_fall:
            self.lbl_status.configure(text=f"⚠ PHÁT HIỆN TÉ NGÃ (ID: {falling_ids}) ⚠", text_color="red")
        else:
            self.lbl_status.configure(text="HỆ THỐNG AN TOÀN", text_color="#2ecc71")
            
        self.after(500, self._check_fall_status)

