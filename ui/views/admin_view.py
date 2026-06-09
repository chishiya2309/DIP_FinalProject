import customtkinter as ctk
from ui.components.video_panel import VideoPanel

class AdminView(ctk.CTkFrame):
    def __init__(self, master, controller, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.controller = controller
        
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        
        # Top bar
        self.topbar = ctk.CTkFrame(self, height=50)
        self.topbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        self.lbl_role = ctk.CTkLabel(self.topbar, text="Role: ADMIN | FPS: -- | Pipeline: YOLOv8-Pose + PoseC3D", font=ctk.CTkFont(weight="bold"))
        self.lbl_role.pack(side="left", padx=20, pady=10)
        
        # Video Area
        self.video_frame = ctk.CTkFrame(self)
        self.video_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 5), pady=(0, 10))
        
        self.video_panel = VideoPanel(self.video_frame, camera_manager=controller.camera_manager)
        self.video_panel.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Status Area (dưới video)
        self.status_frame = ctk.CTkFrame(self, height=100)
        self.status_frame.grid(row=2, column=0, sticky="ew", padx=(0, 5))
        self.status_frame.grid_columnconfigure(0, weight=1)
        
        self.lbl_status = ctk.CTkLabel(self.status_frame, text="HỆ THỐNG AN TOÀN", 
                                       font=ctk.CTkFont(size=28, weight="bold"), text_color="#2ecc71")
        self.lbl_status.grid(row=0, column=0, pady=20)
        
        # Settings / Right Panel
        self.settings_frame = ctk.CTkFrame(self)
        self.settings_frame.grid(row=1, column=1, rowspan=2, sticky="nsew", padx=(5, 0))
        
        self.lbl_settings = ctk.CTkLabel(self.settings_frame, text="CÀI ĐẶT THÔNG SỐ", font=ctk.CTkFont(size=18, weight="bold"))
        self.lbl_settings.pack(pady=20)
        
        # Nguồn Camera
        self.lbl_source = ctk.CTkLabel(self.settings_frame, text="Nguồn Camera:")
        self.lbl_source.pack(anchor="w", padx=20)
        self.opt_source = ctk.CTkOptionMenu(self.settings_frame, values=["Webcam (0)", "demo.mp4", "RTSP Stream"])
        self.opt_source.pack(fill="x", padx=20, pady=(0, 15))
        
        # Fall confidence threshold
        self.lbl_thresh = ctk.CTkLabel(self.settings_frame, text="Ngưỡng cảnh báo (Confidence): 0.75")
        self.lbl_thresh.pack(anchor="w", padx=20)
        self.sld_thresh = ctk.CTkSlider(self.settings_frame, from_=0.0, to=1.0)
        self.sld_thresh.set(0.75)
        self.sld_thresh.pack(fill="x", padx=20, pady=(0, 15))
        
        # Cooldown
        self.lbl_cd = ctk.CTkLabel(self.settings_frame, text="Thời gian Cooldown (giây):")
        self.lbl_cd.pack(anchor="w", padx=20)
        self.opt_cd = ctk.CTkOptionMenu(self.settings_frame, values=["3", "5", "10", "15"])
        self.opt_cd.pack(fill="x", padx=20, pady=(0, 15))
        
        # Save log
        self.sw_log = ctk.CTkSwitch(self.settings_frame, text="Lưu lịch sử cảnh báo (CSV)")
        self.sw_log.select()
        self.sw_log.pack(anchor="w", padx=20, pady=15)
        
        # Action Buttons
        self.btn_apply = ctk.CTkButton(self.settings_frame, text="Áp dụng thay đổi")
        self.btn_apply.pack(fill="x", padx=20, pady=10, side="bottom")
        
        self.btn_export = ctk.CTkButton(self.settings_frame, text="Xuất Log CSV", fg_color="transparent", border_width=1)
        self.btn_export.pack(fill="x", padx=20, pady=10, side="bottom")

    def start_video(self):
        self.controller.camera_manager.start()
        self.video_panel.start()

    def stop_video(self):
        self.video_panel.stop()
