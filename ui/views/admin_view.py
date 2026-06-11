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
        self.opt_source = ctk.CTkOptionMenu(self.settings_frame, values=["Webcam (0)", "demo/demo.mp4", "RTSP Stream"])
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
        
        # DIP Video Enhancement Config
        self.lbl_dip = ctk.CTkLabel(self.settings_frame, text="TĂNG CƯỜNG HÌNH ẢNH (DIP)", font=ctk.CTkFont(size=14, weight="bold"))
        self.lbl_dip.pack(anchor="w", padx=20, pady=(15, 5))
        
        self.sw_enhance = ctk.CTkSwitch(self.settings_frame, text="Bật tăng cường video", command=self.toggle_enhancement)
        self.sw_enhance.select()
        self.sw_enhance.pack(anchor="w", padx=20, pady=5)
        
        self.sw_adaptive = ctk.CTkSwitch(self.settings_frame, text="Tự động thích nghi (Auto)", command=self.toggle_adaptive)
        self.sw_adaptive.select()
        self.sw_adaptive.pack(anchor="w", padx=20, pady=5)
        
        # Đồng bộ trạng thái ban đầu từ CameraManager
        if not self.controller.camera_manager.enhance_enabled:
            self.sw_enhance.deselect()
            self.sw_adaptive.configure(state="disabled")
        if not self.controller.camera_manager.enhance_config.auto_detect:
            self.sw_adaptive.deselect()
            
        # Action Buttons
        self.btn_apply = ctk.CTkButton(self.settings_frame, text="Áp dụng thay đổi", command=self.apply_changes)
        self.btn_apply.pack(fill="x", padx=20, pady=10, side="bottom")
        
        self.btn_export = ctk.CTkButton(self.settings_frame, text="Xuất Log CSV", fg_color="transparent", border_width=1)
        self.btn_export.pack(fill="x", padx=20, pady=10, side="bottom")

    def toggle_enhancement(self):
        enabled = self.sw_enhance.get() == 1
        self.controller.camera_manager.enhance_enabled = enabled
        if not enabled:
            self.sw_adaptive.configure(state="disabled")
        else:
            self.sw_adaptive.configure(state="normal")

    def toggle_adaptive(self):
        adaptive = self.sw_adaptive.get() == 1
        self.controller.camera_manager.enhance_config.auto_detect = adaptive
        if adaptive:
            self.controller.camera_manager.reset_calibration()

    def apply_changes(self):
        # Đọc nguồn camera mới
        source_val = self.opt_source.get()
        new_source = 0
        if "Webcam" in source_val:
            new_source = 0
        elif "demo" in source_val and ".mp4" in source_val:
            # Tim file tuong doi tu thu muc goc cua project
            import os
            demo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "demo", "demo.mp4")
            new_source = demo_path
            
        if self.controller.camera_manager.source != new_source:
            print(f"[AdminView] Đổi nguồn camera sang: {new_source}")
            self.controller.camera_manager.change_source(new_source)
            
        # Cập nhật cấu hình xử lý ảnh
        self.controller.camera_manager.enhance_enabled = self.sw_enhance.get() == 1
        self.controller.camera_manager.enhance_config.auto_detect = self.sw_adaptive.get() == 1
        self.controller.camera_manager.enhance_config.save()
        print("[AdminView] Đã lưu cấu hình tăng cường video.")

    def start_video(self):
        self.controller.camera_manager.start()
        self.video_panel.start()

    def stop_video(self):
        self.video_panel.stop()

