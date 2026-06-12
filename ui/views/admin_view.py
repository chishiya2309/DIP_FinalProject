import customtkinter as ctk
from ui.components.video_panel import VideoPanel
from core.logger import fall_logger

class AdminView(ctk.CTkFrame):
    def __init__(self, master, controller, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.controller = controller
        
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        
        self.video_paths_map = {}
        
        # Top bar
        self.topbar = ctk.CTkFrame(self, height=50)
        self.topbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
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
        self.opt_source = ctk.CTkOptionMenu(self.settings_frame, values=["Webcam (0)", "demo/demo.mp4", "Chọn File Video..."], command=self.on_source_change, dynamic_resizing=False)
        self.opt_source.pack(fill="x", padx=20, pady=(0, 15))
        
        # Fall confidence threshold
        self.lbl_thresh = ctk.CTkLabel(self.settings_frame, text="Ngưỡng cảnh báo (Confidence): 0.50")
        self.lbl_thresh.pack(anchor="w", padx=20)
        self.sld_thresh = ctk.CTkSlider(self.settings_frame, from_=0.0, to=1.0, command=self.update_thresh_label)
        self.sld_thresh.set(0.50)
        self.sld_thresh.pack(fill="x", padx=20, pady=(0, 15))
        
        # Cooldown
        self.lbl_cd = ctk.CTkLabel(self.settings_frame, text="Thời gian Cooldown (giây):")
        self.lbl_cd.pack(anchor="w", padx=20)
        self.opt_cd = ctk.CTkOptionMenu(self.settings_frame, values=["3", "5", "10", "15"], command=self.update_cooldown, dynamic_resizing=False)
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
        
        self.btn_export = ctk.CTkButton(self.settings_frame, text="Xuất Log CSV", fg_color="transparent", border_width=1, command=self.export_csv)
        self.btn_export.pack(fill="x", padx=20, pady=10, side="bottom")

    def on_source_change(self, value):
        if value == "Chọn File Video...":
            from tkinter import filedialog
            import os
            filepath = filedialog.askopenfilename(
                title="Chọn Video",
                filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov"), ("All Files", "*.*")]
            )
            if filepath:
                basename = os.path.basename(filepath)
                self.video_paths_map[basename] = filepath
                
                # Thêm vào danh sách và chọn nó
                current_values = list(self.opt_source.cget("values"))
                if basename not in current_values:
                    current_values.insert(0, basename)
                    self.opt_source.configure(values=current_values)
                self.opt_source.set(basename)
            else:
                self.opt_source.set("Webcam (0)") # Reset nếu người dùng hủy

    def toggle_enhancement(self):
        enabled = self.sw_enhance.get() == 1
        self.controller.camera_manager.enhance_enabled = enabled
        if not enabled:
            self.sw_adaptive.configure(state="disabled")
        else:
            self.sw_adaptive.configure(state="normal")

    def update_thresh_label(self, value):
        self.lbl_thresh.configure(text=f"Ngưỡng cảnh báo (Confidence): {value:.2f}")

    def update_cooldown(self, value):
        self.controller.inference_manager.update_cooldown(value)

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
        elif source_val == "demo/demo.mp4":
            import os
            demo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "demo", "demo.mp4")
            new_source = demo_path
        elif source_val in self.video_paths_map:
            new_source = self.video_paths_map[source_val]
        elif source_val != "Chọn File Video...":
            new_source = source_val
            
        if self.controller.camera_manager.source != new_source:
            print(f"[AdminView] Đổi nguồn camera sang: {new_source}")
            self.controller.inference_manager.stop()
            self.controller.camera_manager.change_source(new_source)
            self.controller.inference_manager.start()
            
        # Cập nhật cấu hình xử lý ảnh
        self.controller.camera_manager.enhance_enabled = self.sw_enhance.get() == 1
        self.controller.camera_manager.enhance_config.auto_detect = self.sw_adaptive.get() == 1
        self.controller.camera_manager.enhance_config.save()
        
        # Cập nhật threshold cho inference
        thresh = self.sld_thresh.get()
        self.controller.inference_manager.update_threshold(thresh)
        
        print("[AdminView] Đã lưu cấu hình.")

    def export_csv(self):
        import shutil
        import os
        from tkinter import filedialog, messagebox
        
        if not os.path.exists("fall_history.csv"):
            messagebox.showinfo("Thông báo", "Chưa có dữ liệu lịch sử té ngã.")
            return
            
        dest_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Lưu file lịch sử",
            initialfile="fall_history_export.csv"
        )
        
        if dest_path:
            try:
                shutil.copy("fall_history.csv", dest_path)
                messagebox.showinfo("Thành công", f"Đã xuất file thành công tới:\n{dest_path}")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể xuất file: {e}")

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
            
            if self.sw_log.get() == 1:
                source_str = str(self.controller.camera_manager.source)
                for tid in new_falls:
                    fall_logger.log_fall(source_str, tid)
            
        if has_fall:
            self.lbl_status.configure(text=f"⚠ PHÁT HIỆN TÉ NGÃ (ID: {falling_ids}) ⚠", text_color="red")
        else:
            self.lbl_status.configure(text="HỆ THỐNG AN TOÀN", text_color="#2ecc71")
            
        self.after(500, self._check_fall_status)

