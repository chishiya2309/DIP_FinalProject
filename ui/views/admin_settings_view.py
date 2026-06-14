import customtkinter as ctk
import os
from tkinter import filedialog, messagebox
import shutil

class AdminSettingsView(ctk.CTkFrame):
    def __init__(self, master, controller, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.controller = controller
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        self.video_paths_map = {}
        
        # Wrapper frame để căn giữa
        self.wrapper = ctk.CTkFrame(self, fg_color="transparent")
        self.wrapper.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        self.wrapper.grid_columnconfigure(0, weight=1)
        
        # Settings Panel
        self.settings_frame = ctk.CTkScrollableFrame(self.wrapper, corner_radius=15)
        self.settings_frame.pack(pady=20, padx=150, fill="both", expand=True)
        
        self.lbl_title = ctk.CTkLabel(self.settings_frame, text="CÀI ĐẶT HỆ THỐNG", font=ctk.CTkFont(size=24, weight="bold"))
        self.lbl_title.pack(pady=(20, 40))
        
        # Nguồn Camera
        self.lbl_source = ctk.CTkLabel(self.settings_frame, text="Nguồn Camera:", font=ctk.CTkFont(weight="bold"))
        self.lbl_source.pack(anchor="w", padx=40)
        
        # Thêm lựa chọn IP Camera
        self.opt_source = ctk.CTkOptionMenu(self.settings_frame, 
                                            values=["Webcam (0)", "demo/demo.mp4", "Chọn File Video...", "Nhập URL Camera IP..."], 
                                            command=self.on_source_change, dynamic_resizing=False, width=300)
        self.opt_source.pack(fill="x", padx=40, pady=(0, 20))
        
        # URL IP Camera Entry (ẩn mặc định)
        self.entry_ip_cam = ctk.CTkEntry(self.settings_frame, placeholder_text="rtsp://192.168.1.100:554/stream", width=300)
        
        # Fall confidence threshold
        self.lbl_thresh = ctk.CTkLabel(self.settings_frame, text="Ngưỡng cảnh báo (Confidence): 0.50", font=ctk.CTkFont(weight="bold"))
        self.lbl_thresh.pack(anchor="w", padx=40)
        self.sld_thresh = ctk.CTkSlider(self.settings_frame, from_=0.0, to=1.0, command=self.update_thresh_label)
        self.sld_thresh.set(0.50)
        self.sld_thresh.pack(fill="x", padx=40, pady=(0, 20))
        
        # Cooldown
        self.lbl_cd = ctk.CTkLabel(self.settings_frame, text="Thời gian Cooldown (giây):", font=ctk.CTkFont(weight="bold"))
        self.lbl_cd.pack(anchor="w", padx=40)
        self.opt_cd = ctk.CTkOptionMenu(self.settings_frame, values=["3", "5", "10", "15"], command=self.update_cooldown, dynamic_resizing=False)
        self.opt_cd.pack(fill="x", padx=40, pady=(0, 20))
        
        # Save log
        self.sw_log = ctk.CTkSwitch(self.settings_frame, text="Lưu lịch sử cảnh báo (CSV & Ảnh)")
        self.sw_log.select()
        self.sw_log.pack(anchor="w", padx=40, pady=20)
        
        # DIP Video Enhancement Config
        self.lbl_dip = ctk.CTkLabel(self.settings_frame, text="TĂNG CƯỜNG HÌNH ẢNH (DIP)", font=ctk.CTkFont(size=16, weight="bold"))
        self.lbl_dip.pack(anchor="w", padx=40, pady=(20, 10))
        
        self.sw_enhance = ctk.CTkSwitch(self.settings_frame, text="Bật tăng cường video", command=self.toggle_enhancement)
        self.sw_enhance.select()
        self.sw_enhance.pack(anchor="w", padx=40, pady=10)
        
        self.sw_adaptive = ctk.CTkSwitch(self.settings_frame, text="Tự động thích nghi (Auto)", command=self.toggle_adaptive)
        self.sw_adaptive.select()
        self.sw_adaptive.pack(anchor="w", padx=40, pady=10)
        
        # Đồng bộ trạng thái ban đầu từ CameraManager
        if not self.controller.camera_manager.enhance_enabled:
            self.sw_enhance.deselect()
            self.sw_adaptive.configure(state="disabled")
        if not self.controller.camera_manager.enhance_config.auto_detect:
            self.sw_adaptive.deselect()
            
        # Telegram Alerts
        self.lbl_tele = ctk.CTkLabel(self.settings_frame, text="CẢNH BÁO QUA TELEGRAM", font=ctk.CTkFont(size=16, weight="bold"))
        self.lbl_tele.pack(anchor="w", padx=40, pady=(20, 10))
        
        from core.telegram_notifier import load_telegram_config, save_telegram_config
        self.tele_config = load_telegram_config()
        
        self.sw_tele = ctk.CTkSwitch(self.settings_frame, text="Bật gửi cảnh báo qua Telegram", command=self.toggle_telegram)
        if self.tele_config.get("enabled"): self.sw_tele.select()
        else: self.sw_tele.deselect()
        self.sw_tele.pack(anchor="w", padx=40, pady=10)
        
        self.entry_tele_token = ctk.CTkEntry(self.settings_frame, placeholder_text="Nhập Bot Token...", width=300)
        self.entry_tele_token.insert(0, self.tele_config.get("bot_token", ""))
        
        self.entry_tele_chat = ctk.CTkEntry(self.settings_frame, placeholder_text="Nhập Chat ID...", width=300)
        self.entry_tele_chat.insert(0, self.tele_config.get("chat_id", ""))
        
        if self.tele_config.get("enabled"):
            self.entry_tele_token.pack(fill="x", padx=40, pady=(5, 5))
            self.entry_tele_chat.pack(fill="x", padx=40, pady=(0, 15))
            
        # Action Buttons
        self.btn_apply = ctk.CTkButton(self.settings_frame, text="Lưu cấu hình", font=ctk.CTkFont(weight="bold"), height=40, command=self.apply_changes)
        self.btn_apply.pack(fill="x", padx=40, pady=(40, 10))
        
        self.btn_export = ctk.CTkButton(self.settings_frame, text="Xuất Log CSV", fg_color="transparent", border_width=2, height=40, command=self.export_csv)
        self.btn_export.pack(fill="x", padx=40, pady=10)

    def on_source_change(self, value):
        if value == "Chọn File Video...":
            self.entry_ip_cam.pack_forget()
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
        elif value == "Nhập URL Camera IP...":
            self.entry_ip_cam.pack(after=self.opt_source, fill="x", padx=40, pady=(5, 15))
        else:
            self.entry_ip_cam.pack_forget()

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

    def toggle_telegram(self):
        if self.sw_tele.get() == 1:
            # Hiện ô nhập liệu (sẽ nằm trên btn_apply)
            # Tốt nhất là dùng pack(after=self.sw_tele)
            self.entry_tele_token.pack(after=self.sw_tele, fill="x", padx=40, pady=(5, 5))
            self.entry_tele_chat.pack(after=self.entry_tele_token, fill="x", padx=40, pady=(0, 15))
        else:
            self.entry_tele_token.pack_forget()
            self.entry_tele_chat.pack_forget()

    def apply_changes(self):
        # Đọc nguồn camera mới
        source_val = self.opt_source.get()
        new_source = 0
        if "Webcam" in source_val:
            new_source = 0
        elif source_val == "demo/demo.mp4":
            demo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "demo", "demo.mp4")
            new_source = demo_path
        elif source_val in self.video_paths_map:
            new_source = self.video_paths_map[source_val]
        elif source_val == "Nhập URL Camera IP...":
            ip_url = self.entry_ip_cam.get().strip()
            if ip_url:
                new_source = ip_url
            else:
                messagebox.showerror("Lỗi", "Vui lòng nhập URL hợp lệ!")
                return
        elif source_val != "Chọn File Video...":
            new_source = source_val
            
        if self.controller.camera_manager.source != new_source:
            print(f"[SettingsView] Đổi nguồn camera sang: {new_source}")
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
        
        # Cập nhật Telegram
        self.tele_config["enabled"] = self.sw_tele.get() == 1
        self.tele_config["bot_token"] = self.entry_tele_token.get().strip()
        self.tele_config["chat_id"] = self.entry_tele_chat.get().strip()
        from core.telegram_notifier import save_telegram_config
        save_telegram_config(self.tele_config)
        
        print("[SettingsView] Đã lưu cấu hình.")
        messagebox.showinfo("Thành công", "Đã áp dụng cấu hình hệ thống!")

    def export_csv(self):
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
