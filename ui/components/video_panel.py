import customtkinter as ctk
from PIL import Image

class VideoPanel(ctk.CTkLabel):
    def __init__(self, master, camera_manager=None, inference_manager=None, **kwargs):
        super().__init__(master, text="[Chưa Kết Nối Camera]", font=ctk.CTkFont(size=24), **kwargs)
        self.camera_manager = camera_manager
        self.inference_manager = inference_manager
        self.is_playing = False
        self._update_job = None

    def set_camera_manager(self, camera_manager):
        self.camera_manager = camera_manager
        
    def set_inference_manager(self, inference_manager):
        self.inference_manager = inference_manager

    def start(self):
        if self.is_playing:
            return # Tránh tạo nhiều vòng lặp
        self.is_playing = True
        self.update_frame()

    def stop(self):
        self.is_playing = False
        if self._update_job is not None:
            self.after_cancel(self._update_job)
            self._update_job = None
        self.configure(image=None, text="[Đã Dừng Camera]")

    def update_frame(self):
        if not self.is_playing or not self.camera_manager:
            return

        # Lấy frame đã chú thích nếu có inference_manager, ngược lại lấy frame gốc
        if self.inference_manager is not None:
            frame = self.inference_manager.get_annotated_frame()
        else:
            frame = self.camera_manager.get_frame()
            
        if frame is not None:
            # Xoá text hiển thị lúc chưa có video
            self.configure(text="")
            
            # Lấy kích thước hiện tại của panel
            width = self.winfo_width()
            height = self.winfo_height()
            
            # Tránh lỗi khi window chưa hiển thị (width/height quá nhỏ)
            if width > 10 and height > 10:
                # Resize frame để giữ tỷ lệ 16:9
                img = Image.fromarray(frame)
                
                # Tính toán kích thước mới dựa trên tỷ lệ
                img_ratio = img.width / img.height
                panel_ratio = width / height
                
                if panel_ratio > img_ratio:
                    new_height = height
                    new_width = int(new_height * img_ratio)
                else:
                    new_width = width
                    new_height = int(new_width / img_ratio)
                
                self.current_image = ctk.CTkImage(light_image=img, dark_image=img, size=(new_width, new_height))
                self.configure(image=self.current_image)

        # Cập nhật liên tục khoảng 30fps (~33ms)
        self._update_job = self.after(33, self.update_frame)
