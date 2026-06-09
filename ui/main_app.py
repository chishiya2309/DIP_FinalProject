import customtkinter as ctk
from ui.views.login_view import LoginView
from ui.views.admin_view import AdminView
from ui.views.client_view import ClientView
from ui.components.sidebar import Sidebar
from core.camera_manager import CameraManager

class MainApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Hệ thống cảnh báo té ngã tự động cho người cao tuổi")
        self.geometry("1280x720")
        self.minsize(1000, 600)
        
        # Cấu hình grid cho cửa sổ chính: 1 hàng, 2 cột (Sidebar và Main Content)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        # Sidebar (Banner) luôn cố định bên trái
        self.sidebar = Sidebar(self)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        # Container cho các view bên phải
        self.view_container = ctk.CTkFrame(self, fg_color="transparent")
        self.view_container.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.view_container.grid_rowconfigure(0, weight=1)
        self.view_container.grid_columnconfigure(0, weight=1)
        
        self.camera_manager = CameraManager(source=0)
        
        self.views = {}
        
        self.setup_views()
        self.show_view("login")
        
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        self.camera_manager.stop()
        self.destroy()

    def setup_views(self):
        # Khởi tạo các view
        self.views["login"] = LoginView(self.view_container, controller=self)
        self.views["admin"] = AdminView(self.view_container, controller=self)
        self.views["client"] = ClientView(self.view_container, controller=self)
        
        for view in self.views.values():
            view.grid(row=0, column=0, sticky="nsew")

    def show_view(self, view_name):
        # Dừng camera của view hiện tại nếu có
        for v in self.views.values():
            if hasattr(v, 'stop_video'):
                v.stop_video()
                
        # Nâng view cần hiển thị lên trên
        view = self.views[view_name]
        view.tkraise()
        
        # Bật camera nếu là màn hình giám sát
        if hasattr(view, 'start_video'):
            view.start_video()
        
        # Cập nhật sidebar tuỳ theo chế độ
        if view_name == "login":
            self.sidebar.hide_navigation()
        elif view_name == "admin":
            self.sidebar.show_admin_navigation()
        elif view_name == "client":
            self.sidebar.show_client_navigation()

    def logout(self):
        self.show_view("login")
