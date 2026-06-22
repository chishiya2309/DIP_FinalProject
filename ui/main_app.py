import time
import threading
import customtkinter as ctk
from ui.views.login_view import LoginView
from ui.views.admin_view import AdminView
from ui.views.admin_settings_view import AdminSettingsView
from ui.views.client_view import ClientView
from ui.views.history_view import HistoryView
from ui.components.sidebar import Sidebar
from core.camera_manager import CameraManager
from core.inference_manager import InferenceManager

class MainApp(ctk.CTk):
    """Lớp ứng dụng chính (Main Application Window) của hệ thống cảnh báo té ngã.

    Lớp này kế thừa từ customtkinter.CTk, chịu trách nhiệm thiết lập cửa sổ chính,
    khởi tạo CameraManager và InferenceManager để xử lý video, quản lý định tuyến
    (routing) giữa các giao diện con (Login, Admin Monitor, Admin Settings, Client View, History).
    """
    def __init__(self):
        """Khởi tạo cửa sổ ứng dụng, cấu hình bố cục lưới, sidebar và nạp các view."""
        # Task 0: đo startup time (trước khi làm bất cứ thứ gì)
        _t0 = time.perf_counter()

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

        # Task 1: Lazy-load InferenceManager.
        # Chỉ tạo CameraManager (nhẹ, không load model) ở đây.
        # InferenceManager (nặng: YOLO + BiGRU) chỉ được tạo khi user vào màn hình giám sát.
        self.camera_manager = CameraManager(source=0)
        self._inference_manager: InferenceManager | None = None  # Lazy
        self._inference_loading = False          # Cờ: đang load trong thread ngầm
        self._inference_load_callbacks: list = []  # Hàng đợi callback chờ load xong

        self.views = {}
        self.current_role = None

        self.setup_views()
        self.show_view("login")

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Task 0: in startup time
        startup_ms = (time.perf_counter() - _t0) * 1000
        print(f"[PERF] startup_ms={startup_ms:.1f}  (login screen visible)")

    # ------------------------------------------------------------------
    # Task 1: Lazy-load API
    # ------------------------------------------------------------------

    @property
    def inference_manager(self) -> "InferenceManager | None":
        """Trả về InferenceManager hiện tại (None nếu chưa load xong)."""
        return self._inference_manager

    def request_inference_manager(self, on_ready_callback):
        """Yêu cầu InferenceManager; nếu chưa có thì load trong background thread.

        Args:
            on_ready_callback: Hàm nhận 1 tham số (InferenceManager) được gọi
                               trên main thread khi model đã sẵn sàng.
        """
        if self._inference_manager is not None:
            # Đã sẵn sàng → gọi ngay
            on_ready_callback(self._inference_manager)
            return

        self._inference_load_callbacks.append(on_ready_callback)

        if self._inference_loading:
            return  # Thread đang chạy, callback sẽ được gọi sau khi xong

        self._inference_loading = True

        def _load():
            mgr = InferenceManager(self.camera_manager)
            # Trả về main thread qua after() để an toàn với Tkinter
            self.after(0, lambda: self._on_inference_ready(mgr))

        threading.Thread(target=_load, daemon=True).start()

    def _on_inference_ready(self, mgr: "InferenceManager"):
        """Callback (main thread) khi InferenceManager đã load xong."""
        self._inference_manager = mgr
        self._inference_loading = False
        print("[PERF] InferenceManager ready (YOLO + BiGRU loaded)")
        for cb in self._inference_load_callbacks:
            cb(mgr)
        self._inference_load_callbacks.clear()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_closing(self):
        """Xử lý sự kiện khi người dùng đóng ứng dụng.

        Dừng luồng suy diễn, giải phóng camera và phá hủy cửa sổ.
        """
        if self._inference_manager is not None:
            self._inference_manager.stop()
        self.camera_manager.stop()
        self.destroy()

    def setup_views(self):
        """Khởi tạo toàn bộ các màn hình giao diện con và xếp chồng chúng lên nhau."""
        self.views["login"] = LoginView(self.view_container, controller=self)
        self.views["admin_monitor"] = AdminView(self.view_container, controller=self)
        self.views["admin_settings"] = AdminSettingsView(self.view_container, controller=self)
        self.views["client"] = ClientView(self.view_container, controller=self)
        self.views["history"] = HistoryView(self.view_container, controller=self)

        for view in self.views.values():
            view.grid(row=0, column=0, sticky="nsew")

    def show_view(self, view_name):
        """Chuyển đổi màn hình hiển thị sang view được chỉ định.

        Quản lý việc bật/tắt camera tương ứng với view được hiển thị,
        cập nhật thanh menu điều hướng Sidebar và tự động nạp lại dữ liệu
        nếu chuyển sang màn hình lịch sử.

        Args:
            view_name: Tên của view cần hiển thị (ví dụ: 'login', 'admin_monitor', 'history').
        """
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
            self.current_role = None
            self.sidebar.hide_navigation()
        elif view_name.startswith("admin"):
            self.current_role = "admin"
            self.sidebar.show_admin_navigation()
        elif view_name == "client":
            self.current_role = "client"
            self.sidebar.show_client_navigation()

        # Nếu vào lịch sử thì load data lại
        if view_name == "history":
            if hasattr(view, 'load_data'):
                view.load_data()

    def logout(self):
        """Đăng xuất người dùng hiện tại và quay về màn hình đăng nhập."""
        self.show_view("login")
