import customtkinter as ctk

class LoginView(ctk.CTkFrame):
    def __init__(self, master, controller, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.controller = controller
        
        self.grid_rowconfigure((0, 3), weight=1)
        self.grid_columnconfigure((0, 2), weight=1)
        
        self.login_frame = ctk.CTkFrame(self, corner_radius=15)
        self.login_frame.grid(row=1, column=1, padx=20, pady=20, ipadx=40, ipady=40)
        
        self.lbl_title = ctk.CTkLabel(self.login_frame, text="CHỌN CHẾ ĐỘ TRUY CẬP", font=ctk.CTkFont(size=24, weight="bold"))
        self.lbl_title.pack(pady=(20, 40))
        
        self.btn_user = ctk.CTkButton(self.login_frame, text="Truy cập User (Chỉ xem)", font=ctk.CTkFont(size=16),
                                      height=40, command=self.login_as_user)
        self.btn_user.pack(pady=10, fill="x", padx=40)
        
        self.lbl_or = ctk.CTkLabel(self.login_frame, text="-- Hoặc --")
        self.lbl_or.pack(pady=10)
        
        self.entry_password = ctk.CTkEntry(self.login_frame, placeholder_text="Mật khẩu Admin", show="*", height=40)
        self.entry_password.pack(pady=10, fill="x", padx=40)
        
        self.btn_admin = ctk.CTkButton(self.login_frame, text="Đăng nhập Admin", font=ctk.CTkFont(size=16),
                                       height=40, fg_color="#C850C0", hover_color="#8A2387", command=self.login_as_admin)
        self.btn_admin.pack(pady=10, fill="x", padx=40)
        
        self.lbl_error = ctk.CTkLabel(self.login_frame, text="", text_color="red")
        self.lbl_error.pack(pady=5)

    def login_as_user(self):
        self.entry_password.delete(0, 'end')
        self.lbl_error.configure(text="")
        self.controller.show_view("client")
        
    def login_as_admin(self):
        password = self.entry_password.get()
        if password == "admin123":
            self.entry_password.delete(0, 'end')
            self.lbl_error.configure(text="")
            self.controller.show_view("admin")
        else:
            self.lbl_error.configure(text="Mật khẩu không đúng!")
