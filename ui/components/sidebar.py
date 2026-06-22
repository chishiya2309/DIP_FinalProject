import customtkinter as ctk

class Sidebar(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, width=280, corner_radius=0, **kwargs)
        
        self.grid_rowconfigure(10, weight=1) # Đẩy nút logout xuống đáy
        
        # Logo / Tiêu đề Trường
        self.lbl_school = ctk.CTkLabel(self, text="TRƯỜNG ĐẠI HỌC CÔNG NGHỆ KỸ THUẬT TP.HCM\nKHOA CNTT", 
                                       font=ctk.CTkFont(size=16, weight="bold"))
        self.lbl_school.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        # Tên Đề tài
        self.lbl_subject = ctk.CTkLabel(self, text="MÔN HỌC: XỬ LÝ ẢNH SỐ", font=ctk.CTkFont(size=14, weight="bold"))
        self.lbl_subject.grid(row=1, column=0, padx=20, pady=(5, 0))
        
        self.lbl_class = ctk.CTkLabel(self, text="Mã LHP: DIPR430685_06CLC", font=ctk.CTkFont(size=13))
        self.lbl_class.grid(row=2, column=0, padx=20, pady=(2, 0))
        
        self.lbl_instructor = ctk.CTkLabel(self, text="GVHD: PGS.TS. Hoàng Văn Dũng", font=ctk.CTkFont(size=13))
        self.lbl_instructor.grid(row=3, column=0, padx=20, pady=(2, 10))
        
        self.lbl_project = ctk.CTkLabel(self, text="HỆ THỐNG CẢNH BÁO\nTÉ NGÃ TỰ ĐỘNG CHO NGƯỜI CAO TUỔI", 
                                        font=ctk.CTkFont(size=18, weight="bold"),
                                        text_color="#3b8ed0")
        self.lbl_project.grid(row=4, column=0, padx=20, pady=(10, 20))
        
        # Thành viên nhóm
        self.lbl_team = ctk.CTkLabel(self, text="NHÓM 05", font=ctk.CTkFont(size=16, weight="bold"))
        self.lbl_team.grid(row=5, column=0, padx=20, pady=(10, 5))
        
        team_members = [
            "23110078 - Nguyễn Thái Bảo",
            "23110110 - Lê Quang Hưng",
            "23110111 - Lương Nguyễn Thành Hưng"
        ]
        
        for i, member in enumerate(team_members):
            lbl = ctk.CTkLabel(self, text=member, font=ctk.CTkFont(size=16))
            lbl.grid(row=6+i, column=0, padx=20, pady=5, sticky="w")
            
        # Navigation Buttons (Sẽ được ẩn/hiện tuỳ chế độ)
        self.btn_monitor = ctk.CTkButton(self, text="Giám sát camera", command=lambda: master.show_view("admin_monitor" if master.current_role == "admin" else "client"))
        self.btn_settings = ctk.CTkButton(self, text="Cài đặt hệ thống", command=lambda: master.show_view("admin_settings"))
        self.btn_history = ctk.CTkButton(self, text="Lịch sử cảnh báo", command=lambda: master.show_view("history"))
        
        # Logout button
        self.btn_logout = ctk.CTkButton(self, text="Đăng xuất", fg_color="transparent", 
                                        border_width=2, text_color=("gray10", "#DCE4EE"),
                                        command=lambda: master.logout())

    def dummy_cmd(self):
        pass

    def hide_navigation(self):
        self.btn_monitor.grid_forget()
        self.btn_settings.grid_forget()
        self.btn_history.grid_forget()
        self.btn_logout.grid_forget()

    def show_admin_navigation(self):
        self.btn_monitor.grid(row=11, column=0, padx=20, pady=10)
        self.btn_settings.grid(row=12, column=0, padx=20, pady=10)
        self.btn_history.grid(row=13, column=0, padx=20, pady=10)
        self.btn_logout.grid(row=14, column=0, padx=20, pady=(20, 20))

    def show_client_navigation(self):
        self.hide_navigation()
        self.btn_monitor.grid(row=11, column=0, padx=20, pady=10)
        self.btn_logout.grid(row=14, column=0, padx=20, pady=(20, 20))
