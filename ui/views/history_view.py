import os
import csv
import customtkinter as ctk
from tkinter import ttk
from PIL import Image

class HistoryView(ctk.CTkFrame):
    def __init__(self, master, controller, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.controller = controller
        
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=1)
        
        # Top bar
        self.topbar = ctk.CTkFrame(self, height=50)
        self.topbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        self.lbl_title = ctk.CTkLabel(self.topbar, text="LỊCH SỬ CẢNH BÁO TÉ NGÃ", font=ctk.CTkFont(size=20, weight="bold"))
        self.lbl_title.pack(side="left", padx=20, pady=10)
        
        self.btn_refresh = ctk.CTkButton(self.topbar, text="Làm mới (Refresh)", command=self.load_data)
        self.btn_refresh.pack(side="right", padx=20, pady=10)
        
        # Table Area (Left)
        self.table_frame = ctk.CTkFrame(self)
        self.table_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        
        # Style cho Treeview (Vì CustomTkinter không có sẵn Table component)
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", 
                        background="#2b2b2b",
                        foreground="white",
                        rowheight=30,
                        fieldbackground="#2b2b2b",
                        bordercolor="#343638",
                        borderwidth=0)
        style.map('Treeview', background=[('selected', '#22559b')])
        style.configure("Treeview.Heading",
                        background="#565b5e",
                        foreground="white",
                        relief="flat")
        style.map("Treeview.Heading", background=[('active', '#3484F0')])

        # Tạo Treeview
        columns = ("Timestamp", "Camera", "TrackID", "Status", "Snapshot")
        self.tree = ttk.Treeview(self.table_frame, columns=columns, show="headings", style="Treeview")
        
        self.tree.heading("Timestamp", text="Thời gian")
        self.tree.heading("Camera", text="Nguồn Camera")
        self.tree.heading("TrackID", text="ID Nạn nhân")
        self.tree.heading("Status", text="Trạng thái")
        self.tree.heading("Snapshot", text="Đường dẫn Ảnh")
        
        self.tree.column("Timestamp", width=150)
        self.tree.column("Camera", width=150)
        self.tree.column("TrackID", width=100, anchor="center")
        self.tree.column("Status", width=100, anchor="center")
        self.tree.column("Snapshot", width=200) # Cột này có thể ẩn đi nếu muốn, nhưng để hiển thị cho rõ ràng
        
        self.tree.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        # Scrollbar cho Treeview
        scrollbar = ttk.Scrollbar(self.table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y", pady=10)
        
        # Bind sự kiện click
        self.tree.bind("<<TreeviewSelect>>", self.on_row_select)
        
        # Image Preview Area (Right)
        self.preview_frame = ctk.CTkFrame(self)
        self.preview_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        
        self.lbl_preview_title = ctk.CTkLabel(self.preview_frame, text="ẢNH BẰNG CHỨNG (SNAPSHOT)", font=ctk.CTkFont(size=16, weight="bold"))
        self.lbl_preview_title.pack(pady=(20, 10))
        
        self.lbl_image = ctk.CTkLabel(self.preview_frame, text="[Chọn một dòng để xem ảnh]")
        self.lbl_image.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Load dữ liệu lần đầu
        self.load_data()

    def load_data(self):
        # Xoá dữ liệu cũ
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        log_file = "fall_history.csv"
        if not os.path.exists(log_file):
            return
            
        try:
            with open(log_file, mode="r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None) # Bỏ qua header
                
                # Đọc ngược file để hiện cảnh báo mới nhất lên đầu
                rows = list(reader)
                for row in reversed(rows):
                    if not row: continue
                    # Đảm bảo đủ 5 cột (cho trường hợp file csv cũ chưa có cột Snapshot)
                    while len(row) < 5:
                        row.append("")
                    self.tree.insert("", "end", values=row)
        except Exception as e:
            print(f"[HistoryView] Lỗi đọc file CSV: {e}")

    def on_row_select(self, event):
        selected_items = self.tree.selection()
        if not selected_items:
            return
            
        item = selected_items[0]
        values = self.tree.item(item, "values")
        
        if len(values) >= 5:
            img_path = values[4]
            if img_path and os.path.exists(img_path):
                self.show_image(img_path)
            else:
                self.lbl_image.configure(image=None, text="[Không tìm thấy ảnh hoặc File CSV cũ]")
        else:
            self.lbl_image.configure(image=None, text="[Không có dữ liệu ảnh]")
            
    def show_image(self, path):
        try:
            img = Image.open(path)
            # Resize để vừa khung hình
            width, height = img.size
            max_size = 400
            
            if width > max_size or height > max_size:
                ratio = min(max_size / width, max_size / height)
                new_size = (int(width * ratio), int(height * ratio))
            else:
                new_size = (width, height)
                
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=new_size)
            self.lbl_image.configure(image=ctk_img, text="")
        except Exception as e:
            print(f"Lỗi hiển thị ảnh: {e}")
            self.lbl_image.configure(image=None, text="[Lỗi hiển thị ảnh]")
