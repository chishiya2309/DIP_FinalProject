import os
import urllib.request
import time

# Cấu hình đường dẫn lưu trữ
OUTPUT_DIR = os.path.join("data", "raw", "ur_fall_detection")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Base URL của UR Fall Detection Dataset (Đã cập nhật tên miền mới của trường)
BASE_URL = "http://fenix.ur.edu.pl/~mkepski/ds/data/"

def download_videos(prefix, total_count):
    """
    Tải hàng loạt video từ UR Fall Detection.
    prefix: 'fall' hoặc 'adl'
    total_count: số lượng video (ví dụ 30 fall, 40 adl)
    """
    success_count = 0
    for i in range(1, total_count + 1):
        # UR Fall thường cung cấp 2 góc camera: cam0 (nhìn thẳng) và cam1 (từ trên xuống)
        for cam in ['cam0', 'cam1']:
            filename = f"{prefix}-{i:02d}-{cam}.mp4"
            url = BASE_URL + filename
            out_path = os.path.join(OUTPUT_DIR, filename)
            
            # Bỏ qua nếu file đã tồn tại
            if os.path.exists(out_path):
                print(f"[ BỎ QUA ] {filename} đã tồn tại.")
                success_count += 1
                continue
                
            print(f"[*] Đang tải {filename}...")
            try:
                # Gửi request tải file
                urllib.request.urlretrieve(url, out_path)
                print(f"  -> [ THÀNH CÔNG ]")
                success_count += 1
            except Exception as e:
                # Báo lỗi nếu 404 (file không tồn tại trên server)
                print(f"  -> [ LỖI ] Không thể tải {filename} ({e})")
                
            # Nghỉ 0.5s để tránh làm quá tải server
            time.sleep(0.5)
            
    return success_count

if __name__ == "__main__":
    print("=" * 50)
    print("CÔNG CỤ TẢI TỰ ĐỘNG: UR FALL DETECTION DATASET")
    print("=" * 50)
    print(f"Thư mục đích: {os.path.abspath(OUTPUT_DIR)}\n")
    
    # Số lượng thực tế theo tài liệu UR Fall (30 falls, 40 ADLs)
    print("1. Đang tải dữ liệu TÉ NGÃ (Fall Sequences)...")
    fall_success = download_videos("fall", 30)
    
    print("\n2. Đang tải dữ liệu SINH HOẠT (ADL Sequences)...")
    adl_success = download_videos("adl", 40)
    
    print("=" * 50)
    print(f"HOÀN TẤT!")
    print(f"- Số file Fall: {fall_success}")
    print(f"- Số file ADL : {adl_success}")
    print("=" * 50)
