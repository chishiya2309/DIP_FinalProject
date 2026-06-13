import pickle
from pathlib import Path

def merge_datasets():
    # Đường dẫn tới 2 bộ dataset đã trích xuất
    multicam_dir = Path("data/processed/multiple_cameras_fall")
    urfall_dir = Path("data/processed/ur_fall_pose")
    output_dir = Path("data/processed/combined_custom_fall")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ["train_data.pkl", "val_data.pkl"]:
        print(f"--- Merging {split} ---")
        combined_data = []
        
        # Đọc bộ Multicam
        multicam_file = multicam_dir / split
        if multicam_file.exists():
            with open(multicam_file, "rb") as f:
                data = pickle.load(f)
                print(f"Loaded {len(data)} samples from {multicam_file}")
                combined_data.extend(data)
                
        # Đọc bộ UR Fall
        urfall_file = urfall_dir / split
        if urfall_file.exists():
            with open(urfall_file, "rb") as f:
                data = pickle.load(f)
                print(f"Loaded {len(data)} samples from {urfall_file}")
                combined_data.extend(data)
                
        # Lưu kết quả gộp
        output_file = output_dir / split
        with open(output_file, "wb") as f:
            pickle.dump(combined_data, f)
        
        print(f"Success! Saved {len(combined_data)} samples to {output_file}\n")

if __name__ == "__main__":
    merge_datasets()
